import cv2
import numpy as np
import os
import time
import csv
import threading
import winsound
import ctypes
import random
import requests
import pythoncom
import win32com.client
import io
import wave
import sounddevice as sd
from collections import deque
import tkinter as tk
from tkinter import simpledialog
import speech_recognition as sr

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES (RESOLUCIÓN AMPLIADA)
# =====================================================================
DATASET_DIR = "dataset"
TRAINER_FILE = "clasificador.yml"
ATTENDANCE_FILE = "asistencia.csv"
WIDTH_RESIZE = 200  # Resolución ampliada para capturar microfacciones
HEIGHT_RESIZE = 200

# Archivos del modelo de Red Neuronal Profunda (DNN) de OpenCV para Rostros
PROTO_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

# Asegurar que las carpetas base existan
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Hilo para la reproducción de voz
voz_lock = threading.Lock()
cola_saludos = deque(maxlen=5)

# Variable global para saber si el asistente está hablando físicamente
asistente_hablando = False

def hablar_worker():
    """Ejecuta en segundo plano la síntesis de voz usando SAPI5 nativo de Windows (win32com)."""
    global asistente_hablando
    pythoncom.CoInitialize()
    try:
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = -1  # Ritmo de habla más natural, pausado y humano
        voices = voice.GetVoices()
        for i in range(voices.Count):
            v_desc = voices.Item(i).GetDescription()
            if "spanish" in v_desc.lower() or "español" in v_desc.lower() or "es-ES" in v_desc.lower():
                voice.Voice = voices.Item(i)
                break
    except Exception as e:
        print(f"[Voz Error] No se pudo instanciar SAPI5: {e}")
        return

    while True:
        if cola_saludos:
            texto = cola_saludos.popleft()
            with voz_lock:
                try:
                    asistente_hablando = True
                    voice.Speak(texto, 0)
                    asistente_hablando = False
                except Exception as e:
                    print(f"[Voz Error] Error de reproducción SAPI5: {e}")
                    asistente_hablando = False
        else:
            time.sleep(0.1)

# Iniciar hilo de voz como demonio
thread_voz = threading.Thread(target=hablar_worker, daemon=True)
thread_voz.start()

def encolar_saludo(texto):
    """Encola un mensaje de voz si no está repetido recientemente."""
    if not cola_saludos or cola_saludos[-1] != texto:
        cola_saludos.append(texto)


# =====================================================================
# CLASE PRINCIPAL: RECONOCEDOR FACIAL INTELIGENTE CON APRENDIZAJE PASIVO Y LLM
# =====================================================================
class SistemaReconocimientoFacial:
    def __init__(self):
        # --- DESCARGA AUTOMÁTICA DEL MODELO DE DEEP LEARNING (RESNET-10 SSD) ---
        self.descargar_archivos_dnn()
        
        # Cargar red neuronal profunda de OpenCV
        self.net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
        
        # Ecualizador de contraste adaptable (CLAHE) para normalización de iluminación
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Inicializar el reconocedor LBPH de OpenCV con hiperparámetros optimizados y estables de memoria
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        
        self.modelo_cargado = False
        self.necesita_recargar_modelo = False
        self.nombres_usuarios = {}
        self.limpiar_dataset_ruido()
        self.cargar_modelo()
        
        # Guía de voz proactiva si la base de datos está completamente vacía
        if len(self.nombres_usuarios) == 0:
            def anunciar_bienvenida():
                time.sleep(2.5)  # Esperar a que la calibración de VAD finalice
                encolar_saludo("Hola, mi base de datos está vacía. Por favor, di tu nombre usando la frase me llamo, o presiona la tecla R para iniciar tu registro.")
            threading.Thread(target=anunciar_bienvenida, daemon=True).start()
        
        # Leer clave API de Groq del archivo local .env
        self.groq_key = self.cargar_groq_key()
        
        # Historial de detecciones para la estabilización
        self.memoria_deteccion = deque(maxlen=7)
        self.cara_detectada_nombre = None
        
        # --- SEGUIDOR POR CENTROIDES ---
        self.ultimo_centroide = None
        
        # --- MÁQUINA DE ESTADOS PARA EL REGISTRO DE VOZ (ALEXA STYLE) ---
        self.registro_estado = None
        self.registro_nombre = ""
        self.registro_timer = 0
        self.registro_fotos_guardadas = 0
        self.registro_paso = 1
        self.registro_paso_fotos_guardadas = 0
        self.registro_paso_hablado = False
        self.registro_paso_timer = 0.0
        
        # --- MÁQUINA DE ESTADOS DE CONVERSACIÓN ACTIVA (LLM CHAT) ---
        self.chat_estado = None
        self.chat_nombre = ""
        self.chat_timer = 0
        self.historial_conversacion = deque(maxlen=16)
        
        # Variables de control
        self.input_nombre_resultado = None
        self.consecutivos_desconocidos = 0
        self.fotos_auto_guardadas = 0
        self.ultimo_registro_asistencia = {}
        self.ultimo_chat_voz = {}
        self.last_mic_rms = 0.0
        self.texto_transcrito_pantalla = ""
        self.texto_timer_pantalla = 0.0
        
        # --- TRACKING DE ROSTROS Y CLASIFICACIÓN ASÍNCRONA ---
        self.tracker_rostros = {}
        self.next_face_id = 1
        self.predicciones_activas = {}
        self.ultimo_tiempo_clasificacion = {}
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.roi_box = None
        self.contador_frames = 0
        self.roi_fail_count = 0
        self.ultimo_tiempo_foto = 0.0

        # --- ESCUCHA CONTINUA EN SEGUNDO PLANO (RAM - CVE 2.0) ---
        self.umbral_silencio = 0.025  # Umbral base por defecto (calibrado en bucle_escucha_continua)
        self.stop_listener = False
        self.hilo_escucha = threading.Thread(target=self.bucle_escucha_continua, daemon=True)
        self.hilo_escucha.start()

    def descargar_archivos_dnn(self):
        """Descarga deploy.prototxt y caffemodel si no existen localmente."""
        if not os.path.exists(PROTO_PATH):
            print("[INFO] Descargando deploy.prototxt del detector de rostros DNN...")
            try:
                url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                r = requests.get(url, timeout=10)
                with open(PROTO_PATH, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"[ERROR] No se pudo descargar deploy.prototxt: {e}")
                
        if not os.path.exists(MODEL_PATH):
            print("[INFO] Descargando res10_300x300_ssd_iter_140000.caffemodel (5.3 MB)...")
            try:
                url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                r = requests.get(url, timeout=15)
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"[ERROR] No se pudo descargar caffemodel: {e}")

    def cargar_groq_key(self):
        """Carga la clave API de Groq desde el archivo local .env."""
        if os.path.exists(".env"):
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("GROQ_API_KEY="):
                            key = line.split("GROQ_API_KEY=")[1].strip()
                            print("[INFO] Clave de API de Groq cargada del archivo local .env")
                            return key
            except Exception as e:
                print(f"[WARN] No se pudo leer .env: {e}")
        return None

    def obtener_mejor_dispositivo_entrada(self):
        """Busca el mejor dispositivo de entrada de audio físico disponible, evitando micrófonos virtuales como Oculus."""
        try:
            devices = sd.query_devices()
            default_in = sd.default.device[0]
            
            default_name = devices[default_in]['name'].lower() if default_in >= 0 else ""
            
            # Si el micrófono por defecto es el de Oculus (VR) o no hay dispositivo por defecto
            if "oculus" in default_name or default_in < 0:
                print("[AUDIO DIAGNÓSTICO] El micrófono por defecto es Oculus (Virtual). Buscando micrófono físico...")
                # Buscar alternativas físicas como Realtek, Intel, Smart Sound, Micrófono
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        name = dev['name'].lower()
                        # Excluir dispositivos virtuales conocidos y mezclas estéreo
                        if "oculus" not in name and "mezcla" not in name and "stereo mix" not in name:
                            if "realtek" in name or "intel" in name or "micr" in name or "smart" in name:
                                print(f"[AUDIO] Seleccionado micrófono físico automáticamente: {dev['name']} (ID: {i})")
                                return i
                
                # Segunda opción: primer micrófono de entrada disponible que no sea Oculus
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        name = dev['name'].lower()
                        if "oculus" not in name:
                            print(f"[AUDIO] Seleccionado dispositivo genérico: {dev['name']} (ID: {i})")
                            return i
            
            return default_in
        except Exception as e:
            print(f"[AUDIO ERROR] Error al buscar dispositivo de entrada: {e}")
            return None

    def generar_frase_llm(self, tipo, nombre=None, texto_conversacion=None):
        """Genera una frase contextual ingeniosa usando Groq Llama-3 con alta personalidad."""
        if not self.groq_key:
            if tipo == "chat":
                return "No tengo internet, pero me gusta escucharte de todos modos."
            return self.generar_frase_local(tipo, nombre)
            
        hora_actual = time.strftime("%H:%M")
        fecha_actual = time.strftime("%Y-%m-%d")
        total_usuarios = len(self.nombres_usuarios)
        
        persona = nombre if nombre else "un extraño"
        system_content = (
            f"Eres Alexa, una asistente virtual con una personalidad sumamente carismática, alegre, ingeniosa, divertida y muy ocurrente en español. "
            f"Hablas de forma fluida, cálida y muy natural (como un amigo ocurrente, nunca como un robot rígido). "
            f"Interactúas con {persona}. Datos del entorno: Hora local: {hora_actual}, Fecha: {fecha_actual}, Usuarios registrados: {total_usuarios}. "
            f"REGLAS CRÍTICAS:\n"
            f"1. Responde de forma concisa pero con ritmo natural (entre 10 y 28 palabras). No hables en telegramas de 4 palabras ni frases eternas.\n"
            f"2. Varía al máximo tus respuestas, evita preguntas o saludos idénticos.\n"
            f"3. Prohibido usar markdown, asteriscos, comillas o viñetas. Escribe texto plano limpio para que el sintetizador de voz lo lea de forma fluida."
        )
        
        if tipo == "chat" and texto_conversacion:
            self.historial_conversacion.append({"role": "user", "content": texto_conversacion})
            messages = [{"role": "system", "content": system_content}] + list(self.historial_conversacion)
        else:
            if tipo == "saludo_conocido":
                prompt = (
                    f"Genera un saludo muy amigable, simpático e ingenioso para {nombre} de máximo 18 palabras. "
                    f"Evita decir siempre lo mismo, haz un comentario breve sobre el día, el clima o que al fin lo ves por cámara."
                )
            elif tipo == "saludo_nuevo":
                prompt = "Dile a la persona en cámara que no la conoces de una forma súper carismática, pidiéndole su nombre en máximo 18 palabras."
            elif tipo == "durante_registro":
                prompt = f"Dile a {nombre} de forma divertida y alegre que se mueva ligeramente o sonría mientras guardas sus fotos, máximo 18 palabras."
            elif tipo == "registro_completo":
                prompt = f"Dale la bienvenida oficial a {nombre} a tu memoria con una frase feliz, ingeniosa y de máximo 18 palabras."
            else:
                return self.generar_frase_local(tipo, nombre)
                
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": 100,  # Aumentado para frases completas y fluidas
                "temperature": 0.95  # Alta temperatura para mayor creatividad y variación léxica
            }
            res = requests.post(url, headers=headers, json=data, timeout=3.5)
            if res.status_code == 200:
                result = res.json()
                frase = result["choices"][0]["message"]["content"].strip().replace('"', '')
                if tipo == "chat":
                    self.historial_conversacion.append({"role": "assistant", "content": frase})
                return frase
            else:
                print(f"[Groq API Error] HTTP {res.status_code}: {res.text}")
        except Exception as e:
            print(f"[Groq LLM Error] Fallback local activo: {e}")
            
        if tipo == "chat":
            return "Comprendo lo que dices. Cuéntame más sobre eso."
        return self.generar_frase_local(tipo, nombre)

    def encolar_saludo_groq(self, tipo, nombre=None):
        """Genera y encola un saludo usando Groq de forma asíncrona para no congelar el video."""
        def run():
            frase = self.generar_frase_llm(tipo, nombre)
            encolar_saludo(frase)
        threading.Thread(target=run, daemon=True).start()

    def generar_frase_local(self, tipo, nombre=None):
        """Generador local aleatorio de reserva con variedad de frases."""
        opciones = {
            "saludo_conocido": [
                f"¡Hola {nombre}! Qué alegría volver a verte en pantalla.",
                f"Vaya, {nombre}, me alegra que pases a saludar hoy.",
                f"¡Qué tal, {nombre}! Te estaba esperando para charlar.",
                f"Hola {nombre}, te veo perfectamente. ¿Cómo va tu día?"
            ],
            "saludo_nuevo": [
                "Vaya, una cara nueva por aquí. ¿Cuál es tu nombre, amigo?",
                "Hola, no reconozco tus facciones todavía. ¿Cómo te llamas?",
                "¡Hey! No te tengo en mi base de datos. Dime tu nombre para registrarte.",
                "Hola, ¿quién anda ahí? Dime tu nombre para agregarte."
            ],
            "durante_registro": [
                f"Perfecto {nombre}, muévete un poquito mientras guardo tus fotos.",
                f"Muy bien {nombre}, te estoy analizando de cerca, sonríe a la cámara.",
                f"Entendido {nombre}, mira fijamente unos segundos más para entrenar mi red."
            ],
            "registro_completo": [
                f"¡Listo {nombre}! Ya te registré con éxito en mi base de datos.",
                f"Registro completado {nombre}. Ahora eres parte oficial de mi memoria.",
                f"Perfecto {nombre}, ya sé quién eres. ¡Bienvenido al sistema!"
            ]
        }
        if tipo in opciones:
            return random.choice(opciones[tipo])
        return ""

    def cargar_modelo(self):
        """Carga el clasificador LBPH y actualiza el mapeo de nombres."""
        if os.path.exists(TRAINER_FILE):
            try:
                self.recognizer.read(TRAINER_FILE)
                self.modelo_cargado = True
                print("[INFO] Modelo clasificador.yml cargado correctamente.")
            except Exception as e:
                print(f"[ERROR] Error al leer clasificador.yml: {e}. Se requiere reentrenar.")
                self.modelo_cargado = False
        else:
            print("[INFO] No se encontró un modelo entrenado previamente.")
            self.modelo_cargado = False
            
        self.actualizar_mapeo_nombres()

    def actualizar_mapeo_nombres(self):
        """Asigna un ID numérico a cada carpeta de usuario de forma alfabética."""
        if not os.path.exists(DATASET_DIR):
            return
        subdirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        self.nombres_usuarios = {i: name for i, name in enumerate(subdirs)}

    def limpiar_dataset_ruido(self):
        """Limpia carpetas del dataset que correspondan a falsas activaciones por ruido de voz."""
        if not os.path.exists(DATASET_DIR):
            return
        nombres_ruido = {"de", "un", "gracias", "ok", "si", "no", "hola", "el", "la", "en", "con", "por", "para", "que"}
        for d in os.listdir(DATASET_DIR):
            ruta_d = os.path.join(DATASET_DIR, d)
            if os.path.isdir(ruta_d):
                if d.lower() in nombres_ruido or len(d) < 2:
                    print(f"[SISTEMA] Limpiando carpeta de ruido detectada en dataset: {d}")
                    try:
                        import shutil
                        shutil.rmtree(ruta_d)
                    except Exception as e:
                        print(f"[ERROR] No se pudo limpiar la carpeta {d}: {e}")

    def alinear_rostro_ojos(self, rostro_gris):
        """Detecta los ojos usando Haar Cascades y rota la imagen para alinearlos horizontalmente."""
        ojos = self.eye_cascade.detectMultiScale(rostro_gris, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
        if len(ojos) >= 2:
            # Ordenar ojos por posición X para identificar el izquierdo y el derecho
            ojos = sorted(ojos, key=lambda o: o[0])
            ojo_izq, ojo_der = ojos[0], ojos[1]
            
            # Centro de cada ojo
            izq_x = ojo_izq[0] + ojo_izq[2] // 2
            izq_y = ojo_izq[1] + ojo_izq[3] // 2
            der_x = ojo_der[0] + ojo_der[2] // 2
            der_y = ojo_der[1] + ojo_der[3] // 2
            
            # Calcular ángulo
            dy = der_y - izq_y
            dx = der_x - izq_x
            angulo = np.degrees(np.arctan2(dy, dx))
            
            # Centro de rotación (OpenCV requiere flotantes en Point2f)
            centro = (float((izq_x + der_x) / 2.0), float((izq_y + der_y) / 2.0))
            
            # Rotar
            h, w = rostro_gris.shape
            M = cv2.getRotationMatrix2D(centro, angulo, scale=1.0)
            rostro_rotado = cv2.warpAffine(rostro_gris, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rostro_rotado
        return rostro_gris

    def preprocesar_rostro_extremo(self, rostro_gris):
        """Aplica un pipeline avanzado para maximizar la nitidez de facciones."""
        # 1. Alinear ojos si es posible
        rostro_alineado = self.alinear_rostro_ojos(rostro_gris)
        
        h, w = rostro_alineado.shape
        margin_x = int(w * 0.08)
        margin_y = int(h * 0.08)
        
        cara_recortada = rostro_alineado[margin_y:h-margin_y, margin_x:w-margin_x]
        cara_suave = cv2.bilateralFilter(cara_recortada, d=5, sigmaColor=50, sigmaSpace=50)
        cara_ecualizada = self.clahe.apply(cara_suave)
        
        return cv2.resize(cara_ecualizada, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)

    def clasificar_rostro_async(self, face_id, cara_gris_crop):
        """Clasifica un rostro recortado en segundo plano de manera asíncrona para evitar tirones."""
        if not self.modelo_cargado:
            return
            
        try:
            # 1. Preprocesar (incluye alineación afín de ojos)
            cara_gris_norm = self.preprocesar_rostro_extremo(cara_gris_crop)
            
            # 2. Inferencia LBPH
            id_prediccion, distancia = self.recognizer.predict(cara_gris_norm)
            
            # 3. Mapear distancia a confianza biométrica
            if distancia < 40:
                confianza_pct = 95.0 + (40 - distancia) * 0.125
            elif distancia < 52:
                confianza_pct = 80.0 + (52 - distancia) * 1.25
            elif distancia < 70:
                confianza_pct = 40.0 + (70 - distancia) * 2.22
            else:
                confianza_pct = max(0.0, 40.0 - (distancia - 70) * 2.0)
                
            # 4. Asignar nombre según umbral estricto
            if distancia < 52:
                nombre = self.nombres_usuarios.get(id_prediccion, "Desconocido")
            else:
                nombre = "Desconocido"
                
            # 5. Guardar predicción
            self.predicciones_activas[face_id] = {
                "nombre": nombre,
                "confianza": confianza_pct,
                "distancia": distancia,
                "tiempo": time.time()
            }
            
            # 6. Auto-aprendizaje incremental en caliente (Online updates)
            if nombre != "Desconocido" and distancia < 35.0:
                try:
                    self.recognizer.update([cara_gris_norm], np.array([id_prediccion]))
                except:
                    pass
                    
        except Exception as e:
            print(f"[Async Classification Error] {e}")

    def entrenar_modelo(self):
        """Entrena el modelo LBPH usando procesamiento avanzado y thread-safety."""
        print("[IA ENTRENAMIENTO] Iniciando entrenamiento en segundo plano...")
        rostros = []
        ids = []
        
        if not os.path.exists(DATASET_DIR):
            return False
        subdirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        temp_nombres_usuarios = {i: name for i, name in enumerate(subdirs)}
        nombre_a_id = {v: k for k, v in temp_nombres_usuarios.items()}
        
        for nombre_usuario in os.listdir(DATASET_DIR):
            ruta_usuario = os.path.join(DATASET_DIR, nombre_usuario)
            if not os.path.isdir(ruta_usuario):
                continue
                
            usuario_id = nombre_a_id[nombre_usuario]
            
            for archivo_imagen in os.listdir(ruta_usuario):
                ruta_imagen = os.path.join(ruta_usuario, archivo_imagen)
                if not (archivo_imagen.endswith('.jpg') or archivo_imagen.endswith('.png')):
                    continue
                    
                img_gris = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
                if img_gris is not None:
                    # Las imágenes del dataset ya están totalmente optimizadas, recortadas y ecualizadas al guardarse
                    rostros.append(img_gris)
                    ids.append(usuario_id)
        
        if len(rostros) == 0:
            return False
            
        try:
            temp_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8
            )
            temp_recognizer.train(rostros, np.array(ids))
            temp_recognizer.write(TRAINER_FILE + ".tmp")
            os.replace(TRAINER_FILE + ".tmp", TRAINER_FILE)
            
            self.necesita_recargar_modelo = True
            print("[IA ENTRENAMIENTO] Entrenamiento finalizado. Pendiente de recarga segura.")
            return True
        except Exception as e:
            print(f"[IA ERROR] Error al entrenar en segundo plano: {e}")
            return False

    def entrenar_en_segundo_plano(self):
        """Ejecuta el entrenamiento en un hilo para evitar congelamientos."""
        hilo = threading.Thread(target=self.entrenar_modelo, daemon=True)
        hilo.start()

    def array_to_wav_bytes(self, audio_data):
        """
        Convierte datos NumPy float32 a un archivo WAV PCM en memoria (RAM).
        Aplica Normalización automática de ganancia (AGC) para amplificar voces bajas.
        """
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            
            # --- MEJORA AUDIO: Normalización de Ganancia Automática (AGC) ---
            max_val = np.max(np.abs(audio_data))
            if max_val > 1e-5:
                audio_data = audio_data / max_val
                
            audio_int = (audio_data * 32767.0).astype(np.int16)
            wav_file.writeframes(audio_int.tobytes())
        wav_buffer.seek(0)
        return wav_buffer

    def transcribir_groq_whisper(self, wav_bytes):
        """Transcribe el audio en memoria usando Groq Whisper v3."""
        if not self.groq_key:
            return ""
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {self.groq_key}"}
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            data = {"model": "whisper-large-v3", "language": "es"}
            
            res = requests.post(url, headers=headers, files=files, data=data, timeout=3.5)
            if res.status_code == 200:
                return res.json().get("text", "").strip()
        except Exception as e:
            print(f"[Whisper API Error] {e}")
        return ""

    def detener_habla(self):
        """Detiene de inmediato cualquier reproducción de voz activa (Barge-in)."""
        global asistente_hablando
        try:
            pythoncom.CoInitialize()
            voice_killer = win32com.client.Dispatch("SAPI.SpVoice")
            voice_killer.Speak("", 3)
            cola_saludos.clear()
            asistente_hablando = False
            print("[Barge-in] Reproducción de voz de la IA interrumpida por el usuario.")
        except Exception as e:
            print(f"[Barge-in Error] {e}")

    def calibrar_microfono(self, stream, chunk_size):
        """Escucha el ruido ambiental durante 1.5 segundos para autocalibrar el umbral VAD."""
        print("[SISTEMA VOZ] Calibrando micrófono... Por favor, guarda silencio.")
        lecturas = []
        for _ in range(15):
            try:
                data, overflow = stream.read(chunk_size)
                rms = np.sqrt(np.mean(data**2))
                lecturas.append(rms)
            except:
                pass
        
        if lecturas:
            avg_noise = np.mean(lecturas)
            self.umbral_silencio = max(0.015, min(avg_noise * 1.6, 0.065))
            print(f"[SISTEMA VOZ] Calibración completa. Umbral VAD establecido en: {self.umbral_silencio:.4f}")
        else:
            self.umbral_silencio = 0.025
            print("[SISTEMA VOZ] Falló lectura de calibración. Usando umbral por defecto (0.025)")

    def bucle_escucha_continua(self):
        """
        Bucle infinito en segundo plano (Siempre escuchando en memoria RAM - CVE 2.0).
        Realiza calibración automática de ruido.
        """
        sample_rate = 16000
        chunk_size = int(sample_rate * 0.1)  # Bloques de 100ms
        
        time.sleep(1.0)
        pythoncom.CoInitialize()
        
        try:
            id_mic = self.obtener_mejor_dispositivo_entrada()
            devices = sd.query_devices()
            mic_name = devices[id_mic]['name'] if id_mic >= 0 else "Ninguno"
            print(f"[SISTEMA VOZ] Usando micrófono: ID {id_mic} - {mic_name}")
            with sd.InputStream(device=id_mic, samplerate=sample_rate, channels=1, blocksize=chunk_size) as stream:
                self.calibrar_microfono(stream, chunk_size)
                
                while not self.stop_listener:
                    if asistente_hablando:
                        try:
                            data, overflow = stream.read(chunk_size)
                            rms = np.sqrt(np.mean(data**2))
                            if rms > (self.umbral_silencio * 3.5):
                                self.detener_habla()
                        except:
                            pass
                        time.sleep(0.05)
                        continue
                        
                    if self.registro_estado == "esperando_popup":
                        time.sleep(0.5)
                        continue
                        
                    try:
                        data, overflow = stream.read(chunk_size)
                        rms = np.sqrt(np.mean(data**2))
                        self.last_mic_rms = float(rms)
                        
                        if rms > self.umbral_silencio:
                            print(f"[VAD] Habla detectada. Grabando...")
                            audio_chunks = [data]
                            silencio_consecutivo = 0
                            
                            while len(audio_chunks) < 100 and not self.stop_listener:
                                if asistente_hablando:
                                    break
                                    
                                chunk, ov = stream.read(chunk_size)
                                audio_chunks.append(chunk)
                                
                                c_rms = np.sqrt(np.mean(chunk**2))
                                self.last_mic_rms = float(c_rms)
                                
                                if c_rms > self.umbral_silencio:
                                    silencio_consecutivo = 0
                                else:
                                    silencio_consecutivo += 1
                                    
                                if silencio_consecutivo >= 8:
                                    break
                            
                            if asistente_hablando:
                                continue
                                
                            audio_data = np.concatenate(audio_chunks, axis=0)
                            wav_bytes = self.array_to_wav_bytes(audio_data)
                            
                            texto_entendido = self.transcribir_groq_whisper(wav_bytes)
                            if texto_entendido:
                                print(f"[Whisper Transcripción] Entendido: '{texto_entendido}'")
                                self.texto_transcrito_pantalla = texto_entendido
                                self.texto_timer_pantalla = time.time()
                                self.procesar_voz_entrada_pasiva(texto_entendido)
                    except Exception as e:
                        print(f"[Audio Loop Inner Error] {e}")
                        time.sleep(0.1)
                        
                    time.sleep(0.02)
        except Exception as e:
            print(f"[Audio Stream Error] No se pudo iniciar el dispositivo de entrada de micrófono: {e}")

    def procesar_voz_entrada_pasiva(self, texto):
        """Procesa el texto capturado por la escucha continua en segundo plano."""
        texto_limpio = texto.strip()
        if not texto_limpio:
            return
            
        print(f"[VOZ A TEXTO] Transcripción recibida: '{texto_limpio}'")
        texto_lower = texto_limpio.lower()
        
        # A. Si ya está escuchando activamente el nombre del usuario durante el registro guiado
        if self.registro_estado == "escuchando":
            palabras = texto_limpio.split()
            nombre = None
            if palabras:
                # Comprobar si dice "me llamo X" o "mi nombre es X"
                for idx, pal in enumerate(palabras):
                    if pal.lower() in ["llamo", "es"] and idx + 1 < len(palabras):
                        nombre = palabras[idx+1]
                        break
                if not nombre:
                    nombre = palabras[0]
            if nombre:
                nombre = ''.join(c for c in nombre if c.isalnum()).capitalize()
                
            # Validar nombre para evitar pronombres o ruidos ("Mi", "Yo", etc.)
            blacklist_nombres = {"mi", "me", "yo", "el", "la", "tu", "su", "un", "de", "no", "si", "hola", "ok", "nombre", "llamo", "es", "este", "esta"}
            if not nombre or nombre.lower() in blacklist_nombres or len(nombre) < 3:
                print(f"[REGISTRO] Nombre inválido o ruido rechazado: '{nombre}'")
                self.registro_estado = "escuchando"  # Permanecer en escucha
                encolar_saludo("No logré entender tu nombre claramente. ¿Podrías repetirlo, por favor?")
                return
                
            self.input_nombre_resultado = nombre
            self.registro_estado = "procesando_voz"
            return
            
        # B. Comprobar comandos globales de registro (siempre tienen prioridad alta)
        # 1. Comando de registro directo: "Me llamo Carlos" o "Mi nombre es Maria"
        if "me llamo " in texto_lower or "mi nombre es " in texto_lower:
            palabras = texto_limpio.split()
            nombre = None
            for idx, pal in enumerate(palabras):
                if pal.lower() in ["llamo", "es"] and idx + 1 < len(palabras):
                    nombre = palabras[idx+1]
                    break
            if nombre:
                nombre = ''.join(c for c in nombre if c.isalnum()).capitalize()
                blacklist_nombres = {"mi", "me", "yo", "el", "la", "tu", "su", "un", "de", "no", "si", "hola", "ok", "nombre", "llamo", "es", "este", "esta"}
                if nombre.lower() not in blacklist_nombres and len(nombre) >= 3:
                    # Cerrar chat previo si estaba activo
                    self.chat_estado = None
                    self.registro_nombre = nombre
                    self.registro_estado = "capturando_dinamico"
                    self.registro_timer = time.time()
                    self.registro_fotos_guardadas = 0
                    self.registro_paso = 1
                    self.registro_paso_fotos_guardadas = 0
                    self.registro_paso_hablado = False
                    self.registro_paso_timer = time.time()
                    self.encolar_saludo_groq("durante_registro", nombre)
                    print(f"[AUTO-REGISTRO] Asistente iniciado directamente vía voz para '{nombre}'")
                    return
        
        # 2. Comando de registro guiado: "Registrar", "Nuevo usuario", etc.
        elif any(cmd in texto_lower for cmd in ["registrar", "registro", "nuevo usuario", "agregar usuario"]):
            print("[VOZ] Comando de inicio de registro detectado.")
            # Cerrar chat previo si estaba activo
            self.chat_estado = None
            self.iniciar_registro_autonomo()
            return

        # C. Si ya hay un chat conversacional activo
        if self.chat_estado is not None:
            print("[FLUJO IA] Enviando texto a la conversación activa con Groq...")
            self.chat_estado = "procesando_usuario"
            threading.Thread(target=self.procesar_charla_worker, args=(texto_limpio,), daemon=True).start()
            
        # D. Si el sistema está en reposo (buscando rostros)
        else:
            nombre = "Extraño"
            if self.cara_detectada_nombre and self.cara_detectada_nombre != "Desconocido":
                nombre = self.cara_detectada_nombre
                
            self.chat_nombre = nombre
            self.chat_estado = "procesando_usuario"
            self.historial_conversacion.clear()
            
            print(f"[CONVERSACIÓN INICIADA] Canalizando texto '{texto_limpio}' hacia la IA para {nombre}...")
            self.ultimo_chat_voz[nombre] = time.time()
            
            threading.Thread(target=self.procesar_charla_worker, args=(texto_limpio,), daemon=True).start()

    def solicitar_nombre_popup_worker(self):
        """Diálogo de Tkinter en segundo plano."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        nombre = simpledialog.askstring("Identificación Inteligente", "Escribe tu nombre por favor:", parent=root)
        root.destroy()
        
        if nombre:
            self.input_nombre_resultado = name = nombre.strip().replace(" ", "_").capitalize()
        else:
            self.input_nombre_resultado = ""
        self.registro_estado = "procesando_voz"

    def iniciar_registro_autonomo(self):
        """Inicia el estado de registro guiado de forma humanizada llamando a Groq."""
        self.registro_estado = "preguntando"
        self.registro_timer = time.time()
        self.registro_fotos_guardadas = 0
        self.registro_paso = 1
        self.registro_paso_fotos_guardadas = 0
        self.registro_paso_hablado = False
        self.registro_paso_timer = 0.0
        self.input_nombre_resultado = None
        self.encolar_saludo_groq("saludo_nuevo")

    def iniciar_charla_conversacional(self, nombre):
        """Inicia el bucle de conversación inteligente con Llama-3."""
        self.chat_estado = "iniciando_chat"
        self.chat_nombre = nombre
        self.chat_timer = time.time()
        self.historial_conversacion.clear()
        
        saludo_inicial = self.generar_frase_llm("saludo_conocido", nombre)
        self.historial_conversacion.append({"role": "assistant", "content": saludo_inicial})
        encolar_saludo(saludo_inicial)
        print(f"[CONVERSACIÓN] Alexa: {saludo_inicial}")

    def procesar_charla_worker(self, texto_usuario):
        """Genera respuesta conversacional con Groq en segundo plano y la encola."""
        respuesta = self.generar_frase_llm("chat", self.chat_nombre, texto_usuario)
        print(f"[CONVERSACIÓN] Alexa: {respuesta}")
        encolar_saludo(respuesta)
        
        palabras_cierre = ["adiós", "adios", "chao", "bye", "salir", "nos vemos", "gracias"]
        cerrar_conversacion = False
        for pc in palabras_cierre:
            if pc in texto_usuario.lower():
                cerrar_conversacion = True
                break
                
        if cerrar_conversacion:
            self.chat_estado = None
        else:
            self.chat_timer = time.time()
            self.chat_estado = "esperando_habla"

    def guardar_asistencia(self, nombre, confianza):
        """Registra la asistencia en un archivo CSV aplicando cooldown."""
        ahora = time.time()
        if nombre in self.ultimo_registro_asistencia:
            if ahora - self.ultimo_registro_asistencia[nombre] < 15:
                return
                
        self.ultimo_registro_asistencia[nombre] = ahora
        fecha_actual = time.strftime("%Y-%m-%d")
        hora_actual = time.strftime("%H:%M:%S")
        
        archivo_nuevo = not os.path.exists(ATTENDANCE_FILE)
        try:
            with open(ATTENDANCE_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if archivo_nuevo:
                    writer.writerow(["Nombre", "Fecha", "Hora", "Confianza (%)"])
                writer.writerow([nombre, fecha_actual, hora_actual, f"{confianza:.2f}%"])
            print(f"[CSV LOG] Asistencia registrada para {nombre} (Match: {confianza:.1f}%)")
        except Exception as e:
            print(f"[CSV ERROR] No se pudo guardar en CSV: {e}")

    def auto_capturar_rostro_interaccion(self, nombre, rostro_gris, w_face):
        """Muestreo selectivo interactivo para asimilar variaciones de distancia."""
        ruta_usuario = os.path.join(DATASET_DIR, nombre)
        if not os.path.exists(ruta_usuario):
            return
            
        total_fotos = len([f for f in os.listdir(ruta_usuario) if f.endswith(".jpg")])
        if total_fotos >= 30: # Límite de aprendizaje pasivo
            return
            
        rostro_opt = self.preprocesar_rostro_extremo(rostro_gris)
        archivo_nombre = f"{nombre}_auto_{total_fotos}.jpg"
        archivo_completo = os.path.join(ruta_usuario, archivo_nombre)
        # Escribir de forma asíncrona en segundo plano para no congelar el frame
        threading.Thread(target=cv2.imwrite, args=(archivo_completo, rostro_opt), daemon=True).start()
        
        self.fotos_auto_guardadas += 1
        print(f"[APRENDIZAJE ONLINE] Captura de pose pasiva ({total_fotos + 1}/30) para '{nombre}'")
        
        if self.fotos_auto_guardadas >= 8:
            self.fotos_auto_guardadas = 0
            self.entrenar_en_segundo_plano()

    def procesar_captura_pasiva(self, rostro_gris):
        """Guarda silenciosa e implícitamente las caras aplicando normalización extrema y asíncrona por pasos/poses."""
        if self.registro_nombre is None or self.registro_estado != "capturando_dinamico":
            return
            
        # Si la IA está hablando (dando instrucciones), pausar la captura temporalmente
        if asistente_hablando:
            return
            
        ahora = time.time()
        
        # Controlar la velocidad de captura (mínimo 400ms entre fotos de la misma pose para dar tiempo a micro-variaciones)
        if ahora - self.ultimo_tiempo_foto < 0.40:
            return
            
        self.ultimo_tiempo_foto = ahora
        
        ruta_usuario = os.path.join(DATASET_DIR, self.registro_nombre)
        if not os.path.exists(ruta_usuario):
            os.makedirs(ruta_usuario)
            
        # Limpiar al empezar si es el primer paso y la primera foto
        if self.registro_fotos_guardadas == 0:
            for f in os.listdir(ruta_usuario):
                if f.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(ruta_usuario, f))
                    except:
                        pass
                        
        rostro_opt = self.preprocesar_rostro_extremo(rostro_gris)
        
        self.registro_fotos_guardadas += 1
        self.registro_paso_fotos_guardadas += 1
        
        archivo = os.path.join(ruta_usuario, f"{self.registro_nombre}_{self.registro_fotos_guardadas}.jpg")
        # Escritura asíncrona en disco en un hilo daemon
        threading.Thread(target=cv2.imwrite, args=(archivo, rostro_opt), daemon=True).start()
        
        print(f"[REGISTRO PASIVO] Paso {self.registro_paso}/5 - Foto {self.registro_paso_fotos_guardadas}/3 Guardada ({self.registro_fotos_guardadas}/15)")
        
        # Avanzar de paso si ya capturamos 3 imágenes en este paso
        if self.registro_paso_fotos_guardadas >= 3 and self.registro_paso < 5:
            self.registro_paso += 1
            self.registro_paso_fotos_guardadas = 0
            self.registro_paso_hablado = False

    def dibujar_hud_futurista(self, frame, x, y, w, h, etiqueta, subtitulo, color):
        """Dibuja brackets esquineros, barra de fondo y escáner sobre el rostro."""
        longitud_linea = int(w * 0.2)
        grosor = 3
        
        cv2.line(frame, (x, y), (x + longitud_linea, y), color, gromosor := grosor)
        cv2.line(frame, (x, y), (x, y + longitud_linea), color, gromosor)
        cv2.line(frame, (x + w, y), (x + w - longitud_linea, y), color, gromosor)
        cv2.line(frame, (x + w, y), (x + w, y + longitud_linea), color, gromosor)
        cv2.line(frame, (x, y + h), (x + longitud_linea, y + h), color, gromosor)
        cv2.line(frame, (x, y + h), (x, y + h - longitud_linea), color, gromosor)
        cv2.line(frame, (x + w, y + h), (x + w - longitud_linea, y + h), color, gromosor)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - longitud_linea), color, gromosor)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 1)
        velocidad_escaner = int((time.time() * 250) % h)
        cv2.line(overlay, (x, y + velocidad_escaner), (x + w, y + velocidad_escaner), color, 2)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Etiqueta
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
        cv2.putText(frame, etiqueta, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, subtitulo, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def iniciar_bucle_principal(self):
        """Inicia el streaming de la cámara y el reconocimiento facial."""
        print("[SISTEMA] Inicializando captura de video...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CRÍTICO] No se pudo acceder a la cámara. Verifique la conexión.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps = 0
        ultimo_guardado_interactivo = 0
        
        print("\nSISTEMA INTELIGENTE DE RECONOCIMIENTO FACIAL INICIADO (OPENCV DEEP LEARNING & WHISPER).")
        print("Modo Conversacional y Autónomo Activo:")
        print("  - Detección de Rostros: ResNet-10 SSD (Red Neuronal Profunda - Rango completo hasta 5m).")
        print("  - Audio: Calibración dinámica de ruido ambiental y ganancia automática AGC.")
        print("  - Presione 'Q' en la pantalla del video para salir.")
        print("-" * 60)
        
        while True:
            t_inicio = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Error al recibir fotograma de la cámara. Reintentando...")
                time.sleep(0.1)
                continue
                
            frame_original = frame.copy()
            h_orig, w_orig = frame.shape[:2]
            
            # --- OPTIMIZACIÓN DE FPS ---
            escala = 2
            ancho_red = w_orig // escala
            alto_red = h_orig // escala
            frame_pequeno = cv2.resize(frame, (ancho_red, alto_red))
            
            # --- RECARGA DE MODELO SEGURA EN CALIENTE ---
            if self.necesita_recargar_modelo:
                self.cargar_modelo()
                self.necesita_recargar_modelo = False
                print("[HOT-RELOAD] El clasificador facial se ha actualizado de forma segura en caliente.")
            
            # --- DETECCIÓN DE ROSTROS OPTIMIZADA CON FRAME SKIPPING Y DYNAMIC RoI ---
            self.contador_frames += 1
            gris = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)
            gris_opt = gris
            
            caras_combinadas = []
            
            # Decidir si escaneamos RoI local o frame completo
            escanear_completo = True
            if self.roi_box is not None and self.contador_frames % 2 != 0:
                # Intentar escanear solo el RoI para ahorrar CPU
                rx1, ry1, rx2, ry2 = self.roi_box
                # Asegurar límites del RoI
                rx1 = max(0, rx1)
                ry1 = max(0, ry1)
                rx2 = min(ancho_red, rx2)
                ry2 = min(alto_red, ry2)
                
                roi_w = rx2 - rx1
                roi_h = ry2 - ry1
                
                if roi_w > 40 and roi_h > 40:
                    roi_frame = frame_pequeno[ry1:ry2, rx1:rx2]
                    # Inferencia en el RoI (el modelo SSD requiere blob de 300x300 exactamente)
                    blob = cv2.dnn.blobFromImage(roi_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.55:
                            # Mapear coordenadas locales del RoI a coordenadas de frame_pequeno
                            x_start = rx1 + int(detections[0, 0, i, 3] * roi_w)
                            y_start = ry1 + int(detections[0, 0, i, 4] * roi_h)
                            x_end = rx1 + int(detections[0, 0, i, 5] * roi_w)
                            y_end = rx1 + int(detections[0, 0, i, 6] * roi_h)
                            
                            caras_combinadas.append((x_start, y_start, x_end - x_start, y_end - y_start))
                            
                    if len(caras_combinadas) > 0:
                        escanear_completo = False
                        self.roi_fail_count = 0
                    else:
                        self.roi_fail_count += 1
                        if self.roi_fail_count > 5:
                            self.roi_box = None  # Perdimos el rostro, fallback a escaneo completo
            
            # Si no hay RoI activo o falló, escaneamos el frame completo cada 3 fotogramas (o si es frame 1)
            if escanear_completo:
                if self.contador_frames % 3 == 0 or self.roi_box is None:
                    # Inferencia en el frame completo (requiere blob de 300x300 exactamente)
                    blob = cv2.dnn.blobFromImage(frame_pequeno, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.55:
                            x_start = int(detections[0, 0, i, 3] * ancho_red)
                            y_start = int(detections[0, 0, i, 4] * alto_red)
                            x_end = int(detections[0, 0, i, 5] * ancho_red)
                            y_end = int(detections[0, 0, i, 6] * alto_red)
                            
                            caras_combinadas.append((x_start, y_start, x_end - x_start, y_end - y_start))
                else:
                    # En frames intermedios sin escaneo, reusamos el último RoI box como rostro activo
                    if self.roi_box is not None:
                        rx1, ry1, rx2, ry2 = self.roi_box
                        caras_combinadas.append((rx1, ry1, rx2 - rx1, ry2 - ry1))
                        
            # Actualizar dinámicamente la Región de Interés (RoI) en caliente
            if len(caras_combinadas) > 0:
                x_c, y_c, w_c, h_c = caras_combinadas[0]
                margen_w = int(w_c * 0.3)
                margen_h = int(h_c * 0.3)
                self.roi_box = [
                    x_c - margen_w,
                    y_c - margen_h,
                    x_c + w_c + margen_w,
                    y_c + h_c + margen_h
                ]
            else:
                self.roi_box = None
            
            # --- MÁQUINA DE ESTADOS GLOBAL ---
            ahora = time.time()
            hud_color_global = (0, 255, 0)
            hud_texto_global = f"ASISTENTE: ESCANEANDO... | FPS: {fps:.1f} | REGISTRADOS: {len(self.nombres_usuarios)}"
            
            # A. Registro Activo
            if self.registro_estado is not None:
                hud_color_global = (0, 165, 255)
                
                if self.registro_estado == "preguntando":
                    hud_texto_global = "ASISTENTE: PREGUNTANDO NOMBRE..."
                    if ahora - self.registro_timer > 6.0:
                        try:
                            winsound.Beep(1000, 300)
                        except:
                            pass
                        self.registro_estado = "escuchando"
                        self.registro_timer = ahora
                        self.input_nombre_resultado = None
                
                elif self.registro_estado == "escuchando":
                    hud_texto_global = "ASISTENTE: ESCUCHANDO NOMBRE... [HABLE AHORA]"
                    cv2.circle(frame_original, (w_orig - 30, 25), 10, (0, 0, 255), -1)
                
                elif self.registro_estado == "procesando_voz":
                    hud_texto_global = "ASISTENTE: PROCESANDO VOZ..."
                    if self.input_nombre_resultado is not None:
                        if self.input_nombre_resultado != "":
                            self.registro_nombre = self.input_nombre_resultado
                            self.registro_estado = "capturando_dinamico"
                            self.registro_timer = ahora
                            self.registro_fotos_guardadas = 0
                            self.registro_paso = 1
                            self.registro_paso_fotos_guardadas = 0
                            self.registro_paso_hablado = False
                            self.registro_paso_timer = ahora
                            
                            self.encolar_saludo_groq("durante_registro", self.registro_nombre)
                            print(f"[AUTO-REGISTRO] Asistente iniciado para '{self.registro_nombre}'")
                        else:
                            self.registro_estado = "esperando_popup"
                            self.input_nombre_resultado = None
                            encolar_saludo("No te escuché bien. Por favor escribe tu nombre en la ventana emergente.")
                            threading.Thread(target=self.solicitar_nombre_popup_worker, daemon=True).start()
                
                elif self.registro_estado == "esperando_popup":
                    hud_texto_global = "ASISTENTE: ESPERANDO NOMBRE EN POPUP..."
                
                elif self.registro_estado == "capturando_dinamico":
                    # Guiado por voz interactivo y por pasos/poses
                    if self.registro_paso == 1:
                        if not self.registro_paso_hablado:
                            encolar_saludo(f"Hola {self.registro_nombre}, iniciemos tu registro. Por favor mira de frente al centro y quédate serio.")
                            self.registro_paso_hablado = True
                            self.registro_paso_timer = ahora
                        hud_texto_global = f"REGISTRO: PASO 1/5 - MIRA DE FRENTE [SERIO]"
                    elif self.registro_paso == 2:
                        if not self.registro_paso_hablado:
                            encolar_saludo("Perfecto. Ahora mira de frente y sonríe.")
                            self.registro_paso_hablado = True
                            self.registro_paso_timer = ahora
                        hud_texto_global = f"REGISTRO: PASO 2/5 - MIRA DE FRENTE [SONRÍE]"
                    elif self.registro_paso == 3:
                        if not self.registro_paso_hablado:
                            encolar_saludo("Muy bien. Gira tu cabeza un poco hacia la izquierda.")
                            self.registro_paso_hablado = True
                            self.registro_paso_timer = ahora
                        hud_texto_global = f"REGISTRO: PASO 3/5 - GIRA A LA IZQUIERDA"
                    elif self.registro_paso == 4:
                        if not self.registro_paso_hablado:
                            encolar_saludo("Entendido. Ahora gira tu cabeza un poco hacia la derecha.")
                            self.registro_paso_hablado = True
                            self.registro_paso_timer = ahora
                        hud_texto_global = f"REGISTRO: PASO 4/5 - GIRA A LA DERECHA"
                    elif self.registro_paso == 5:
                        if not self.registro_paso_hablado:
                            encolar_saludo("Casi listo. Por último, mira un poco hacia arriba.")
                            self.registro_paso_hablado = True
                            self.registro_paso_timer = ahora
                        hud_texto_global = f"REGISTRO: PASO 5/5 - MIRA HACIA ARRRIBA"
                        
                    if self.registro_fotos_guardadas >= 15:
                        self.entrenar_en_segundo_plano()
                        self.encolar_saludo_groq("registro_completo", self.registro_nombre)
                        self.consecutivos_desconocidos = 0
                        self.memoria_deteccion.clear()
                        self.registro_estado = None
                    elif ahora - self.registro_timer > 60.0:  # Límite extendido a 60 segundos
                        self.entrenar_en_segundo_plano()
                        encolar_saludo(f"Perfecto {self.registro_nombre}, he completado tu registro.")
                        self.consecutivos_desconocidos = 0
                        self.memoria_deteccion.clear()
                        self.registro_estado = None

            # B. Conversación Activa
            elif self.chat_estado is not None:
                hud_color_global = (255, 100, 100)
                
                if self.chat_estado == "iniciando_chat":
                    hud_texto_global = f"CHARLANDO CON {self.chat_nombre.upper()}... [INICIANDO]"
                    if ahora - self.chat_timer > 5.0:
                        self.chat_estado = "esperando_habla"
                        self.chat_timer = ahora
                
                elif self.chat_estado == "esperando_habla":
                    hud_texto_global = f"CHARLANDO CON {self.chat_nombre.upper()}... [PREPARANDO]"
                    if ahora - self.chat_timer > 1.2:
                        try:
                            winsound.Beep(880, 200)
                        except:
                            pass
                        self.chat_estado = "escuchando_usuario"
                        self.chat_timer = ahora
                
                elif self.chat_estado == "escuchando_usuario":
                    hud_texto_global = f"CONVERSANDO: ALEXA ESCUCHA A {self.chat_nombre.upper()}..."
                    cv2.circle(frame_original, (w_orig - 30, 25), 10, (255, 0, 0), -1)
                
                elif self.chat_estado == "procesando_usuario":
                    hud_texto_global = "CONVERSANDO: ALEXA PROCESA RESPUESTA..."

            # Cartel informativo inicial
            if len(self.nombres_usuarios) == 0 and self.registro_estado is None:
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 0, 150), -1)
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 165, 255), 2)
                cv2.putText(frame_original, "MODO APRENDIZAJE INICIAL", (115, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_original, "Párate frente a la cámara para iniciar auto-registro por voz", (95, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
            # --- PROCESAR ROSTROS DETECTADOS ---
            nombre_actual_en_camara = None
            
            # Limpiar entradas antiguas del tracker (inactivas por más de 2 segundos)
            self.tracker_rostros = {fid: val for fid, val in self.tracker_rostros.items() if ahora - val[2] < 2.0}
            
            for (x_peq, y_peq, w_peq, h_peq) in caras_combinadas:
                x = x_peq * escala
                y = y_peq * escala
                w = w_peq * escala
                h = h_peq * escala
                
                cx = x + w // 2
                cy = y + h // 2
                
                # Asignar un ID persistente a este rostro por distancia de centroides
                face_id = None
                menor_dist = 999.0
                for fid, (fx, fy, ft) in self.tracker_rostros.items():
                    dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
                    if dist < 90.0 and dist < menor_dist:
                        face_id = fid
                        menor_dist = dist
                        
                if face_id is None:
                    face_id = self.next_face_id
                    self.next_face_id += 1
                    
                # Registrar posición actual en el tracker
                self.tracker_rostros[face_id] = (cx, cy, ahora)
                
                etiqueta = "Analizando..."
                subtitulo = "Estimando pose..."
                color = (0, 255, 255)
                
                cara_gris_recortada = gris_opt[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                
                # FQA: Filtro de desenfoque (Laplacian Variance)
                var_nitidez = 999.0
                if cara_gris_recortada.size > 0:
                    try:
                        var_nitidez = cv2.Laplacian(cara_gris_recortada, cv2.CV_64F).var()
                    except:
                        pass
                
                # 1. Modo Registro Activo
                if self.registro_estado is not None:
                    etiqueta = f"REGISTRANDO: {self.registro_nombre.upper()}"
                    if self.registro_paso == 1:
                        subtitulo = "Paso 1/5: Mira al frente (serio)"
                    elif self.registro_paso == 2:
                        subtitulo = "Paso 2/5: Mira al frente (sonrie)"
                    elif self.registro_paso == 3:
                        subtitulo = "Paso 3/5: Gira a la izquierda"
                    elif self.registro_paso == 4:
                        subtitulo = "Paso 4/5: Gira a la derecha"
                    elif self.registro_paso == 5:
                        subtitulo = "Paso 5/5: Mira hacia arriba"
                    else:
                        subtitulo = f"Capturas: {self.registro_fotos_guardadas}/15"
                    color = (0, 165, 255)
                    
                    self.procesar_captura_pasiva(cara_gris_recortada)
                        
                # 2. Modo Normal / Conversación (Preprocesar y clasificar de forma asíncrona)
                else:
                    # FQA: Si está muy borroso, omitir predicción para ahorrar CPU y evitar falsos positivos
                    if var_nitidez < 70.0:
                        etiqueta = "DESCARTADO"
                        subtitulo = f"Borroso (FQA): {var_nitidez:.1f}"
                        color = (0, 165, 255)
                        self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
                        continue
                        
                    # Lanzar clasificación asíncrona si ha pasado el intervalo (4 veces por segundo por rostro)
                    ultimo_t = self.ultimo_tiempo_clasificacion.get(face_id, 0.0)
                    if ahora - ultimo_t > 0.25:
                        self.ultimo_tiempo_clasificacion[face_id] = ahora
                        threading.Thread(
                            target=self.clasificar_rostro_async,
                            args=(face_id, cara_gris_recortada.copy()),
                            daemon=True
                        ).start()
                        
                    # Leer predicción asíncrona
                    pred = self.predicciones_activas.get(face_id, {
                        "nombre": "Identificando...",
                        "confianza": 0.0,
                        "distancia": 999.0
                    })
                    
                    nombre_detectado = pred["nombre"]
                    confianza_pct = pred["confianza"]
                    distancia = pred["distancia"]
                    
                    if nombre_detectado != "Identificando..." and nombre_detectado != "Desconocido":
                        # Si es con el que está charlando actualmente
                        if self.chat_estado is not None and nombre_detectado == self.chat_nombre:
                            etiqueta = f"HABLANDO: {nombre_detectado.upper()}"
                            color = (255, 100, 100)  # Color especial para el hablante activo
                        else:
                            etiqueta = nombre_detectado.upper()
                            color = (0, 255, 0)
                        subtitulo = f"Match: {confianza_pct:.1f}%"
                        self.consecutivos_desconocidos = 0
                        nombre_actual_en_camara = nombre_detectado
                    elif nombre_detectado == "Desconocido":
                        etiqueta = "DESCONOCIDO"
                        subtitulo = f"Match: {confianza_pct:.1f}%"
                        color = (0, 0, 255)
                        self.consecutivos_desconocidos += 1
                    else:
                        etiqueta = "IDENTIFICANDO..."
                        subtitulo = "Espere..."
                        color = (0, 255, 255)
                        
                    # Procesar lógica conversacional y auto-aprendizaje sólo si es modo normal (sin registro)
                    if self.registro_estado is None:
                        if len(self.memoria_deteccion) > 0:
                            voto_ganador = max(set(self.memoria_deteccion), key=self.memoria_deteccion.count)
                        else:
                            voto_ganador = "Desconocido"
                        votos_winner = self.memoria_deteccion.count(voto_ganador)
                        
                        if voto_ganador != "Desconocido" and votos_winner >= 4 and nombre_detectado == voto_ganador:
                            ahora_chat = time.time()
                            if ahora_chat - self.ultimo_chat_voz.get(voto_ganador, 0) > 90.0:
                                self.ultimo_chat_voz[voto_ganador] = ahora_chat
                                self.iniciar_charla_conversacional(voto_ganador)
                                
                            self.guardar_asistencia(voto_ganador, confianza_pct)
                            
                            ahora_t = time.time()
                            if ahora_t - ultimo_guardado_interactivo > 3.0:
                                self.auto_capturar_rostro_interaccion(voto_ganador, cara_gris_recortada, w)
                                ultimo_guardado_interactivo = ahora_t
                                
                        # Si no hay usuarios en la base de datos, iniciar auto-registro guiado una única vez al detectar a alguien
                        if len(self.nombres_usuarios) == 0 and self.consecutivos_desconocidos >= 40:
                            self.iniciar_registro_autonomo()
                            self.consecutivos_desconocidos = 0

                self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
            
            self.cara_detectada_nombre = nombre_actual_en_camara
            
            if len(caras_combinadas) == 0:
                self.cara_detectada_nombre = None
                if self.consecutivos_desconocidos > 0:
                    self.consecutivos_desconocidos -= 1
            
            # HUD Superior de Estado General
            cv2.rectangle(frame_original, (0, 0), (w_orig, 40), (15, 15, 15), -1)
            cv2.putText(frame_original, hud_texto_global, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color_global, 1, cv2.LINE_AA)
            
            # --- DIBUJAR NIVEL DE MICRÓFONO (VAD LIVE METER) ---
            mic_w = int(min(self.last_mic_rms * 500.0, 100.0))
            meter_x = w_orig - 120
            cv2.rectangle(frame_original, (meter_x, 15), (meter_x + 100, 25), (40, 40, 40), -1)
            color_barra = (0, 255, 255) if self.last_mic_rms > self.umbral_silencio else (120, 120, 120)
            cv2.rectangle(frame_original, (meter_x, 15), (meter_x + mic_w, 25), color_barra, -1)
            cv2.putText(frame_original, f"MIC (Thresh: {self.umbral_silencio:.3f})", (meter_x - 170, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
            # --- DIBUJAR SUBTÍTULOS DE TRANSCRIPCIÓN (VOZ A TEXTO EN PANTALLA) ---
            if self.texto_transcrito_pantalla and (time.time() - self.texto_timer_pantalla < 5.0):
                sub_y = h_orig - 45
                cv2.rectangle(frame_original, (10, sub_y - 20), (w_orig - 10, sub_y + 20), (15, 15, 15), -1)
                cv2.rectangle(frame_original, (10, sub_y - 20), (w_orig - 10, sub_y + 20), (100, 100, 100), 1)
                texto_sub = f"ESCUCHADO: {self.texto_transcrito_pantalla.upper()}"
                cv2.putText(frame_original, texto_sub, (25, sub_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("Antigravity Smart Recognition HUD", frame_original)
            
            t_fin = time.time()
            fps = 1.0 / (t_fin - t_inicio)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[SISTEMA] Cerrando aplicación y liberando recursos...")
                self.stop_listener = True
                break
            elif key == ord('r') or key == ord('R'):
                print("[SISTEMA] Solicitud de registro manual detectada por teclado.")
                self.iniciar_registro_autonomo()
                
        cap.release()
        cv2.destroyAllWindows()

# =====================================================================
# PUNTO DE ENTRADA
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA DE VISIÓN CON IA: AUTO-APRENDIZAJE Y APRENDIZAJE PASIVO CON GROQ LLM")
    print("=" * 60)
    
    try:
        app = SistemaReconocimientoFacial()
        app.iniciar_bucle_principal()
    except Exception as e:
        print(f"\n[CRÍTICO] Error al inicializar la aplicación: {e}")
        input("Presione ENTER para salir...")
