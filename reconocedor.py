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
from collections import deque
import tkinter as tk
from tkinter import simpledialog
import speech_recognition as sr

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================
DATASET_DIR = "dataset"
TRAINER_FILE = "clasificador.yml"
ATTENDANCE_FILE = "asistencia.csv"
HAAR_FRONTAL_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
HAAR_PROFILE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'
WIDTH_RESIZE = 150
HEIGHT_RESIZE = 150

# Asegurar que las carpetas base existan
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Hilo para la reproducción de voz (evita que el bucle de video se congele)
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
        # Cargar clasificadores (Frontal y Perfil)
        if not os.path.exists(HAAR_FRONTAL_PATH):
            raise FileNotFoundError(f"No se encontró Haar Cascade Frontal en {HAAR_FRONTAL_PATH}")
        if not os.path.exists(HAAR_PROFILE_PATH):
            raise FileNotFoundError(f"No se encontró Haar Cascade Perfil en {HAAR_PROFILE_PATH}")
            
        self.face_cascade = cv2.CascadeClassifier(HAAR_FRONTAL_PATH)
        self.profile_cascade = cv2.CascadeClassifier(HAAR_PROFILE_PATH)
        
        # Ecualizador de contraste adaptable (CLAHE) para normalización de iluminación
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Inicializar el reconocedor LBPH de OpenCV con hiperparámetros optimizados
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=12,
            grid_x=10,
            grid_y=10
        )
        
        self.modelo_cargado = False
        self.necesita_recargar_modelo = False  # Flag de control para recarga segura de modelo
        self.nombres_usuarios = {}
        self.cargar_modelo()
        
        # Leer clave API de Groq del archivo local .env
        self.groq_key = self.cargar_groq_key()
        
        # Historial de detecciones para estabilizar predicciones
        self.memoria_deteccion = deque(maxlen=7)
        self.cara_detectada_nombre = None
        
        # --- SEGUIDOR POR CENTROIDES (ID TEMPORAL) ---
        self.ultimo_centroide = None
        
        # --- MÁQUINA DE ESTADOS PARA EL REGISTRO DE VOZ (ALEXA STYLE) ---
        self.registro_estado = None
        self.registro_nombre = ""
        self.registro_timer = 0
        
        self.registro_fotos_front = 0
        self.registro_fotos_profile = 0
        self.registro_fotos_dist = 0
        
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

        # --- ESCUCHA CONTINUA EN SEGUNDO PLANO (SIEMPRE ESCUCHANDO) ---
        self.stop_listener = False
        self.hilo_escucha = threading.Thread(target=self.bucle_escucha_continua, daemon=True)
        self.hilo_escucha.start()

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

    def generar_frase_llm(self, tipo, nombre=None, texto_conversacion=None):
        """Genera una frase contextual humanizada, con personalidad e información del entorno usando Llama-3."""
        if not self.groq_key:
            if tipo == "chat":
                return "No tengo conexión a internet para charlar en este momento."
            return self.generar_frase_local(tipo, nombre)
            
        hora_actual = time.strftime("%H:%M")
        fecha_actual = time.strftime("%Y-%m-%d")
        total_usuarios = len(self.nombres_usuarios)
        
        persona = nombre if nombre else "un extraño"
        system_content = (
            f"Eres Alexa, una inteligencia artificial de visión muy ingeniosa, carismática y simpática en español. "
            f"Interactúas con {persona}. Datos actuales del entorno: Hora local: {hora_actual}, Fecha: {fecha_actual}, "
            f"Usuarios registrados en el sistema: {total_usuarios}. "
            f"Responde SIEMPRE en español, de forma muy amigable, ingeniosa, divertida y extremadamente breve (máximo 15 palabras). "
            f"Evita respuestas aburridas o predecibles. No uses markdown, asteriscos ni comillas."
        )
        
        if tipo == "chat" and texto_conversacion:
            self.historial_conversacion.append({"role": "user", "content": texto_conversacion})
            messages = [{"role": "system", "content": system_content}] + list(self.historial_conversacion)
        else:
            if tipo == "saludo_conocido":
                prompt = (
                    f"Genera un saludo ocurrente y muy corto (máximo 12 palabras) en español para {nombre}. "
                    f"Ej: '¡Vaya, al fin apareces {nombre}! Estaba aburriéndome de ver la pared.'"
                )
            elif tipo == "saludo_nuevo":
                prompt = "Dime que no te tengo registrado de una forma divertida e ingeniosa, y pídeme mi nombre (máximo 12 palabras)."
            elif tipo == "durante_registro":
                prompt = f"Dile a {nombre} con entusiasmo que ya sabes su nombre y que se mueva libremente mientras terminas de registrarle (máximo 12 palabras)."
            elif tipo == "registro_completo":
                prompt = f"Dile a {nombre} que el registro ha terminado con éxito y haz un comentario gracioso de bienvenida (máximo 12 palabras)."
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
                "model": "llama3-8b-8192",
                "messages": messages,
                "max_tokens": 45,
                "temperature": 0.85
            }
            res = requests.post(url, headers=headers, json=data, timeout=1.8)
            if res.status_code == 200:
                result = res.json()
                frase = result["choices"][0]["message"]["content"].strip().replace('"', '')
                if tipo == "chat":
                    self.historial_conversacion.append({"role": "assistant", "content": frase})
                return frase
        except Exception as e:
            print(f"[Groq LLM Error] Fallback local activo: {e}")
            
        if tipo == "chat":
            return "Interesante. ¿Qué más me cuentas?"
        return self.generar_frase_local(tipo, nombre)

    def encolar_saludo_groq(self, tipo, nombre=None):
        """Genera y encola un saludo usando Groq de forma asíncrona para no congelar el video."""
        def run():
            frase = self.generar_frase_llm(tipo, nombre)
            encolar_saludo(frase)
        threading.Thread(target=run, daemon=True).start()

    def generar_frase_local(self, tipo, nombre=None):
        """Generador local aleatorio de reserva."""
        if tipo == "saludo_conocido":
            return f"Hola {nombre}, bienvenido de nuevo al sistema."
        elif tipo == "saludo_nuevo":
            return "Hola, no te tengo en mi base de datos. ¿Cómo te llamas?"
        elif tipo == "durante_registro":
            return f"Perfecto {nombre}, quédate ahí quieto un momento mientras te analizo."
        elif tipo == "registro_completo":
            return f"Registro completado. Un placer conocerte, {nombre}."
        return ""

    def cargar_modelo(self):
        """Carga de forma segura el clasificador LBPH y actualiza el mapeo de nombres."""
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

    def entrenar_modelo(self):
        """
        Entrena el modelo LBPH usando un reconocedor temporal para evitar
        conflictos de hilos (Thread-safe) con el bucle de predicción principal.
        """
        print("[IA ENTRENAMIENTO] Iniciando entrenamiento en segundo plano...")
        rostros = []
        ids = []
        
        # Mapeo temporal local para la duración del entrenamiento
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
                    # Normalización de iluminación local CLAHE
                    img_gris = self.clahe.apply(img_gris)
                    img_gris = cv2.resize(img_gris, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                    rostros.append(img_gris)
                    ids.append(usuario_id)
        
        if len(rostros) == 0:
            return False
            
        try:
            # Crear reconocedor local aislado para entrenar
            temp_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2,
                neighbors=12,
                grid_x=10,
                grid_y=10
            )
            temp_recognizer.train(rostros, np.array(ids))
            temp_recognizer.write(TRAINER_FILE)
            
            # Notificar al hilo principal para que cargue el archivo de forma segura
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

    def bucle_escucha_continua(self):
        """Bucle infinito en segundo plano (Siempre escuchando)."""
        winmm = ctypes.windll.winmm
        r = sr.Recognizer()
        wav_path = "temp_background_mic.wav"
        
        time.sleep(2.0)
        
        while not self.stop_listener:
            if asistente_hablando:
                time.sleep(0.5)
                continue
                
            if self.registro_estado == "esperando_popup":
                time.sleep(0.5)
                continue
                
            winmm.mciSendStringW("close recsound", None, 0, 0)
            winmm.mciSendStringW("open new type waveaudio alias recsound", None, 0, 0)
            winmm.mciSendStringW("set recsound time format ms", None, 0, 0)
            winmm.mciSendStringW("set recsound bitspersample 16", None, 0, 0)
            winmm.mciSendStringW("set recsound samplespersec 16000", None, 0, 0)
            winmm.mciSendStringW("set recsound channels 1", None, 0, 0)
            
            winmm.mciSendStringW("record recsound", None, 0, 0)
            
            time.sleep(3.5)
            
            winmm.mciSendStringW("stop recsound", None, 0, 0)
            winmm.mciSendStringW(f"save recsound {wav_path}", None, 0, 0)
            winmm.mciSendStringW("close recsound", None, 0, 0)
            
            if asistente_hablando:
                try:
                    os.remove(wav_path)
                except:
                    pass
                continue
                
            texto_entendido = ""
            if os.path.exists(wav_path):
                try:
                    with sr.AudioFile(wav_path) as source:
                        audio_data = r.record(source)
                        texto_entendido = r.recognize_google(audio_data, language="es-ES").strip()
                        print(f"[PASIVO Escucha] He oído: '{texto_entendido}'")
                except:
                    pass
                finally:
                    try:
                        os.remove(wav_path)
                    except:
                        pass
            
            if texto_entendido:
                self.procesar_voz_entrada_pasiva(texto_entendido)

    def procesar_voz_entrada_pasiva(self, texto):
        """Procesa el texto capturado por la escucha continua en segundo plano."""
        if self.registro_estado == "escuchando":
            palabras = texto.strip().split()
            nombre = None
            if palabras:
                if len(palabras) >= 3 and palabras[0].lower() in ["me", "mi"] and palabras[1].lower() in ["llamo", "nombre"]:
                    nombre = palabras[2]
                else:
                    nombre = palabras[0]
            self.input_nombre_resultado = nombre
            self.registro_estado = "procesando_voz"
            
        elif self.chat_estado is not None:
            self.chat_estado = "procesando_usuario"
            threading.Thread(target=self.procesar_charla_worker, args=(texto,), daemon=True).start()
            
        elif self.chat_estado is None and self.registro_estado is None:
            if self.cara_detectada_nombre and self.cara_detectada_nombre != "Desconocido":
                nombre = self.cara_detectada_nombre
                self.chat_nombre = nombre
                self.chat_estado = "procesando_usuario"
                self.historial_conversacion.clear()
                
                print(f"[CONVERSACIÓN INICIADA PASIVAMENTE] {nombre} dijo: '{texto}'")
                self.ultimo_chat_voz[nombre] = time.time()
                
                threading.Thread(target=self.procesar_charla_worker, args=(texto,), daemon=True).start()

    def solicitar_nombre_popup_worker(self):
        """Diálogo de Tkinter en segundo plano."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        nombre = simpledialog.askstring("Identificación Inteligente", "Escribe tu nombre por favor:", parent=root)
        root.destroy()
        
        if nombre:
            self.input_nombre_resultado = nombre.strip().replace(" ", "_").capitalize()
        else:
            self.input_nombre_resultado = ""
        self.registro_estado = "procesando_voz"

    def iniciar_registro_autonomo(self):
        """Inicia el estado de registro guiado de forma humanizada llamando a Groq."""
        self.registro_estado = "preguntando"
        self.registro_timer = time.time()
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
            
        if w_face >= 120:
            distancia_tag = "cerca"
        elif w_face >= 60:
            distancia_tag = "medio"
        else:
            distancia_tag = "lejos"
            
        archivos_categoria = [f for f in os.listdir(ruta_usuario) if f.startswith(f"{nombre}_auto_{distancia_tag}_")]
        if len(archivos_categoria) >= 5:
            return
            
        rostro_gris_opt = self.clahe.apply(rostro_gris)
        rostro_red = cv2.resize(rostro_gris_opt, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
        
        index = len(archivos_categoria)
        archivo_nombre = f"{nombre}_auto_{distancia_tag}_{index}.jpg"
        cv2.imwrite(os.path.join(ruta_usuario, archivo_nombre), rostro_red)
        
        self.fotos_auto_guardadas += 1
        print(f"[APRENDIZAJE ONLINE] Captura selectiva en [{distancia_tag.upper()}] ({index + 1}/5) para '{nombre}'")
        
        if self.fotos_auto_guardadas >= 10:
            self.fotos_auto_guardadas = 0
            self.entrenar_en_segundo_plano()

    def procesar_captura_pasiva(self, rostro_gris, w_face, es_frontal):
        """Guarda silenciosa e implícitamente las caras aplicando normalización CLAHE local."""
        ruta_usuario = os.path.join(DATASET_DIR, self.registro_nombre)
        if not os.path.exists(ruta_usuario):
            os.makedirs(ruta_usuario)
            
        rostro_opt = self.clahe.apply(rostro_gris)
        rostro_red = cv2.resize(rostro_opt, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
        
        if w_face >= 120:
            dist_tag = "cerca"
        elif w_face >= 60:
            dist_tag = "medio"
        else:
            dist_tag = "lejos"
            
        if es_frontal and self.registro_fotos_front < 3:
            archivo = os.path.join(ruta_usuario, f"{self.registro_nombre}_front_{self.registro_fotos_front}.jpg")
            cv2.imwrite(archivo, rostro_red)
            self.registro_fotos_front += 1
            print(f"[REGISTRO PASIVO] Foto Frontal Guardada ({self.registro_fotos_front}/3)")
            time.sleep(0.15)
            
        elif not es_frontal and self.registro_fotos_profile < 3:
            archivo = os.path.join(ruta_usuario, f"{self.registro_nombre}_profile_{self.registro_fotos_profile}.jpg")
            cv2.imwrite(archivo, rostro_red)
            self.registro_fotos_profile += 1
            print(f"[REGISTRO PASIVO] Foto Perfil Guardada ({self.registro_fotos_profile}/3)")
            time.sleep(0.15)
            
        else:
            fotos_dist = [f for f in os.listdir(ruta_usuario) if f.startswith(f"{self.registro_nombre}_dist_{dist_tag}_")]
            if len(fotos_dist) < 3 and self.registro_fotos_dist < 3:
                index = len(fotos_dist)
                archivo = os.path.join(ruta_usuario, f"{self.registro_nombre}_dist_{dist_tag}_{index}.jpg")
                cv2.imwrite(archivo, rostro_red)
                self.registro_fotos_dist += 1
                print(f"[REGISTRO PASIVO] Foto Distancia [{dist_tag.upper()}] Guardada ({self.registro_fotos_dist}/3)")
                time.sleep(0.15)

    def dibujar_hud_futurista(self, frame, x, y, w, h, etiqueta, subtitulo, color):
        """Dibuja brackets esquineros, barra de fondo y escáner sobre el rostro."""
        longitud_linea = int(w * 0.2)
        grosor = 3
        
        # Arriba Izquierda
        cv2.line(frame, (x, y), (x + longitud_linea, y), color, grosor)
        cv2.line(frame, (x, y), (x, y + longitud_linea), color, grosor)
        # Arriba Derecha
        cv2.line(frame, (x + w, y), (x + w - longitud_linea, y), color, grosor)
        cv2.line(frame, (x + w, y), (x + w, y + longitud_linea), color, grosor)
        # Abajo Izquierda
        cv2.line(frame, (x, y + h), (x + longitud_linea, y + h), color, grosor)
        cv2.line(frame, (x, y + h), (x, y + h - longitud_linea), color, grosor)
        # Abajo Derecha
        cv2.line(frame, (x + w, y + h), (x + w - longitud_linea, y + h), color, grosor)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - longitud_linea), color, grosor)

        # Transparencia
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
        
        print("\nSISTEMA INTELIGENTE DE RECONOCIMIENTO FACIAL INICIADO (PRECISIÓN EXTREMA Y MULTIUSUARIOS).")
        print("Modo Conversacional y Autónomo Activo:")
        print("  - Registro dinámico multiusuario ilimitado e instantáneo.")
        print("  - Recarga segura de modelo (Thread-safe) en tiempo de ejecución sin interrupciones.")
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
            
            # --- RECARGA DE MODELO SEGURA (THREAD-SAFE) EN CALIENTE ---
            if self.necesita_recargar_modelo:
                self.cargar_modelo()
                self.necesita_recargar_modelo = False
                print("[HOT-RELOAD] El clasificador facial se ha actualizado de forma segura en caliente.")
            
            # --- OPTIMIZACIÓN DE FPS (Procesar a 1/2 resolución) ---
            escala = 2
            ancho_red = w_orig // escala
            alto_red = h_orig // escala
            frame_pequeno = cv2.resize(frame, (ancho_red, alto_red))
            
            gris = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)
            gris_opt = self.clahe.apply(gris)
            
            # Detectores de Rostros (Frontal y Perfil)
            caras_frontales = self.face_cascade.detectMultiScale(
                gris_opt, 
                scaleFactor=1.08, 
                minNeighbors=4, 
                minSize=(20, 20)
            )
            
            caras_perfil = []
            if len(caras_frontales) == 0:
                caras_perfil = self.profile_cascade.detectMultiScale(
                    gris_opt,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(20, 20)
                )
            
            # Consolidar caras
            caras_combinadas = []
            for (xf, yf, wf, hf) in caras_frontales:
                caras_combinadas.append((xf, yf, wf, hf, True))
            for (xp, yp, wp, hp) in caras_perfil:
                caras_combinadas.append((xp, yp, wp, hp, False))
            
            cara_detectada_este_frame = len(caras_combinadas) > 0
            
            # --- MÁQUINA DE ESTADOS GLOBAL (REGISTRO Y CHARLA) ---
            ahora = time.time()
            hud_color_global = (0, 255, 0)
            hud_texto_global = f"ASISTENTE: ESCANEANDO... | FPS: {fps:.1f} | REGISTRADOS: {len(self.nombres_usuarios)}"
            
            # A. Máquina de Estados de Registro Activo
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
                            self.registro_fotos_front = 0
                            self.registro_fotos_profile = 0
                            self.registro_fotos_dist = 0
                            
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
                    total_capturas = self.registro_fotos_front + self.registro_fotos_profile + self.registro_fotos_dist
                    hud_texto_global = f"REGISTRO ACTIVO: {self.registro_nombre.upper()} | CAPTURAS: {total_capturas}/9"
                    
                    if self.registro_fotos_front >= 3 and self.registro_fotos_profile >= 3 and self.registro_fotos_dist >= 3:
                        self.entrenar_en_segundo_plano()
                        self.encolar_saludo_groq("registro_completo", self.registro_nombre)
                        
                        self.consecutivos_desconocidos = 0
                        self.memoria_deteccion.clear()
                        self.registro_estado = None
                    elif ahora - self.registro_timer > 20.0:
                        self.entrenar_en_segundo_plano()
                        encolar_saludo(f"Perfecto {self.registro_nombre}, he completado tu registro.")
                        self.consecutivos_desconocidos = 0
                        self.memoria_deteccion.clear()
                        self.registro_estado = None

            # B. Máquina de Estados de Conversación Conversacional Activa
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
            
            for (x_peq, y_peq, w_peq, h_peq, es_frontal) in caras_combinadas:
                x = x_peq * escala
                y = y_peq * escala
                w = w_peq * escala
                h = h_peq * escala
                
                cx = x + w // 2
                cy = y + h // 2
                
                es_mismo_rostro = False
                if self.ultimo_centroide is not None:
                    dist_centroides = np.sqrt((cx - self.ultimo_centroide[0])**2 + (cy - self.ultimo_centroide[1])**2)
                    if dist_centroides < 85:
                        es_mismo_rostro = True
                
                self.ultimo_centroide = (cx, cy)
                
                etiqueta = "Analizando..."
                subtitulo = "Estimando pose..."
                color = (0, 255, 255)
                
                cara_gris_recortada = gris_opt[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                
                # 1. Modo Normal
                if self.registro_estado is None and self.chat_estado is None:
                    if self.modelo_cargado:
                        try:
                            # Normalización de contraste local CLAHE
                            cara_gris_norm = self.clahe.apply(cara_gris_recortada)
                            cara_gris_norm = cv2.resize(cara_gris_norm, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                            
                            id_prediccion, distancia = self.recognizer.predict(cara_gris_norm)
                            confianza_pct = max(0, 100 - distancia)
                            
                            if confianza_pct > 38:
                                nombre_detectado = self.nombres_usuarios.get(id_prediccion, "Desconocido")
                                self.memoria_deteccion.append(nombre_detectado)
                            else:
                                self.memoria_deteccion.append("Desconocido")
                                
                            if len(self.memoria_deteccion) > 0:
                                voto_ganador = max(set(self.memoria_deteccion), key=self.memoria_deteccion.count)
                            else:
                                voto_ganador = "Desconocido"
                                
                            if voto_ganador != "Desconocido":
                                etiqueta = f"ACTIVO: {voto_ganador.upper()}"
                                subtitulo = f"Match: {confianza_pct:.1f}% ({'Frente' if es_frontal else 'Perfil'})"
                                color = (0, 255, 0)
                                
                                self.consecutivos_desconocidos = 0
                                nombre_actual_en_camara = voto_ganador
                                
                                ahora_chat = time.time()
                                if ahora_chat - self.ultimo_chat_voz.get(voto_ganador, 0) > 90.0:
                                    self.ultimo_chat_voz[voto_ganador] = ahora_chat
                                    self.iniciar_charla_conversacional(voto_ganador)
                                
                                self.guardar_asistencia(voto_ganador, confianza_pct)
                                
                                ahora_t = time.time()
                                if ahora_t - ultimo_guardado_interactivo > 3.0:
                                    self.auto_capturar_rostro_interaccion(voto_ganador, cara_gris_recortada, w)
                                    ultimo_guardado_interactivo = ahora_t
                            else:
                                etiqueta = "DESCONOCIDO"
                                subtitulo = "Identificando..."
                                color = (0, 0, 255)
                                self.consecutivos_desconocidos += 1
                                
                        except Exception as e:
                            subtitulo = f"Error IA: {e}"
                            color = (0, 0, 255)
                    else:
                        etiqueta = "NUEVA PERSONA"
                        subtitulo = "Auto-registro..."
                        color = (0, 165, 255)
                        self.consecutivos_desconocidos += 1
                    
                    if self.consecutivos_desconocidos >= 45:
                        self.iniciar_registro_autonomo()
                
                # 2. Modo Registro Activo
                elif self.registro_estado is not None:
                    pose_tag = "Frontal" if es_frontal else "Perfil"
                    etiqueta = f"REGISTRANDO: {self.registro_nombre.upper()}"
                    subtitulo = f"Pose: {pose_tag.upper()}"
                    color = (0, 165, 255)
                    
                    if es_mismo_rostro or self.registro_fotos_front == 0:
                        self.procesar_captura_pasiva(cara_gris_recortada, w, es_frontal)
                        
                # 3. Modo Conversación Activo
                elif self.chat_estado is not None:
                    etiqueta = f"CHARLANDO CON {self.chat_nombre.upper()}"
                    subtitulo = f"Alexa: Conexión Activa ({'Frente' if es_frontal else 'Perfil'})"
                    color = (255, 100, 100)

                self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
            
            self.cara_detectada_nombre = nombre_actual_en_camara
            
            if not cara_detectada_este_frame:
                self.ultimo_centroide = None
                self.cara_detectada_nombre = None
                if self.consecutivos_desconocidos > 0:
                    self.consecutivos_desconocidos -= 1
            
            cv2.imshow("Antigravity Smart Recognition HUD", frame_original)
            
            t_fin = time.time()
            fps = 1.0 / (t_fin - t_inicio)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[SISTEMA] Cerrando aplicación y liberando recursos...")
                self.stop_listener = True
                break
                
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
