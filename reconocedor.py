import cv2
import numpy as np
import os
import time
import csv
import threading
import pyttsx3
import winsound
import ctypes
import random
import requests
import pythoncom
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

# =====================================================================
# DICCIONARIOS DE FRASES HUMANIZADAS LOCALES (FALLBACK)
# =====================================================================
FRASES_SALUDOS_CONOCIDOS = [
    "¡Hola [nombre]! Qué bueno verte por aquí.",
    "Hola [nombre], ¿cómo va tu día?",
    "Qué tal [nombre], un placer saludarte de nuevo.",
    "Hola de nuevo, [nombre]. Te veo genial hoy.",
    "¡Hola [nombre]! Acceso verificado."
]

FRASES_SALUDOS_NUEVOS = [
    "¡Hola! Oye, estaba revisando y no te tengo en mi base de datos. ¿Cómo te llamas?",
    "Hola. Qué curioso, no te reconozco. ¿Me dirías tu nombre por favor?",
    "¡Qué tal! Veo una cara nueva. ¿Cuál es tu nombre?",
    "Hola, no te tengo registrado todavía. ¿Cómo te llamas?"
]

FRASES_DURANTE_REGISTRO = [
    "¡Qué bien, [nombre]! Un gusto conocerte. Quédate ahí charlando conmigo mientras configuro tu perfil.",
    "Un placer, [nombre]. Solo quédate ahí un momento, estoy analizando tus ángulos en segundo plano.",
    "Estupendo, [nombre]. Háblame un poco o muévete despacio, estoy registrando tus facciones.",
    "Excelente, [nombre]. Ya tengo tu nombre. Sigue mirándome un momento."
]

FRASES_REGISTRO_COMPLETO = [
    "¡Todo listo! Ya registré tu rostro en mi base de datos. Un placer conocerte, [nombre].",
    "Perfecto, [nombre]. Configuración terminada. Ya sé quién eres.",
    "Listo, [nombre]. Registro completado con éxito. Ahora ya te reconozco.",
    "Hecho, [nombre]. He asimilado tus firmas faciales."
]

# Hilo para la reproducción de voz (evita que el bucle de video se congele)
voz_lock = threading.Lock()
cola_saludos = deque(maxlen=5)

def hablar_worker():
    """Ejecuta en segundo plano la síntesis de voz inicializando COM para SAPI5 en Windows."""
    # Inicialización obligatoria de COM para habilitar voz en hilos de Windows (SAPI5)
    pythoncom.CoInitialize()
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if "spanish" in voice.name.lower() or "es" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 160)
    except Exception as e:
        print(f"[Voz Error] No se pudo inicializar el motor de voz: {e}")
        return

    while True:
        if cola_saludos:
            texto = cola_saludos.popleft()
            with voz_lock:
                try:
                    engine.say(texto)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[Voz Error] Error al reproducir audio: {e}")
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
        
        # Inicializar el reconocedor LBPH de OpenCV
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.modelo_cargado = False
        self.nombres_usuarios = {}
        self.cargar_modelo()
        
        # Leer clave API de Groq del archivo local .env
        self.groq_key = self.cargar_groq_key()
        
        # Ecualizador de contraste adaptable (CLAHE) para optimizar detección lejana
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Historial de detecciones para estabilizar predicciones
        self.memoria_deteccion = deque(maxlen=7)
        
        # --- SEGUIDOR POR CENTROIDES (ID TEMPORAL) ---
        self.ultimo_centroide = None  # (cx, cy)
        
        # --- MÁQUINA DE ESTADOS PARA EL REGISTRO DE VOZ (ALEXA STYLE) ---
        self.registro_estado = None
        self.registro_nombre = ""
        self.registro_timer = 0
        
        # Conteo de fotos por categorías de pose para evitar saturación
        self.registro_fotos_front = 0
        self.registro_fotos_profile = 0
        self.registro_fotos_dist = 0
        
        # --- MÁQUINA DE ESTADOS DE CONVERSACIÓN ACTIVA (LLM CHAT) ---
        # Estados: None (escaneo), "iniciando_chat", "esperando_habla", "escuchando_usuario", "procesando_usuario"
        self.chat_estado = None
        self.chat_nombre = ""
        self.chat_timer = 0
        self.historial_conversacion = deque(maxlen=6)  # Almacena el contexto de charla
        
        # Variables de control de hilos de voz e inputs
        self.input_nombre_resultado = None
        self.consecutivos_desconocidos = 0
        self.fotos_auto_guardadas = 0
        
        # Cooldown temporal de registros en CSV (15 segundos)
        self.ultimo_registro_asistencia = {}
        
        # Cooldown de chat por voz (para no iniciar charla automáticamente cada segundo si el usuario está en cámara)
        # Se puede volver a chatear después de 90 segundos de inactividad
        self.ultimo_chat_voz = {}

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
        """Genera una frase contextual humanizada usando Groq Llama-3 o fallback local."""
        if not self.groq_key:
            if tipo == "chat":
                return "No tengo conexión a internet para charlar en este momento."
            return self.generar_frase_local(tipo, nombre)
            
        system_content = f"Eres Alexa, el asistente de voz en español de un sistema de visión artificial. Tu interlocutor es {nombre if nombre else 'un extraño'}. Responde siempre de forma muy breve, amigable, natural y con chispa (máximo 15 palabras). Nunca uses markdown, asteriscos, guiones ni comillas."
        
        if tipo == "chat" and texto_conversacion:
            # Añadir mensaje al historial
            self.historial_conversacion.append({"role": "user", "content": texto_conversacion})
            messages = [{"role": "system", "content": system_content}] + list(self.historial_conversacion)
        else:
            if tipo == "saludo_conocido":
                prompt = f"Genera un saludo muy corto y casual (máximo 12 palabras) en español para un usuario conocido llamado {nombre}. Ej: '¡Hola Carlos! Qué bueno verte, ¿cómo va todo?'"
            elif tipo == "saludo_nuevo":
                prompt = "Genera una frase muy corta (máximo 12 palabras) en español para decirle a un extraño que es una cara nueva y preguntarle su nombre de forma casual y educada."
            elif tipo == "durante_registro":
                prompt = f"Genera una frase muy corta (máximo 12 palabras) en español para decirle a {nombre} que te alegra conocerle y que continúe mirándote un momento mientras analizas sus rasgos en segundo plano."
            elif tipo == "registro_completo":
                prompt = f"Genera una frase de bienvenida muy corta (máximo 12 palabras) en español para decirle a {nombre} que el registro terminó con éxito y ya lo tienes registrado."
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
                "temperature": 0.8
            }
            res = requests.post(url, headers=headers, json=data, timeout=1.8)
            if res.status_code == 200:
                result = res.json()
                frase = result["choices"][0]["message"]["content"].strip().replace('"', '')
                if tipo == "chat":
                    # Registrar respuesta en el historial
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
            return random.choice(FRASES_SALUDOS_CONOCIDOS).replace("[nombre]", nombre)
        elif tipo == "saludo_nuevo":
            return random.choice(FRASES_SALUDOS_NUEVOS)
        elif tipo == "durante_registro":
            return random.choice(FRASES_DURANTE_REGISTRO).replace("[nombre]", nombre)
        elif tipo == "registro_completo":
            return random.choice(FRASES_REGISTRO_COMPLETO).replace("[nombre]", nombre)
        return ""

    def cargar_modelo(self):
        """Carga el clasificador LBPH y el mapeo de IDs de usuarios."""
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
        """Asigna un ID numérico a cada carpeta de usuario."""
        if not os.path.exists(DATASET_DIR):
            return
        subdirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        self.nombres_usuarios = {i: name for i, name in enumerate(subdirs)}

    def entrenar_modelo(self):
        """Entrena el modelo LBPH de OpenCV con las imágenes en dataset/."""
        print("[IA ENTRENAMIENTO] Iniciando entrenamiento en segundo plano...")
        rostros = []
        ids = []
        
        self.actualizar_mapeo_nombres()
        nombre_a_id = {v: k for k, v in self.nombres_usuarios.items()}
        
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
                    img_gris = cv2.resize(img_gris, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                    rostros.append(img_gris)
                    ids.append(usuario_id)
        
        if len(rostros) == 0:
            self.modelo_cargado = False
            return False
            
        try:
            self.recognizer.train(rostros, np.array(ids))
            self.recognizer.write(TRAINER_FILE)
            self.modelo_cargado = True
            print("[IA ENTRENAMIENTO] Modelo actualizado y cargado con éxito.")
            return True
        except Exception as e:
            print(f"[IA ERROR] Error al entrenar: {e}")
            return False

    def entrenar_en_segundo_plano(self):
        """Ejecuta el entrenamiento en un hilo para evitar congelamientos."""
        hilo = threading.Thread(target=self.entrenar_modelo, daemon=True)
        hilo.start()

    def grabar_audio_mci_worker(self):
        """
        Graba audio del micrófono en segundo plano configurando la tarjeta a 16kHz Mono.
        Esto optimiza al máximo la precisión del motor de Speech Recognition.
        """
        winmm = ctypes.windll.winmm
        wav_path = "registro_voz_temp.wav"
        
        winmm.mciSendStringW("close recsound", None, 0, 0)
        
        # --- MEJORA DE MICRÓFONO: Configuración a 16000Hz, 16 bits y canal Mono ---
        winmm.mciSendStringW("open new type waveaudio alias recsound", None, 0, 0)
        winmm.mciSendStringW("set recsound time format ms", None, 0, 0)
        winmm.mciSendStringW("set recsound bitspersample 16", None, 0, 0)
        winmm.mciSendStringW("set recsound samplespersec 16000", None, 0, 0)
        winmm.mciSendStringW("set recsound channels 1", None, 0, 0)
        
        winmm.mciSendStringW("record recsound", None, 0, 0)
        
        # Grabar durante 4.5 segundos
        time.sleep(4.5)
        
        winmm.mciSendStringW("stop recsound", None, 0, 0)
        winmm.mciSendStringW(f"save recsound {wav_path}", None, 0, 0)
        winmm.mciSendStringW("close recsound", None, 0, 0)
        
        r = sr.Recognizer()
        nombre_transcrito = None
        if os.path.exists(wav_path):
            try:
                with sr.AudioFile(wav_path) as source:
                    audio_data = r.record(source)
                    transcripcion = r.recognize_google(audio_data, language="es-ES")
                    print(f"[Voz Entendida] {transcripcion}")
                    
                    # Si es modo conversación regular, devolver la frase completa
                    if self.chat_estado is not None:
                        nombre_transcrito = transcripcion
                    else:
                        # Si es modo registro de nombre, extraer el nombre único
                        palabras = transcripcion.strip().split()
                        if palabras:
                            if len(palabras) >= 3 and palabras[0].lower() in ["me", "mi"] and palabras[1].lower() in ["llamo", "nombre"]:
                                nombre_transcrito = palabras[2]
                            else:
                                nombre_transcrito = palabras[0]
            except Exception as e:
                print(f"[Voz Error] {e}")
            finally:
                try:
                    os.remove(wav_path)
                except:
                    pass
        
        self.input_nombre_resultado = nombre_transcrito
        
        # Redirigir según el estado activo
        if self.chat_estado == "escuchando_usuario":
            self.chat_estado = "procesando_usuario"
        else:
            self.registro_estado = "procesando_voz"

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
        
        # Verificar si el usuario dijo adiós para terminar el chat
        palabras_cierre = ["adiós", "adios", "chao", "bye", "salir", "nos vemos", "gracias"]
        cerrar_conversacion = False
        for pc in palabras_cierre:
            if pc in texto_usuario.lower():
                cerrar_conversacion = True
                break
                
        if cerrar_conversacion:
            self.chat_estado = None
        else:
            # Volver a escuchar
            self.chat_estado = "esperando_habla"
            self.chat_timer = time.time()

    def auto_capturar_rostro_interaccion(self, nombre, rostro_gris, w_face):
        """
        Muestreo selectivo interactivo para asimilar variaciones de distancia.
        Máximo 5 fotos por categoría (Cerca/Medio/Lejos) para evitar saturar el disco.
        """
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
            
        rostro_red = cv2.resize(rostro_gris, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
        index = len(archivos_categoria)
        archivo_nombre = f"{nombre}_auto_{distancia_tag}_{index}.jpg"
        cv2.imwrite(os.path.join(ruta_usuario, archivo_nombre), rostro_red)
        
        self.fotos_auto_guardadas += 1
        print(f"[APRENDIZAJE ONLINE] Captura selectiva en [{distancia_tag.upper()}] ({index + 1}/5) para '{nombre}'")
        
        if self.fotos_auto_guardadas >= 10:
            self.fotos_auto_guardadas = 0
            self.entrenar_en_segundo_plano()

    def procesar_captura_pasiva(self, rostro_gris, w_face, es_frontal):
        """
        Guarda silenciosa e implícitamente las caras en diferentes poses y distancias
        durante el proceso de registro activo (Alexa Style) sin interrumpir el video.
        """
        ruta_usuario = os.path.join(DATASET_DIR, self.registro_nombre)
        if not os.path.exists(ruta_usuario):
            os.makedirs(ruta_usuario)
            
        rostro_red = cv2.resize(rostro_gris, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
        
        # Categorizar por distancia
        if w_face >= 120:
            dist_tag = "cerca"
        elif w_face >= 60:
            dist_tag = "medio"
        else:
            dist_tag = "lejos"
            
        # 1. Si la pose es frontal, guardarla en "front"
        if es_frontal and self.registro_fotos_front < 3:
            archivo = os.path.join(ruta_usuario, f"{self.registro_nombre}_front_{self.registro_fotos_front}.jpg")
            cv2.imwrite(archivo, rostro_red)
            self.registro_fotos_front += 1
            print(f"[REGISTRO PASIVO] Foto Frontal Guardada ({self.registro_fotos_front}/3)")
            time.sleep(0.15)
            
        # 2. Si la pose es perfil, guardarla en "profile"
        elif not es_frontal and self.registro_fotos_profile < 3:
            archivo = os.path.join(ruta_usuario, f"{self.registro_nombre}_profile_{self.registro_fotos_profile}.jpg")
            cv2.imwrite(archivo, rostro_red)
            self.registro_fotos_profile += 1
            print(f"[REGISTRO PASIVO] Foto Perfil Guardada ({self.registro_fotos_profile}/3)")
            time.sleep(0.15)
            
        # 3. Guardar por variación de distancia en "dist"
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
        
        print("\nSISTEMA INTELIGENTE DE RECONOCIMIENTO FACIAL INICIADO.")
        print("Modo Conversacional y Autónomo Activo:")
        print("  - Voz funcional habilitada en hilos mediante CoInitialize.")
        print("  - Diálogos inteligentes bidireccionales con Groq Llama-3.")
        print("  - Grabación de micrófono a 16kHz Mono optimizada para alta precisión.")
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
                hud_color_global = (0, 165, 255)  # Naranja
                
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
                        threading.Thread(target=self.grabar_audio_mci_worker, daemon=True).start()
                
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

            # B. Máquina de Estados de Conversación Conversacional Activa (Chat con Llama-3)
            elif self.chat_estado is not None:
                hud_color_global = (255, 100, 100)  # Azul claro/violeta de conversación
                
                if self.chat_estado == "iniciando_chat":
                    hud_texto_global = f"CHARLANDO CON {self.chat_nombre.upper()}... [INICIANDO]"
                    if ahora - self.chat_timer > 5.0:  # Termina el saludo
                        self.chat_estado = "esperando_habla"
                        self.chat_timer = ahora
                
                elif self.chat_estado == "esperando_habla":
                    hud_texto_global = f"CHARLANDO CON {self.chat_nombre.upper()}... [PREPARANDO]"
                    if ahora - self.chat_timer > 1.2:
                        try:
                            winsound.Beep(880, 200)  # Tono de escucha
                        except:
                            pass
                        self.chat_estado = "escuchando_usuario"
                        self.chat_timer = ahora
                        self.input_nombre_resultado = None
                        threading.Thread(target=self.grabar_audio_mci_worker, daemon=True).start()
                
                elif self.chat_estado == "escuchando_usuario":
                    hud_texto_global = f"CONVERSANDO: ALEXA ESCUCHA A {self.chat_nombre.upper()}..."
                    cv2.circle(frame_original, (w_orig - 30, 25), 10, (255, 0, 0), -1)  # Círculo azul de Alexa
                
                elif self.chat_estado == "procesando_usuario":
                    hud_texto_global = "CONVERSANDO: ALEXA PROCESA RESPUESTA..."
                    if self.input_nombre_resultado is not None:
                        # Si entendió alguna frase
                        if self.input_nombre_resultado.strip() != "":
                            texto_dicho = self.input_nombre_resultado
                            print(f"[CONVERSACIÓN] {self.chat_nombre}: {texto_dicho}")
                            self.input_nombre_resultado = None
                            
                            # Generar respuesta de Groq en hilo de fondo
                            self.chat_estado = "procesando_usuario"
                            threading.Thread(target=self.procesar_charla_worker, args=(texto_dicho,), daemon=True).start()
                        else:
                            # Silencio, cerrar charla con saludo amigable
                            print("[CONVERSACIÓN] Silencio detectado. Cerrando canal.")
                            encolar_saludo(f"Nos vemos luego, {self.chat_nombre}.")
                            self.chat_estado = None
                    else:
                        # Procesando en hilo en ejecución
                        pass

            # Cartel informativo inicial
            if len(self.nombres_usuarios) == 0 and self.registro_estado is None:
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 0, 150), -1)
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 165, 255), 2)
                cv2.putText(frame_original, "MODO APRENDIZAJE INICIAL", (115, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_original, "Párate frente a la cámara para iniciar auto-registro por voz", (95, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
            # --- PROCESAR ROSTROS DETECTADOS ---
            for (x_peq, y_peq, w_peq, h_peq, es_frontal) in caras_combinadas:
                x = x_peq * escala
                y = y_peq * escala
                w = w_peq * escala
                h = h_peq * escala
                
                cx = x + w // 2
                cy = y + h // 2
                
                # Seguimiento por centroides
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
                
                # 1. Modo Normal (Escaneo y disparador de charla)
                if self.registro_estado is None and self.chat_estado is None:
                    if self.modelo_cargado:
                        try:
                            cara_gris_norm = cv2.resize(cara_gris_recortada, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
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
                                
                                # Disparar bucle conversacional inteligente (cooldown de 90 segundos entre charlas)
                                ahora_chat = time.time()
                                if ahora_chat - self.ultimo_chat_voz.get(voto_ganador, 0) > 90.0:
                                    self.ultimo_chat_voz[voto_ganador] = ahora_chat
                                    self.iniciar_charla_conversacional(voto_ganador)
                                
                                self.guardar_asistencia(voto_ganador, confianza_pct)
                                
                                # Captura interactiva selectiva
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
                    
                    # DISPARAR REGISTRO AUTÓNOMO
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
                        
                # 3. Modo Conversación Activo (Charla con Llama-3)
                elif self.chat_estado is not None:
                    etiqueta = f"CHARLANDO CON {self.chat_nombre.upper()}"
                    subtitulo = f"Alexa: Conexión Activa ({'Frente' if es_frontal else 'Perfil'})"
                    color = (255, 100, 100)  # Azul/Púrpura

                # Dibujar HUD
                self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
            
            # Resetear tracker si no hay caras
            if not cara_detectada_este_frame:
                self.ultimo_centroide = None
                if self.consecutivos_desconocidos > 0:
                    self.consecutivos_desconocidos -= 1
            
            cv2.imshow("Antigravity Smart Recognition HUD", frame_original)
            
            # Medir FPS
            t_fin = time.time()
            fps = 1.0 / (t_fin - t_inicio)
            
            # Salir con 'Q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[SISTEMA] Cerrando aplicación y liberando recursos...")
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
