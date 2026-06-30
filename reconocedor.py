import cv2
import numpy as np
import os
import time
import csv
import threading
import pyttsx3
import winsound
import ctypes
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
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
WIDTH_RESIZE = 150
HEIGHT_RESIZE = 150

# Asegurar que las carpetas base existan
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Hilo para la reproducción de voz (evita que el bucle de video se congele)
voz_lock = threading.Lock()
cola_saludos = deque(maxlen=5)

def hablar_worker():
    """Ejecuta en segundo plano la síntesis de voz para no interrumpir los FPS del video."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if "spanish" in voice.name.lower() or "es" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 165)
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
                except Exception:
                    pass
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
# CLASE PRINCIPAL: RECONOCEDOR FACIAL INTELIGENTE AUTÓNOMO
# =====================================================================
class SistemaReconocimientoFacial:
    def __init__(self):
        # Cargar clasificador de rostros (Haar Cascade)
        if not os.path.exists(HAAR_CASCADE_PATH):
            raise FileNotFoundError(f"No se encontró Haar Cascade en {HAAR_CASCADE_PATH}")
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        
        # Inicializar el reconocedor LBPH de OpenCV
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.modelo_cargado = False
        self.nombres_usuarios = {}
        self.cargar_modelo()
        
        # Ecualizador de contraste adaptable (CLAHE) para optimizar detección en condiciones variables
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Historial de detecciones para estabilizar predicciones (Evita parpadeos)
        self.memoria_deteccion = deque(maxlen=7)
        
        # --- MÁQUINA DE ESTADOS PARA EL REGISTRO DE VOZ (ALEXA STYLE) ---
        # Estados: None (escaneo normal), "preguntando", "escuchando", "procesando_voz",
        #          "esperando_popup", "aviso_frente", "capturando_frente",
        #          "aviso_izquierda", "capturando_izquierda", "aviso_derecha", "capturando_derecha"
        self.registro_estado = None
        self.registro_nombre = ""
        self.registro_timer = 0
        self.registro_capturas = 0
        
        # Variables de control de hilos de voz e inputs
        self.voz_hilo_activo = False
        self.input_nombre_resultado = None
        self.consecutivos_desconocidos = 0
        self.fotos_auto_guardadas = 0
        self.ultimo_registro_asistencia = {}

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
        """Graba en segundo plano para no congelar el bucle de video en la pantalla."""
        winmm = ctypes.windll.winmm
        wav_path = "registro_voz_temp.wav"
        
        winmm.mciSendStringW("close recsound", None, 0, 0)
        winmm.mciSendStringW("open new type waveaudio alias recsound", None, 0, 0)
        winmm.mciSendStringW("record recsound", None, 0, 0)
        
        # Graba por 4 segundos
        time.sleep(4.0)
        
        winmm.mciSendStringW("stop recsound", None, 0, 0)
        winmm.mciSendStringW(f"save recsound {wav_path}", None, 0, 0)
        winmm.mciSendStringW("close recsound", None, 0, 0)
        
        # Procesar audio usando SpeechRecognition
        r = sr.Recognizer()
        nombre_transcrito = None
        if os.path.exists(wav_path):
            try:
                with sr.AudioFile(wav_path) as source:
                    audio_data = r.record(source)
                    transcripcion = r.recognize_google(audio_data, language="es-ES")
                    print(f"[Voz Entendida] {transcripcion}")
                    
                    palabras = transcripcion.strip().split()
                    if palabras:
                        # Limpiar saludos usuales
                        if len(palabras) >= 3 and palabras[0].lower() in ["me", "mi"] and palabras[1].lower() in ["llamo", "nombre"]:
                            nombre_transcrito = palabras[2]
                        else:
                            nombre_transcrito = palabras[0]
            except Exception as e:
                print(f"[Voz Error] No se entendió: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except:
                    pass
        
        # Devolver el resultado a la máquina de estados principal
        self.input_nombre_resultado = nombre_transcrito
        self.registro_estado = "procesando_voz"

    def solicitar_nombre_popup_worker(self):
        """Abre el diálogo Tkinter en un hilo separado para que el video continúe actualizándose."""
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
        """Inicia el estado de registro guiado."""
        self.registro_estado = "preguntando"
        self.registro_timer = time.time()
        encolar_saludo("Hola, detecto que eres alguien nuevo. ¿Cómo te llamas? Por favor, dime tu nombre después del tono.")
        print("[AUTO-REGISTRO] Iniciando secuencia de voz...")

    def guardar_asistencia(self, nombre, confianza):
        """Registra la asistencia en un archivo CSV aplicando cooldown temporal."""
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
        """
        Captura rostros automáticamente clasificándolos por distancias para evitar saturar el disco.
        Guarda un máximo de 5 fotos para Cerca, 5 para Medio y 5 para Lejos (Máximo 15 fotos interactivos).
        """
        ruta_usuario = os.path.join(DATASET_DIR, nombre)
        if not os.path.exists(ruta_usuario):
            return
            
        # Determinar categoría de distancia
        if w_face >= 120:
            distancia_tag = "cerca"
        elif w_face >= 60:
            distancia_tag = "medio"
        else:
            distancia_tag = "lejos"
            
        # Contar archivos de esta categoría en la carpeta
        archivos_categoria = [f for f in os.listdir(ruta_usuario) if f.startswith(f"{nombre}_auto_{distancia_tag}_")]
        if len(archivos_categoria) >= 5:
            return  # Ya tenemos suficientes muestras de esta distancia. Evitamos saturación.
            
        # Guardar muestra
        rostro_red = cv2.resize(rostro_gris, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
        index = len(archivos_categoria)
        archivo_nombre = f"{nombre}_auto_{distancia_tag}_{index}.jpg"
        cv2.imwrite(os.path.join(ruta_usuario, archivo_nombre), rostro_red)
        
        self.fotos_auto_guardadas += 1
        print(f"[APRENDIZAJE ONLINE] Guardada cara '{nombre}' en rango [{distancia_tag.upper()}] ({index + 1}/5)")
        
        # Reentrenar cuando se acumulen 10 fotos nuevas globalmente en segundo plano
        if self.fotos_auto_guardadas >= 10:
            self.fotos_auto_guardadas = 0
            print("[APRENDIZAJE ONLINE] Retrenando clasificador en segundo plano para asimilar nuevos ángulos...")
            self.entrenar_en_segundo_plano()

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
        print("Modo Autónomo e Interactivo:")
        print("  - Detección lejana mejorada mediante ecualización adaptativa CLAHE.")
        print("  - Auto-registro guiado por voz (Alexa Style) sin cerrar la cámara.")
        print("  - Muestreo selectivo inteligente por rango de distancias (evita saturar).")
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
            
            # CLAHE ecualización de contraste para capturar rostros lejanos/oscuros
            gris_opt = self.clahe.apply(gris)
            
            # Detección ultra-sensible (scaleFactor=1.08, minSize=20x20)
            caras = self.face_cascade.detectMultiScale(
                gris_opt, 
                scaleFactor=1.08, 
                minNeighbors=4, 
                minSize=(20, 20)
            )
            
            # --- ACTUALIZAR MÁQUINA DE ESTADOS DE REGISTRO (ALEXA STYLE) ---
            ahora = time.time()
            hud_color_global = (0, 255, 0)  # Verde normal
            hud_texto_global = f"IA: ESCANEO ACTIVO | FPS: {fps:.1f} | REGISTRADOS: {len(self.nombres_usuarios)}"
            
            if self.registro_estado is not None:
                hud_color_global = (0, 165, 255)  # Naranja
                
                # Estado 1: Esperando que termine el mensaje por voz
                if self.registro_estado == "preguntando":
                    hud_texto_global = "ALEXA IA: PREGUNTANDO NOMBRE..."
                    if ahora - self.registro_timer > 5.5:  # Termina de hablar
                        # Emitir Beep y lanzar hilo de grabación MCI
                        try:
                            winsound.Beep(1000, 300)
                        except:
                            pass
                        self.registro_estado = "escuchando"
                        self.registro_timer = ahora
                        self.input_nombre_resultado = None
                        threading.Thread(target=self.grabar_audio_mci_worker, daemon=True).start()
                
                # Estado 2: Grabando audio del micrófono en segundo plano
                elif self.registro_estado == "escuchando":
                    hud_texto_global = "ALEXA IA: ESCUCHANDO NOMBRE... [HABLE AHORA]"
                    # Dibujar un icono de "Grabando/Micrófono" en pantalla
                    cv2.circle(frame_original, (w_orig - 30, 25), 10, (0, 0, 255), -1)
                
                # Estado 3: Procesando la voz y traduciendo a texto
                elif self.registro_estado == "procesando_voz":
                    hud_texto_global = "ALEXA IA: PROCESANDO VOZ..."
                    if self.input_nombre_resultado is not None:
                        if self.input_nombre_resultado != "":  # Nombre obtenido correctamente
                            self.registro_nombre = self.input_nombre_resultado
                            self.registro_estado = "aviso_frente"
                            self.registro_timer = ahora
                            encolar_saludo(f"Perfecto {self.registro_nombre}. Mírame de frente por favor.")
                        else:
                            # Falla de voz, activar diálogo de texto no bloqueante
                            self.registro_estado = "esperando_popup"
                            self.input_nombre_resultado = None
                            encolar_saludo("No te escuché bien. Por favor escribe tu nombre en la ventana emergente.")
                            threading.Thread(target=self.solicitar_nombre_popup_worker, daemon=True).start()
                
                # Estado 4: Esperando diálogo de texto en segundo plano
                elif self.registro_estado == "esperando_popup":
                    hud_texto_global = "ALEXA IA: ESPERANDO NOMBRE EN POPUP..."
                
                # Estado 5: Guiar postura frontal
                elif self.registro_estado == "aviso_frente":
                    hud_texto_global = f"REGISTRO: MÍRAME DE FRENTE ({3 - int(ahora - self.registro_timer)}s)"
                    if ahora - self.registro_timer > 3.0:
                        self.registro_estado = "capturando_frente"
                        self.registro_capturas = 0
                        
                # Estado 6: Guardar fotos frontales
                elif self.registro_estado == "capturando_frente":
                    hud_texto_global = f"REGISTRO: CAPTURANDO FRENTE... [{self.registro_capturas}/3]"
                    if len(caras) > 0 and self.registro_capturas < 3:
                        x_peq, y_peq, w_peq, h_peq = caras[0]
                        rostro_crop = gris_opt[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                        rostro_red = cv2.resize(rostro_crop, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                        
                        ruta_usr = os.path.join(DATASET_DIR, self.registro_nombre)
                        if not os.path.exists(ruta_usr):
                            os.makedirs(ruta_usr)
                        cv2.imwrite(os.path.join(ruta_usr, f"{self.registro_nombre}_front_{self.registro_capturas}.jpg"), rostro_red)
                        self.registro_capturas += 1
                        time.sleep(0.15)
                    elif self.registro_capturas >= 3:
                        self.registro_estado = "aviso_izquierda"
                        self.registro_timer = ahora
                        encolar_saludo("Bien. Ahora gira tu cabeza a la izquierda.")

                # Estado 7: Guiar postura izquierda
                elif self.registro_estado == "aviso_izquierda":
                    hud_texto_global = f"REGISTRO: GIRA A LA IZQUIERDA ({3 - int(ahora - self.registro_timer)}s)"
                    if ahora - self.registro_timer > 3.0:
                        self.registro_estado = "capturando_izquierda"
                        self.registro_capturas = 0
                        
                # Estado 8: Guardar fotos perfil izquierdo
                elif self.registro_estado == "capturando_izquierda":
                    hud_texto_global = f"REGISTRO: CAPTURANDO PERFIL IZQ... [{self.registro_capturas}/3]"
                    if len(caras) > 0 and self.registro_capturas < 3:
                        x_peq, y_peq, w_peq, h_peq = caras[0]
                        rostro_crop = gris_opt[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                        rostro_red = cv2.resize(rostro_crop, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                        
                        ruta_usr = os.path.join(DATASET_DIR, self.registro_nombre)
                        cv2.imwrite(os.path.join(ruta_usr, f"{self.registro_nombre}_left_{self.registro_capturas}.jpg"), rostro_red)
                        self.registro_capturas += 1
                        time.sleep(0.15)
                    elif self.registro_capturas >= 3:
                        self.registro_estado = "aviso_derecha"
                        self.registro_timer = ahora
                        encolar_saludo("Por último, gira tu cabeza a la derecha.")

                # Estado 9: Guiar postura derecha
                elif self.registro_estado == "aviso_derecha":
                    hud_texto_global = f"REGISTRO: GIRA A LA DERECHA ({3 - int(ahora - self.registro_timer)}s)"
                    if ahora - self.registro_timer > 3.0:
                        self.registro_estado = "capturando_derecha"
                        self.registro_capturas = 0
                        
                # Estado 10: Guardar fotos perfil derecho
                elif self.registro_estado == "capturando_derecha":
                    hud_texto_global = f"REGISTRO: CAPTURANDO PERFIL DER... [{self.registro_capturas}/3]"
                    if len(caras) > 0 and self.registro_capturas < 3:
                        x_peq, y_peq, w_peq, h_peq = caras[0]
                        rostro_crop = gris_opt[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                        rostro_red = cv2.resize(rostro_crop, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                        
                        ruta_usr = os.path.join(DATASET_DIR, self.registro_nombre)
                        cv2.imwrite(os.path.join(ruta_usr, f"{self.registro_nombre}_right_{self.registro_capturas}.jpg"), rostro_red)
                        self.registro_capturas += 1
                        time.sleep(0.15)
                    elif self.registro_capturas >= 3:
                        # Guardado final y entrenamiento express
                        self.entrenar_modelo()
                        encolar_saludo(f"Registro completado. Bienvenido al sistema, {self.registro_nombre}.")
                        self.consecutivos_desconocidos = 0
                        self.memoria_deteccion.clear()
                        self.registro_estado = None

            # Pintar barra superior global de estadísticas
            cv2.rectangle(frame_original, (0, 0), (w_orig, 40), (20, 20, 20), -1)
            cv2.line(frame_original, (0, 40), (w_orig, 40), hud_color_global, 1)
            cv2.putText(frame_original, hud_texto_global, (15, 26), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color_global, 1, cv2.LINE_AA)
            
            # Cartel informativo si no hay usuarios
            if len(self.nombres_usuarios) == 0 and self.registro_estado is None:
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 0, 150), -1)
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 165, 255), 2)
                cv2.putText(frame_original, "MODO APRENDIZAJE INICIAL", (115, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_original, "Párate frente a la cámara para auto-registrarte", (95, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
            cara_detectada_este_frame = len(caras) > 0
            
            # --- PROCESAR ROSTROS DETECTADOS ---
            for (x_peq, y_peq, w_peq, h_peq) in caras:
                x = x_peq * escala
                y = y_peq * escala
                w = w_peq * escala
                h = h_peq * escala
                
                etiqueta = "Analizando..."
                subtitulo = "Leyendo rostro..."
                color = (0, 255, 255)
                
                cara_gris_recortada = gris_opt[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                
                # Solo clasificar si no estamos en medio de un registro activo
                if self.registro_estado is None:
                    if self.modelo_cargado:
                        try:
                            # Escalado bicúbico para máxima definición de rostros pequeños/lejanos
                            cara_gris_norm = cv2.resize(cara_gris_recortada, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                            id_prediccion, distancia = self.recognizer.predict(cara_gris_norm)
                            confianza_pct = max(0, 100 - distancia)
                            
                            # Umbral flexible optimizado para distancias
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
                                subtitulo = f"Match: {confianza_pct:.1f}%"
                                color = (0, 255, 0)  # Verde
                                
                                self.consecutivos_desconocidos = 0
                                
                                # Saludar y guardar asistencia
                                encolar_saludo(f"Hola {voto_ganador}, bienvenido.")
                                self.guardar_asistencia(voto_ganador, confianza_pct)
                                
                                # Capturar de forma selectiva según distancia (máximo 5 fotos por distancia)
                                ahora_t = time.time()
                                if ahora_t - ultimo_guardado_interactivo > 3.0:
                                    self.auto_capturar_rostro_interaccion(voto_ganador, cara_gris_recortada, w)
                                    ultimo_guardado_interactivo = ahora_t
                            else:
                                etiqueta = "DESCONOCIDO"
                                subtitulo = "Identificando..."
                                color = (0, 0, 255)  # Rojo
                                self.consecutivos_desconocidos += 1
                                
                        except Exception as e:
                            subtitulo = f"Error IA: {e}"
                            color = (0, 0, 255)
                    else:
                        etiqueta = "NUEVA PERSONA"
                        subtitulo = "Auto-registro por voz..."
                        color = (0, 165, 255)
                        self.consecutivos_desconocidos += 1
                    
                    # DISPARAR AUTO-REGISTRO
                    # Si detecta rostro desconocido durante 45 frames seguidos, inicia máquina de estados de voz
                    if self.consecutivos_desconocidos >= 45:
                        self.iniciar_registro_autonomo()
                else:
                    # En modo registro activo, dibujar un HUD de análisis en color naranja sobre las caras
                    etiqueta = "CONFIGURACIÓN IA"
                    subtitulo = "Registrando posturas..."
                    color = (0, 165, 255)
                
                self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
            
            # Decrementar contador de desconocidos en ausencia de caras para evitar disparos en falsos positivos rápidos
            if not cara_detectada_este_frame and self.consecutivos_desconocidos > 0:
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
    print("SISTEMA DE VISIÓN CON IA: AUTO-REGISTRO POR VOZ NATIVO & HUD FLUIDO")
    print("=" * 60)
    
    try:
        app = SistemaReconocimientoFacial()
        app.iniciar_bucle_principal()
    except Exception as e:
        print(f"\n[CRÍTICO] Error al inicializar la aplicación: {e}")
        input("Presione ENTER para salir...")
