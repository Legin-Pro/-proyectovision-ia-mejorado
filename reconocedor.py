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

# Límites de imágenes por usuario para evitar saturar el disco
MAX_FOTOS_POR_USUARIO = 150

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
        
        # Historial de rostros consecutivos no identificados para Auto-Registro
        self.consecutivos_desconocidos = 0
        self.is_registering = False  # Flag para pausar la IA mientras se registra
        
        # Lock de entrenamiento para evitar condiciones de carrera en multihilo
        self.entrenamiento_lock = threading.Lock()
        
        # Contador de fotos de interacción
        self.fotos_auto_guardadas = 0
        
        # Cooldown temporal de registros en CSV (15 segundos)
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
        with self.entrenamiento_lock:
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
                        # Redimensionar usando interpolación bicúbica para mayor definición de texturas faciales
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
        """Ejecuta el entrenamiento en un hilo para evitar que se congele el video en tiempo real."""
        hilo = threading.Thread(target=self.entrenar_modelo, daemon=True)
        hilo.start()

    def grabar_audio_mci(self, duracion=4):
        """Usa el MCI nativo de Windows (vía ctypes) para grabar del micrófono sin depender de PyAudio."""
        winmm = ctypes.windll.winmm
        wav_path = "registro_voz_temp.wav"
        
        # Cerrar alias previo por seguridad
        winmm.mciSendStringW("close recsound", None, 0, 0)
        
        # Abrir dispositivo waveaudio y comenzar grabación
        winmm.mciSendStringW("open new type waveaudio alias recsound", None, 0, 0)
        winmm.mciSendStringW("record recsound", None, 0, 0)
        
        print(f"[MICROFONO] Grabando durante {duracion} segundos...")
        time.sleep(duracion)
        
        # Detener y guardar archivo temporal
        winmm.mciSendStringW("stop recsound", None, 0, 0)
        winmm.mciSendStringW(f"save recsound {wav_path}", None, 0, 0)
        winmm.mciSendStringW("close recsound", None, 0, 0)
        
        return wav_path

    def escuchar_nombre_por_voz(self):
        """Pregunta por voz y procesa la grabación usando SpeechRecognition y Google Speech API."""
        encolar_saludo("Hola, detecto que eres alguien nuevo. ¿Cómo te llamas? Por favor, dime tu nombre después del tono.")
        time.sleep(5.5)  # Esperar a que el asistente termine de hablar
        
        # Sonar pitido para avisar al usuario que comience a hablar
        try:
            winsound.Beep(1000, 300)
        except:
            pass
            
        wav_file = self.grabar_audio_mci(duracion=4)
        
        nombre_transcrito = None
        r = sr.Recognizer()
        
        if os.path.exists(wav_file):
            try:
                with sr.AudioFile(wav_file) as source:
                    audio_data = r.record(source)
                    # Transcribir audio a texto (API de Google gratuita y rápida)
                    transcripcion = r.recognize_google(audio_data, language="es-ES")
                    print(f"[MICROFONO] Voz entendida: {transcripcion}")
                    
                    # Limpiar texto para extraer solo el nombre principal
                    palabras = transcripcion.strip().split()
                    if palabras:
                        # Si dice "Me llamo Carlos", tomamos "Carlos"
                        if len(palabras) >= 3 and palabras[0].lower() in ["me", "mi"] and palabras[1].lower() in ["llamo", "nombre"]:
                            nombre_transcrito = palabras[2]
                        else:
                            nombre_transcrito = palabras[0]
            except sr.UnknownValueError:
                print("[MICROFONO] No se entendió el nombre.")
            except Exception as e:
                print(f"[MICROFONO ERROR] {e}")
            finally:
                try:
                    os.remove(wav_file)
                except:
                    pass
                    
        return nombre_transcrito

    def preguntar_nombre_y_registrar(self, cap, rostro_inicial_gris):
        """Pregunta el nombre por micrófono. Si falla, cae al popup de Tkinter como backup."""
        self.is_registering = True
        cv2.destroyAllWindows()
        
        # Intentar obtener el nombre por voz
        nombre = self.escuchar_nombre_por_voz()
        
        # Fallback de seguridad: Si no se entendió por voz, abrir cajita de diálogo Tkinter
        if not nombre:
            print("[AUTO-REGISTRO] No se pudo capturar por voz. Abriendo cajita de texto...")
            encolar_saludo("No logré escucharte bien. Por favor escribe tu nombre en la ventana emergente.")
            
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            nombre = simpledialog.askstring("Identificación Inteligente", "Escribe tu nombre por favor:", parent=root)
            root.destroy()
            
        if not nombre:
            print("[AUTO-REGISTRO] Cancelado. Reanudando...")
            self.consecutivos_desconocidos = 0
            self.is_registering = False
            return
            
        nombre = nombre.strip().replace(" ", "_").capitalize()
        ruta_usuario = os.path.join(DATASET_DIR, nombre)
        if not os.path.exists(ruta_usuario):
            os.makedirs(ruta_usuario)
            
        # Guardar la muestra inicial que gatilló el registro
        rostro_red = cv2.resize(rostro_inicial_gris, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(ruta_usuario, f"{nombre}_auto_0.jpg"), rostro_red)
        
        print(f"[AUTO-REGISTRO] Guardando a: {nombre}")
        encolar_saludo(f"Perfecto {nombre}. Por favor mira a la cámara mientras tomo tus capturas.")
        
        # Captura express de 30 fotos
        capturados = 1
        
        while capturados < 35:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_visual = frame.copy()
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detección ultra-sensible para el registro
            caras = self.face_cascade.detectMultiScale(gris, scaleFactor=1.08, minNeighbors=4, minSize=(25, 25))
            
            for (x, y, w, h) in caras:
                if capturados < 35:
                    capturados += 1
                    rostro_recortado = gris[y:y+h, x:x+w]
                    rostro_recortado = cv2.resize(rostro_recortado, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                    
                    archivo_path = os.path.join(ruta_usuario, f"{nombre}_auto_{capturados}.jpg")
                    cv2.imwrite(archivo_path, rostro_recortado)
                    
                    # HUD de Captura
                    cv2.rectangle(frame_visual, (x, y), (x+w, y+h), (0, 165, 255), 2)
                    cv2.putText(frame_visual, f"Capturando: {capturados}/35", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Panel superior
            cv2.rectangle(frame_visual, (0, 0), (frame_visual.shape[1], 40), (20, 20, 20), -1)
            cv2.putText(frame_visual, f"AUTO-REGISTRO: {nombre.upper()} | FOTOS: {capturados}/35", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            cv2.imshow("Antigravity Smart Recognition HUD", frame_visual)
            cv2.waitKey(80)
            
        # Entrenar de inmediato
        self.entrenar_modelo()
        encolar_saludo(f"Registro completo. Bienvenido al sistema, {nombre}.")
        
        # Resetear contadores de detección para volver a la normalidad
        self.consecutivos_desconocidos = 0
        self.memoria_deteccion.clear()
        self.is_registering = False

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

    def auto_capturar_rostro_interaccion(self, nombre, rostro_gris):
        """Captura rostros de forma automática e interactiva para mejorar la precisión con el tiempo."""
        ruta_usuario = os.path.join(DATASET_DIR, nombre)
        if not os.path.exists(ruta_usuario):
            return
            
        fotos_existentes = len([f for f in os.listdir(ruta_usuario) if f.endswith('.jpg')])
        if fotos_existentes >= MAX_FOTOS_POR_USUARIO:
            return
            
        # Redimensionado bicúbico
        rostro_red = cv2.resize(rostro_gris, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
        timestamp = int(time.time() * 100)
        archivo_nombre = f"{nombre}_interactivo_{timestamp}.jpg"
        cv2.imwrite(os.path.join(ruta_usuario, archivo_nombre), rostro_red)
        
        self.fotos_auto_guardadas += 1
        print(f"[APRENDIZAJE ONLINE] Nueva muestra guardada para '{nombre}' ({fotos_existentes + 1} fotos totales)")
        
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

        # Transparencia y Escáner
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
        
        print("\nSISTEMA INTELIGENTE DE RECONOCIMIENTO FACIAL INICIADO (DISTANCIA OPTIMIZADA).")
        print("Modo Autónomo Activo:")
        print("  - Detección lejana mejorada mediante ecualización adaptativa CLAHE.")
        print("  - Auto-registro interactivo por voz y micrófono nativo.")
        print("  - Presione 'Q' en la pantalla del video para salir.")
        print("-" * 60)
        
        while True:
            if self.is_registering:
                time.sleep(0.1)
                continue
                
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
            
            # Convertir a escala de grises
            gris = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)
            
            # --- MEJORA DE DETECCIÓN LEJANA (CLAHE) ---
            # Aplicar ecualización adaptativa de histograma para mejorar el contraste del rostro lejano/sombreado
            gris_opt = self.clahe.apply(gris)
            
            # Detección ultra-sensible (scaleFactor reducido a 1.08, minSize a 20x20 para rostros lejanos)
            caras = self.face_cascade.detectMultiScale(
                gris_opt, 
                scaleFactor=1.08, 
                minNeighbors=4, 
                minSize=(20, 20)
            )
            
            # Barra de estadísticas global
            cv2.rectangle(frame_original, (0, 0), (w_orig, 40), (20, 20, 20), -1)
            cv2.line(frame_original, (0, 40), (w_orig, 40), (0, 255, 0), 1)
            
            total_usuarios = len(self.nombres_usuarios)
            texto_status = f"IA: SENSITIVIDAD DE LARGA DISTANCIA ACTIVA | FPS: {fps:.1f} | REGISTRADOS: {total_usuarios}"
            cv2.putText(frame_original, texto_status, (15, 26), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            if total_usuarios == 0:
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 0, 150), -1)
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 165, 255), 2)
                cv2.putText(frame_original, "MODO APRENDIZAJE INICIAL", (115, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_original, "Párate frente a la cámara para iniciar auto-registro por voz", (95, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
            cara_detectada_este_frame = len(caras) > 0
            
            for (x_peq, y_peq, w_peq, h_peq) in caras:
                x = x_peq * escala
                y = y_peq * escala
                w = w_peq * escala
                h = h_peq * escala
                
                etiqueta = "Analizando..."
                subtitulo = "Leyendo rostro..."
                color = (0, 255, 255)
                
                # Recortar rostro usando el frame gris original (o el opt)
                cara_gris_recortada = gris_opt[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                
                if self.modelo_cargado:
                    try:
                        # Redimensionar usando interpolación bicúbica para mantener los detalles faciales desde lejos
                        cara_gris_norm = cv2.resize(cara_gris_recortada, (WIDTH_RESIZE, HEIGHT_RESIZE), interpolation=cv2.INTER_CUBIC)
                        id_prediccion, distancia = self.recognizer.predict(cara_gris_norm)
                        confianza_pct = max(0, 100 - distancia)
                        
                        if confianza_pct > 38:  # Umbral ligeramente más flexible para detecciones lejanas
                            nombre_detectado = self.nombres_usuarios.get(id_prediccion, "Desconocido")
                            self.memoria_deteccion.append(nombre_detectado)
                        else:
                            self.memoria_deteccion.append("Desconocido")
                            
                        # Voto ganador
                        if len(self.memoria_deteccion) > 0:
                            voto_ganador = max(set(self.memoria_deteccion), key=self.memoria_deteccion.count)
                        else:
                            voto_ganador = "Desconocido"
                            
                        if voto_ganador != "Desconocido":
                            etiqueta = f"ACTIVO: {voto_ganador.upper()}"
                            subtitulo = f"Match: {confianza_pct:.1f}%"
                            color = (0, 255, 0)
                            
                            self.consecutivos_desconocidos = 0
                            
                            # Saludar por voz y persistir
                            encolar_saludo(f"Hola {voto_ganador}, bienvenido.")
                            self.guardar_asistencia(voto_ganador, confianza_pct)
                            
                            # Aprendizaje continuo interactivo (nueva foto cada 3 segundos si el usuario se mantiene en cámara)
                            ahora = time.time()
                            if ahora - ultimo_guardado_interactivo > 3.0:
                                self.auto_capturar_rostro_interaccion(voto_ganador, cara_gris_recortada)
                                ultimo_guardado_interactivo = ahora
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
                    subtitulo = "Auto-registro por voz..."
                    color = (0, 165, 255)
                    self.consecutivos_desconocidos += 1
                
                # AUTO-REGISTRO
                # Si una persona no está registrada o es desconocida durante 45 frames seguidos,
                # iniciamos el flujo por voz interactivo.
                if self.consecutivos_desconocidos >= 45:
                    self.preguntar_nombre_y_registrar(cap, cara_gris_recortada)
                    break
                
                self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
            
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
    print("SISTEMA DE VISIÓN CON IA: AUTO-REGISTRO E INTERACCIÓN POR VOZ NATIVA")
    print("=" * 60)
    
    try:
        app = SistemaReconocimientoFacial()
        app.iniciar_bucle_principal()
    except Exception as e:
        print(f"\n[CRÍTICO] Error al inicializar la aplicación: {e}")
        input("Presione ENTER para salir...")
