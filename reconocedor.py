import cv2
import numpy as np
import os
import time
import csv
import threading
import pyttsx3
from collections import deque
import tkinter as tk
from tkinter import simpledialog

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================
DATASET_DIR = "dataset"
TRAINER_FILE = "clasificador.yml"
ATTENDANCE_FILE = "asistencia.csv"
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
WIDTH_RESIZE = 150  # Tamaño al que se redimensionan las caras
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
        
        # Historial de detecciones para estabilizar predicciones (Evita parpadeos)
        self.memoria_deteccion = deque(maxlen=5)
        
        # Historial de rostros consecutivos no identificados para Auto-Registro
        # Si se detecta un rostro y durante 45 frames seguidos es "Desconocido", se dispara el registro
        self.consecutivos_desconocidos = 0
        self.is_registering = False  # Flag para pausar la IA mientras se registra
        
        # Lock de entrenamiento para evitar condiciones de carrera en multihilo
        self.entrenamiento_lock = threading.Lock()
        
        # Diccionario para contar las fotos auto-capturadas y evitar excesos
        self.fotos_auto_guardadas = 0
        
        # Cooldown temporal de registros en CSV (10 segundos)
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
                        img_gris = cv2.resize(img_gris, (WIDTH_RESIZE, HEIGHT_RESIZE))
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

    def preguntar_nombre_y_registrar(self, cap, rostro_inicial_gris):
        """Secuencia de auto-registro guiada por voz mediante un diálogo Tkinter no bloqueante."""
        self.is_registering = True
        
        # Detener la detección sumando voz
        print("[AUTO-REGISTRO] Se ha detectado una persona nueva.")
        encolar_saludo("Hola. Veo que eres una persona nueva. ¿Cómo te llamas? Por favor introduce tu nombre en la pantalla.")
        
        # Usar Tkinter emergente para solicitar el nombre (solución limpia y libre de compilaciones complejas en Windows)
        root = tk.Tk()
        root.withdraw()  # Ocultar la ventana de fondo de Tkinter
        root.attributes("-topmost", True)  # Traer la caja al frente
        
        nombre = simpledialog.askstring("Identificación Inteligente", "¿Cuál es tu nombre?", parent=root)
        root.destroy()
        
        if not nombre:
            print("[AUTO-REGISTRO] Cancelado o vacío. Reanudando...")
            self.consecutivos_desconocidos = 0
            self.is_registering = False
            return
            
        nombre = nombre.strip().replace(" ", "_")
        ruta_usuario = os.path.join(DATASET_DIR, nombre)
        if not os.path.exists(ruta_usuario):
            os.makedirs(ruta_usuario)
            
        # Guardar la foto actual que causó la detección
        rostro_red = cv2.resize(rostro_inicial_gris, (WIDTH_RESIZE, HEIGHT_RESIZE))
        cv2.imwrite(os.path.join(ruta_usuario, f"{nombre}_auto_0.jpg"), rostro_red)
        
        print(f"[AUTO-REGISTRO] Registrando a: {nombre}")
        encolar_saludo(f"Perfecto {nombre}. Por favor mira a la cámara para tomar tus fotos de registro.")
        
        # Captura express de 30 fotos
        capturados = 1
        start_time = time.time()
        
        while capturados < 30:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_visual = frame.copy()
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = self.face_cascade.detectMultiScale(gris, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
            
            for (x, y, w, h) in caras:
                if capturados < 30:
                    capturados += 1
                    rostro_recortado = gris[y:y+h, x:x+w]
                    rostro_recortado = cv2.resize(rostro_recortado, (WIDTH_RESIZE, HEIGHT_RESIZE))
                    
                    archivo_path = os.path.join(ruta_usuario, f"{nombre}_auto_{capturados}.jpg")
                    cv2.imwrite(archivo_path, rostro_recortado)
                    
                    # HUD de Captura
                    cv2.rectangle(frame_visual, (x, y), (x+w, y+h), (0, 165, 255), 2)
                    cv2.putText(frame_visual, f"Capturando: {capturados}/30", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Panel superior
            cv2.rectangle(frame_visual, (0, 0), (frame_visual.shape[1], 40), (20, 20, 20), -1)
            cv2.putText(frame_visual, f"AUTO-REGISTRO: {nombre.upper()} | FOTOS: {capturados}/30", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            cv2.imshow("Antigravity Smart Recognition HUD", frame_visual)
            cv2.waitKey(60)
            
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
            
        # Contar fotos existentes para evitar saturar el disco
        fotos_existentes = len([f for f in os.listdir(ruta_usuario) if f.endswith('.jpg')])
        if fotos_existentes >= MAX_FOTOS_POR_USUARIO:
            return
            
        # Guardar nueva muestra con timestamp
        rostro_red = cv2.resize(rostro_gris, (WIDTH_RESIZE, HEIGHT_RESIZE))
        timestamp = int(time.time() * 100)
        archivo_nombre = f"{nombre}_interactivo_{timestamp}.jpg"
        cv2.imwrite(os.path.join(ruta_usuario, archivo_nombre), rostro_red)
        
        self.fotos_auto_guardadas += 1
        print(f"[APRENDIZAJE ONLINE] Nueva muestra guardada para '{nombre}' ({fotos_existentes + 1} fotos totales)")
        
        # Cada vez que guardamos 10 fotos nuevas en total interactuando, reentrenamos la IA en segundo plano
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
        total_usuarios = len(self.nombres_usuarios)
        
        # Contadores para el aprendizaje interactivo continuo (cooldown de fotos auto-guardadas)
        ultimo_guardado_interactivo = 0
        
        print("\nSISTEMA INTELIGENTE DE RECONOCIMIENTO FACIAL INICIADO.")
        print("Modo Autónomo Activo:")
        print("  - El sistema detectará personas nuevas automáticamente tras 3 segundos.")
        print("  - El sistema capturará nuevos ángulos del rostro dinámicamente durante interacciones.")
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
            gris = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)
            
            # Detección rápida
            caras = self.face_cascade.detectMultiScale(
                gris, 
                scaleFactor=1.15, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Barra de estadísticas global
            cv2.rectangle(frame_original, (0, 0), (w_orig, 40), (20, 20, 20), -1)
            cv2.line(frame_original, (0, 40), (w_orig, 40), (0, 255, 0), 1)
            
            total_usuarios = len(self.nombres_usuarios)
            texto_status = f"IA: AUTO-APRENDIZAJE ACTIVO | FPS: {fps:.1f} | REGISTRADOS: {total_usuarios}"
            cv2.putText(frame_original, texto_status, (15, 26), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            if total_usuarios == 0:
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 0, 150), -1)
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 165, 255), 2)
                cv2.putText(frame_original, "MODO APRENDIZAJE INICIAL", (115, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_original, "Párate frente a la cámara para auto-registrarte", (110, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cara_detectada_este_frame = len(caras) > 0
            
            for (x_peq, y_peq, w_peq, h_peq) in caras:
                x = x_peq * escala
                y = y_peq * escala
                w = w_peq * escala
                h = h_peq * escala
                
                etiqueta = "Analizando..."
                subtitulo = "Leyendo rostro..."
                color = (0, 255, 255)  # Amarillo (Procesando)
                
                cara_gris_recortada = gris[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                
                if self.modelo_cargado:
                    try:
                        cara_gris_norm = cv2.resize(cara_gris_recortada, (WIDTH_RESIZE, HEIGHT_RESIZE))
                        id_prediccion, distancia = self.recognizer.predict(cara_gris_norm)
                        confianza_pct = max(0, 100 - distancia)
                        
                        if confianza_pct > 40:
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
                            # Rostro conocido
                            etiqueta = f"ACTIVO: {voto_ganador.upper()}"
                            subtitulo = f"Match: {confianza_pct:.1f}%"
                            color = (0, 255, 0)  # Verde
                            
                            self.consecutivos_desconocidos = 0  # Resetear contador de desconocidos
                            
                            # Saludar por voz y persistir en CSV
                            encolar_saludo(f"Hola {voto_ganador}, bienvenido.")
                            self.guardar_asistencia(voto_ganador, confianza_pct)
                            
                            # --- APRENDIZAJE ONLINE CONTINUO (Mejora de Precisión) ---
                            # Guardamos una nueva muestra de cara cada 50 frames (aprox 3 segundos) para mejorar la precisión
                            ahora = time.time()
                            if ahora - ultimo_guardado_interactivo > 3.0:
                                self.auto_capturar_rostro_interaccion(voto_ganador, cara_gris_recortada)
                                ultimo_guardado_interactivo = ahora
                        else:
                            # Rostro desconocido
                            etiqueta = "DESCONOCIDO"
                            subtitulo = "Identificando..."
                            color = (0, 0, 255)  # Rojo
                            
                            self.consecutivos_desconocidos += 1
                            
                    except Exception as e:
                        subtitulo = f"Error IA: {e}"
                        color = (0, 0, 255)
                else:
                    # Si no hay modelo entrenado aún
                    etiqueta = "NUEVA PERSONA"
                    subtitulo = "Auto-registro inminente..."
                    color = (0, 165, 255)
                    self.consecutivos_desconocidos += 1
                
                # --- AUTO-REGISTRO ---
                # Si una persona no está registrada o es desconocida durante 45 frames seguidos (aprox. 3 segundos),
                # el sistema entra automáticamente al modo registro y le pregunta por voz cómo se llama.
                if self.consecutivos_desconocidos >= 45:
                    self.preguntar_nombre_y_registrar(cap, cara_gris_recortada)
                    break  # Romper para evitar procesar frames inválidos
                
                self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
            
            # Si no hay caras en escena, decrementamos lentamente el contador de desconocidos para tolerar parpadeos de detección
            if not cara_detectada_este_frame and self.consecutivos_desconocidos > 0:
                self.consecutivos_desconocidos -= 1
            
            cv2.imshow("Antigravity Smart Recognition HUD", frame_original)
            
            # Medir FPS
            t_fin = time.time()
            fps = 1.0 / (t_fin - t_inicio)
            
            # Salir del bucle con 'Q'
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
    print("SISTEMA DE VISIÓN CON IA: AUTO-APRENDIZAJE Y REGISTRO POR VOZ")
    print("=" * 60)
    
    try:
        app = SistemaReconocimientoFacial()
        app.iniciar_bucle_principal()
    except Exception as e:
        print(f"\n[CRÍTICO] Error al inicializar la aplicación: {e}")
        input("Presione ENTER para salir...")
