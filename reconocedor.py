import cv2
import numpy as np
import os
import time
import csv
import threading
import pyttsx3
from collections import deque

# =====================================================================
# CONFIGURACIÓN Y CONSTANTES
# =====================================================================
DATASET_DIR = "dataset"
TRAINER_FILE = "clasificador.yml"
ATTENDANCE_FILE = "asistencia.csv"
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
WIDTH_RESIZE = 150  # Tamaño al que se redimensionan las caras para el entrenamiento
HEIGHT_RESIZE = 150

# Asegurar que las carpetas base existan
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Hilo para la reproducción de voz (evita que el bucle de video se congele)
voz_lock = threading.Lock()
cola_saludos = deque(maxlen=5)

def hablar_worker():
    """Ejecuta en segundo plano la síntesis de voz para no interrumpir los FPS del video."""
    # Inicializar el motor de voz en este hilo
    try:
        engine = pyttsx3.init()
        # Configurar idioma en español si está disponible
        voices = engine.getProperty('voices')
        for voice in voices:
            if "spanish" in voice.name.lower() or "es" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 160)  # Velocidad de voz moderada
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

def encolar_saludo(nombre):
    """Encola un saludo si no está repetido recientemente."""
    if not cola_saludos or cola_saludos[-1] != nombre:
        if nombre == "Desconocido":
            cola_saludos.append("Persona no registrada detectada. Presiona la tecla R para registrarte.")
        else:
            cola_saludos.append(f"Hola {nombre}, bienvenido.")

# =====================================================================
# CLASE PRINCIPAL: RECONOCEDOR Y REGISTRADOR FACIAL
# =====================================================================
class SistemaReconocimientoFacial:
    def __init__(self):
        # Cargar clasificador de rostros (Haar Cascade)
        if not os.path.exists(HAAR_CASCADE_PATH):
            raise FileNotFoundError(f"No se encontró Haar Cascade en {HAAR_CASCADE_PATH}")
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        
        # Inicializar el reconocedor LBPH
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.modelo_cargado = False
        self.nombres_usuarios = {}
        
        # Cargar modelo entrenado si existe
        self.cargar_modelo()
        
        # Historial de detecciones para estabilizar predicciones (Evita parpadeos)
        self.memoria_deteccion = deque(maxlen=7)
        
        # Registro de última vez que se guardó asistencia para evitar duplicados en CSV (10 segundos de cooldown)
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
            print("[INFO] No se encontró un modelo entrenado previamente. El sistema iniciará en modo lectura/registro.")
            self.modelo_cargado = False
            
        # Reconstruir diccionario de nombres basándose en la estructura de dataset/
        self.actualizar_mapeo_nombres()

    def actualizar_mapeo_nombres(self):
        """Asigna un ID numérico a cada carpeta de usuario."""
        if not os.path.exists(DATASET_DIR):
            return
        
        subdirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        self.nombres_usuarios = {i: name for i, name in enumerate(subdirs)}
        print(f"[INFO] Usuarios registrados en el sistema: {list(self.nombres_usuarios.values())}")

    def entrenar_modelo(self):
        """Entrena el modelo LBPH de OpenCV con las imágenes en dataset/."""
        print("[ENTRENAMIENTO] Iniciando entrenamiento de la IA...")
        rostros = []
        ids = []
        
        self.actualizar_mapeo_nombres()
        
        # Mapeo inverso de nombre a ID
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
                    # Asegurar tamaño uniforme
                    img_gris = cv2.resize(img_gris, (WIDTH_RESIZE, HEIGHT_RESIZE))
                    rostros.append(img_gris)
                    ids.append(usuario_id)
        
        if len(rostros) == 0:
            print("[WARN] No hay datos suficientes para entrenar. Añada usuarios primero.")
            self.modelo_cargado = False
            if os.path.exists(TRAINER_FILE):
                os.remove(TRAINER_FILE)
            return False
            
        try:
            self.recognizer.train(rostros, np.array(ids))
            self.recognizer.write(TRAINER_FILE)
            self.modelo_cargado = True
            print(f"[ENTRENAMIENTO] Modelo guardado con éxito como '{TRAINER_FILE}'!")
            return True
        except Exception as e:
            print(f"[ERROR] Ocurrió un error al entrenar el modelo: {e}")
            return False

    def registrar_nuevo_usuario(self, frame_actual, cap):
        """Detiene el flujo para registrar a una persona capturando 30 fotos de su rostro."""
        cv2.destroyAllWindows()
        print("\n" + "="*50)
        print("         MODO REGISTRO DE NUEVO USUARIO")
        print("="*50)
        
        nombre = input("Ingrese el nombre de la persona a registrar: ").strip().replace(" ", "_")
        if not nombre:
            print("[ERROR] Nombre inválido. Registro cancelado.")
            return
            
        ruta_usuario = os.path.join(DATASET_DIR, nombre)
        if not os.path.exists(ruta_usuario):
            os.makedirs(ruta_usuario)
            
        print(f"\nPreparando cámara para registrar a: {nombre}")
        print("Por favor, mire a la cámara de frente y muévase ligeramente.")
        print("Capturando 30 fotogramas...")
        time.sleep(1.5)
        
        capturados = 0
        
        # Volver a capturar frames específicos para el registro
        while capturados < 30:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo leer de la cámara.")
                break
                
            frame_visual = frame.copy()
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detectar en escala de grises optimizada
            caras = self.face_cascade.detectMultiScale(gris, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            
            for (x, y, w, h) in caras:
                capturados += 1
                # Guardar el rostro recortado
                rostro_recortado = gris[y:y+h, x:x+w]
                rostro_recortado = cv2.resize(rostro_recortado, (WIDTH_RESIZE, HEIGHT_RESIZE))
                
                archivo_path = os.path.join(ruta_usuario, f"{nombre}_{capturados}.jpg")
                cv2.imwrite(archivo_path, rostro_recortado)
                
                # Dibujar HUD indicador en la captura
                cv2.rectangle(frame_visual, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(frame_visual, f"Capturando: {capturados}/30", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
            # Panel superior en el frame visual de captura
            cv2.rectangle(frame_visual, (0, 0), (frame_visual.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame_visual, f"REGISTRANDO A: {nombre.upper()} | FOTOS: {capturados}/30", (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            cv2.imshow("Registro Facial - Mire a la Camara", frame_visual)
            
            # Pequeño retardo entre capturas para variedad angular
            if cv2.waitKey(80) & 0xFF == ord('q'):
                print("[REGISTRO] Cancelado por el usuario.")
                break
                
        cv2.destroyAllWindows()
        
        if capturados >= 10:
            print(f"[REGISTRO] Captura finalizada con {capturados} imágenes.")
            # Entrenar de inmediato
            exito = self.entrenar_modelo()
            if exito:
                encolar_saludo(nombre)
        else:
            print("[REGISTRO] No se capturaron suficientes imágenes. Registro fallido.")

    def registrar_en_csv(self, nombre, confianza):
        """Registra la asistencia en un archivo CSV aplicando cooldown temporal."""
        ahora = time.time()
        # Evitar re-registrar al mismo usuario en menos de 10 segundos
        if nombre in self.ultimo_registro_asistencia:
            if ahora - self.ultimo_registro_asistencia[nombre] < 10:
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
            print(f"[LOG ASISTENCIA] Registro guardado para {nombre} a las {hora_actual} (Confianza: {confianza:.2f}%)")
        except Exception as e:
            print(f"[ERROR CSV] No se pudo escribir en {ATTENDANCE_FILE}: {e}")

    def dibujar_hud_futurista(self, frame, x, y, w, h, etiqueta, subtitulo, color):
        """Dibuja brackets esquineros, barra de fondo y escáner sobre el rostro."""
        # 1. Dibujar brackets esquineros (Sci-Fi)
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

        # 2. Rectángulo interno con transparencia sutil (efecto overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 1)
        
        # Línea de escaneo láser dinámico
        velocidad_escaner = int((time.time() * 200) % h)
        y_escaner = y + velocidad_escaner
        cv2.line(overlay, (x, y_escaner), (x + w, y_escaner), color, 2)
        
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 3. Cuadro de texto de fondo para la etiqueta
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
        cv2.putText(frame, etiqueta, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Mostrar confianza o subtítulo debajo
        cv2.putText(frame, subtitulo, (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def iniciar_bucle_principal(self):
        """Inicia el streaming de la cámara y el reconocimiento facial."""
        print("[SISTEMA] Inicializando captura de video...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CRÍTICO] No se pudo acceder a la cámara. Verifique la conexión.")
            return

        # Ajustar resolución de captura interna
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_prev = 0
        fps = 0
        
        print("\nSISTEMA INICIADO CORRECTAMENTE.")
        print("Comandos disponibles:")
        print("  - Presione 'R' para registrar una nueva persona.")
        print("  - Presione 'Q' para salir de la aplicación.")
        print("-" * 50)
        
        while True:
            t_inicio = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Error al recibir fotograma. Reintentando...")
                time.sleep(0.1)
                continue
                
            frame_original = frame.copy()
            h_orig, w_orig = frame.shape[:2]
            
            # --- OPTIMIZACIÓN DE RENDIMIENTO (FPS) ---
            # Reducir imagen a 1/2 tamaño para detección rápida
            escala = 2
            ancho_red = w_orig // escala
            alto_red = h_orig // escala
            frame_pequeno = cv2.resize(frame, (ancho_red, alto_red))
            gris = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2GRAY)
            
            # Detectar caras en la imagen pequeña
            caras = self.face_cascade.detectMultiScale(
                gris, 
                scaleFactor=1.15, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Dibujar panel superior global de estadísticas
            cv2.rectangle(frame_original, (0, 0), (w_orig, 40), (20, 20, 20), -1)
            cv2.line(frame_original, (0, 40), (w_orig, 40), (0, 255, 0), 1)
            
            total_usuarios = len(self.nombres_usuarios)
            texto_status = f"ESTADO: LISTO | FPS: {fps:.1f} | USUARIOS EN IA: {total_usuarios}"
            cv2.putText(frame_original, texto_status, (15, 26), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Si no hay usuarios en la IA, indicar cómo registrar
            if total_usuarios == 0:
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 0, 150), -1)
                cv2.rectangle(frame_original, (80, 200), (w_orig - 80, 280), (0, 165, 255), 2)
                cv2.putText(frame_original, "NO HAY USUARIOS REGISTRADOS", (110, 230),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_original, "Presione 'R' en la terminal para registrar el primero", (100, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Procesar cada cara detectada
            for (x_peq, y_peq, w_peq, h_peq) in caras:
                # Reescalar coordenadas a tamaño original
                x = x_peq * escala
                y = y_peq * escala
                w = w_peq * escala
                h = h_peq * escala
                
                etiqueta = "Analizando..."
                subtitulo = "Comparando firmas..."
                color = (0, 255, 255)  # Amarillo/Naranja de análisis
                
                if self.modelo_cargado:
                    # Recortar rostro de la versión gris pequeña
                    cara_gris_recortada = gris[y_peq:y_peq+h_peq, x_peq:x_peq+w_peq]
                    cara_gris_recortada = cv2.resize(cara_gris_recortada, (WIDTH_RESIZE, HEIGHT_RESIZE))
                    
                    try:
                        id_prediccion, distancia = self.recognizer.predict(cara_gris_recortada)
                        
                        # El clasificador LBPH devuelve distancia.
                        # Valores de distancia más bajos indican mayor similitud (típicamente < 70-80 es buena predicción)
                        # Convertimos la distancia a un porcentaje de confianza aproximado
                        confianza_pct = max(0, 100 - distancia)
                        
                        if confianza_pct > 40:  # Umbral de confianza aceptable
                            nombre_detectado = self.nombres_usuarios.get(id_prediccion, "Desconocido")
                            self.memoria_deteccion.append(nombre_detectado)
                        else:
                            self.memoria_deteccion.append("Desconocido")
                            
                        # Votación por mayoría en memoria temporal para evitar oscilaciones
                        if len(self.memoria_deteccion) > 0:
                            voto_ganador = max(set(self.memoria_deteccion), key=self.memoria_deteccion.count)
                        else:
                            voto_ganador = "Desconocido"
                            
                        if voto_ganador != "Desconocido":
                            etiqueta = f"ACTIVO: {voto_ganador.upper()}"
                            subtitulo = f"Match: {confianza_pct:.1f}%"
                            color = (0, 255, 0)  # Verde
                            
                            # Saludo por voz en segundo plano y guardado en CSV
                            encolar_saludo(voto_ganador)
                            self.registrar_en_csv(voto_ganador, confianza_pct)
                        else:
                            etiqueta = "DESCONOCIDO"
                            subtitulo = "Registrese pulsando 'R'"
                            color = (0, 0, 255)  # Rojo
                            encolar_saludo("Desconocido")
                    except Exception as e:
                        subtitulo = f"Error IA: {e}"
                        color = (0, 0, 255)
                else:
                    subtitulo = "IA sin clasificador.yml"
                    color = (0, 165, 255)
                
                # Dibujar HUD interactivo
                self.dibujar_hud_futurista(frame_original, x, y, w, h, etiqueta, subtitulo, color)
            
            # Mostrar frame
            cv2.imshow("Antigravity Smart Recognition HUD", frame_original)
            
            # Medir FPS
            t_fin = time.time()
            fps = 1.0 / (t_fin - t_inicio)
            
            # Capturar entrada por teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[SISTEMA] Cerrando aplicación y liberando recursos...")
                break
            elif key == ord('r') or key == ord('R'):
                # Llamar al método de registro de usuario
                self.registrar_nuevo_usuario(frame_original, cap)
                # Al volver, recargar el modelo por si se entrenó
                self.cargar_modelo()
                
        cap.release()
        cv2.destroyAllWindows()

# =====================================================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA DE ASISTENCIA Y RECONOCIMIENTO FACIAL CON IA")
    print("=" * 60)
    
    try:
        app = SistemaReconocimientoFacial()
        app.iniciar_bucle_principal()
    except Exception as e:
        print(f"\n[CRÍTICO] Error al inicializar la aplicación: {e}")
        input("Presione ENTER para salir...")
