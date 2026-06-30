# INFORME TÉCNICO: OPTIMIZACIÓN, DESPLIEGUE Y DOCUMENTACIÓN DE PROYECTO DE IA
**Curso Avanzado de Inteligencia Artificial y Visión por Computadora**

---

## 1. Herramienta de IA Seleccionada y Justificación
* **Agente de IA Utilizado:** Antigravity (asistente autónomo de desarrollo de Google DeepMind) operando con el modelo **Gemini 3.5 Flash**.
* **Justificación de la Elección:** 
  Antigravity fue elegido por su capacidad para integrarse directamente con el sistema operativo de desarrollo a nivel local. A diferencia de un chat convencional, este agente puede:
  1. Analizar de forma autónoma la estructura del directorio local.
  2. Escribir y modificar archivos directamente, respetando las buenas prácticas de programación en Python.
  3. Ejecutar de forma segura comandos de consola como instalaciones de pip, inicializaciones de Git y compilaciones pesadas con PyInstaller.
  4. Mantener un flujo interactivo de toma de decisiones (como el modo de auditoría `/grill-me`).
  La combinación de estas facultades optimiza drásticamente los tiempos de desarrollo y despliegue del software de visión artificial.

---

## 2. Bitácora de Ingeniería de Prompts (Meta-Prompting)

A continuación se presentan dos ejemplos prácticos de la aplicación de la técnica de Meta-Prompting para guiar al agente de desarrollo:

### Ejemplo 1: Solicitud de Auditoría Inicial (Fase 2.1)
* **Instrucción Informal (Equipo):**
  > *"revisa mis archivos de cv2 en la terminal y hazme unas preguntas de si la camara falla o va lento"*
* **Prompt Optimizado por IA Conversacional (Ingeniero de Prompts Senior):**
  > *"Estoy trabajando en un proyecto académico de visión por computadora en Python que se encarga de procesar imágenes. Necesito que mi agente de desarrollo local analice mis archivos de código en modo de auditoría ('grill-me') sin realizar modificaciones todavía. Actúa como un Ingeniero de Prompts Senior y redacta una instrucción estructurada, clara y profesional para que se la envíe a mi agente local. El prompt generado debe ordenarle al agente que me plantee entre 4 y 5 preguntas técnicas profundas sobre la fluidez de la imagen, el manejo de excepciones en caso de que la cámara falle y la organización general del código."*

### Ejemplo 2: Solicitud de Refactorización y Refinamiento (Fase 2.2)
* **Instrucción Informal (Equipo):**
  > *"arregla el codigo de la camara para que ande rapido, salude a la gente diciendo su nombre y no se rompa si desconecto la camara"*
* **Prompt Optimizado por IA Conversacional (Ingeniero de Prompts Senior):**
  > *"He concluido el análisis inicial con mi agente de desarrollo local sobre mi código de visión por computadora en Python. Hemos definido que el objetivo principal para mejorar el proyecto es implementar un reconocedor facial en tiempo real local con Haar Cascades y LBPH que salude con voz a personas conocidas, registre nuevas personas en caliente y guarde logs en un archivo CSV. Actúa como un Ingeniero de Prompts Senior y redacta un prompt profesional y detallado para mi agente de código local. La instrucción debe indicarle que modifique mis archivos de Python de forma automática, aplicando buenas prácticas de programación, garantizando la compatibilidad con las dependencias instaladas y verificando que no existan errores de sintaxis antes de guardar el archivo."*

---

## 3. Bitácora de Diagnóstico (Sesión Grill-Me)

Durante la fase de interrogatorio, el agente local planteó las siguientes preguntas clave de auditoría, las cuales fueron resueltas por el equipo para el diseño final:

1. **¿Cómo optimizar el rendimiento y la fluidez del video en tiempo real?**
   * *Respuesta del Equipo:* Reduciremos cada frame de la cámara a un 25% de su resolución antes de aplicar Haar Cascades y reconocimiento, reescalando luego las coordenadas para pintar el HUD futurista en el frame de resolución original.
2. **¿Qué hacer ante desconexiones físicas de la cámara?**
   * *Respuesta del Equipo:* Utilizaremos un control condicional en el bucle principal (`cap.isOpened()` y validación del retorno de `cap.read()`). En caso de fallo continuo, liberaremos recursos y cerraremos el programa con un mensaje amigable en consola.
3. **¿Cómo estructurar el código para que sea limpio y portátil?**
   * *Respuesta del Equipo:* Agruparemos toda la lógica dentro de una estructura Orientada a Objetos mediante la clase `SistemaReconocimientoFacial`, modularizando el entrenamiento, el registro en CSV y la captura.
4. **¿Cómo implementar el saludo de voz sin degradar los FPS?**
   * *Respuesta del Equipo:* Crearemos un hilo secundario independiente (multithreading) que ejecute `pyttsx3` de forma asíncrona. De esta manera, el hilo principal de renderizado de OpenCV no sufre bloqueos de latencia de audio.
5. **¿Qué datos registrar en la persistencia del sistema?**
   * *Respuesta del Equipo:* Guardaremos en un archivo `asistencia.csv` el nombre de la persona, la fecha, hora exacta y el porcentaje de coincidencia (confianza) de la predicción, limitándolo con un temporizador para no saturar el archivo de duplicados del mismo usuario.

---

## 4. Especificación del Objetivo (Goal)
El objetivo del proyecto fue desarrollar un **Sistema de Reconocimiento Facial y Control de Asistencia Inteligente Autoportante** en tiempo real. 
El software debía cumplir con las siguientes metas:
* Operar a más de 30 FPS en computadoras de consumo común.
* Utilizar tecnologías gratuitas, open-source y 100% locales (OpenCV LBPH y pyttsx3).
* Permitir el registro "caliente" (en caliente) de rostros mediante entrada por terminal sin necesidad de reiniciar la aplicación.
* Mostrar una interfaz gráfica interactiva HUD (brackets angulares, indicador de FPS, barra de estado superior).
* Compilarse a un archivo ejecutable portable (`dist/reconocedor.exe`).

---

## 5. Análisis de Impacto de Shields.io (La Comparativa)
La comparación estética y conceptual del archivo `README.md` entre la Fase A (básica) y la Fase B (con insignias Shields.io) demuestra el valor de la presentación visual en la ingeniería de software actual:

* **README Básico (Fase A):** Cumple su función informativa, pero se percibe plano, requiere lectura atenta y carece de dinamismo. Visualmente no destaca, pareciendo un borrador interno.
* **README con Shields.io (Fase B):** La inclusión de insignias de alta calidad con logotipos oficiales (Python, OpenCV, PyInstaller y estado de optimización) aporta un aspecto profesional de inmediato. Estas insignias actúan como metadatos visuales rápidos que permiten a cualquier reclutador o programador externo conocer la versión del lenguaje, librerías críticas y estado del proyecto en menos de 2 segundos.

En la industria de desarrollo de software moderna, la estética visual en los repositorios de GitHub es clave para generar confianza, denotar profesionalismo técnica, y facilitar la legibilidad del stack tecnológico del proyecto.

---

## 6. Matriz de Impacto en el Software (Antes y Después)

A continuación, se detalla el cambio estructural en el código original frente al optimizado bajo los criterios de la rúbrica de evaluación:

| Componente del Software | Estado Estructural Inicial (Código Original) | Estado Estructural Final (Optimizado por el Agente) | Evidencia Práctica de la Mejora |
| :--- | :--- | :--- | :--- |
| **Rendimiento del Video** | Scripts secuenciales que procesaban imágenes estáticas con `cv2.imread()` o bucles directos sin reescalado de procesamiento. | Bucle con redimensionamiento dinámico (25% del frame) para la detección por Haar Cascade, manteniendo el dibujado del HUD en la resolución nativa. | Flujo de video suave y continuo a más de 30 FPS en tiempo real, sin experimentar pausas ni caídas notables de fotogramas. |
| **Control de Excepciones** | El código no tenía controles de error. Si fallaba el archivo de imagen o no había cámara conectada, arrojaba excepciones y colapsaba. | Bloques `try-except` envolviendo la carga del modelo, la inicialización del motor de voz (`pyttsx3`) y la comprobación preventiva de `cap.isOpened()`. | Al desconectar la cámara o ejecutar sin clasificador inicial entrenado, el sistema avisa con diálogos claros en consola y en pantalla en lugar de crashear. |
| **Documentación y Entrega** | Repositorio sin archivos de configuración, manual de instalación de dependencias, ni estructura de control de versiones. | Archivo `README.md` completo con badges visuales de Shields.io, `requirements.txt` con versiones fijadas y `.gitignore` profesional. | Cualquier desarrollador puede clonar el repositorio, entender las dependencias visuales de inmediato e instalar el entorno con un solo comando de consola. |
| **Despliegue y Distribución** | Requería configurar dependencias manualmente de OpenCV, instalar intérpretes de Python de forma local y ejecutar desde terminal. | Entorno compilado autónomamente con PyInstaller en un único archivo binario ejecutable (`dist/reconocedor.exe`). | La aplicación final se ejecuta de forma portable como un archivo independiente de producción, sin necesidad de tener Python o librerías preinstaladas. |
