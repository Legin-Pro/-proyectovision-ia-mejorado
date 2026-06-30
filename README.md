# Sistema Inteligente de Reconocimiento Facial y Registro de Asistencia (S.I.R.F.R.A.)

![Python](https://img.shields.io/badge/Python-3.14%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-contrib--python-green?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Optimized-success?style=for-the-badge&logo=lightning&logoColor=white)
![Deploy](https://img.shields.io/badge/Deploy-PyInstaller-blueviolet?style=for-the-badge&logo=pythoncore&logoColor=white)

Un sistema avanzado de visión artificial diseñado para la detección, reconocimiento y registro en tiempo real de personas mediante técnicas de aprendizaje local con costo cero.

## Características del Software
* **Detección de Rostros en Tiempo Real:** Implementación optimizada mediante clasificadores en cascada de Haar (Haar Cascades).
* **Reconocimiento Facial de Costo Cero:** Utiliza el algoritmo local LBPH (Local Binary Patterns Histograms) para clasificar rostros sin necesidad de APIs pagas ni servicios en la nube.
* **Interfaz HUD (Heads-Up Display) Futurista:** Dibujo interactivo sobre el video de brackets angulares, indicador de estado dinámico y línea de escaneo dinámico.
* **Saludos por Voz (Text-to-Speech):** Síntesis de voz asíncrona mediante hilos para saludar a los usuarios por su nombre sin congelar el video.
* **Base de Datos de Asistencia Local:** Exportación y registro automático de entradas al sistema en un archivo `asistencia.csv` con control de duplicados (cooldown temporal).
* **Registro Express de Usuarios:** Menú integrado que detiene el video, solicita el nombre y captura 30 imágenes de entrenamiento en segundos para actualizar el clasificador al instante.

## Requisitos de Infraestructura
* **Sistema Operativo:** Windows, macOS o Linux.
* **Hardware:** Cámara web integrada o USB, y altavoces/salida de audio.
* **Procesador:** CPU estándar (gracias a la optimización, corre fluidamente a más de 30 FPS sin necesidad de GPU dedicada).
* **Python:** Versión 3.10 o superior.

## Guía de Instalación de Dependencias
Sigue estos pasos para preparar el entorno de ejecución:

1. Clona o copia los archivos de este proyecto en una carpeta local.
2. Abre una terminal de comandos en el directorio del proyecto.
3. Ejecuta el comando para instalar las librerías necesarias mediante `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Instrucciones de Uso
1. **Ejecutar el programa:**
   * Mediante Python: Ejecuta `python reconocedor.py` en la terminal.
   * Mediante el ejecutable compilado: Ejecuta el archivo autoportante `dist/reconocedor.exe`.
2. **Reconocimiento:** El sistema abrirá la cámara y comenzará a buscar rostros en la base de datos local.
3. **Registrar una nueva persona:**
   * Presiona la tecla **`R`** (en mayúscula o minúscula) en la ventana de OpenCV.
   * En la consola de comandos, ingresa el nombre de la persona cuando se te solicite.
   * Vuelve a mirar a la cámara. El sistema capturará 30 fotos de tu rostro y entrenará automáticamente al clasificador.
4. **Salir:** Presiona la tecla **`Q`** para cerrar la aplicación y liberar la cámara web de forma segura.

## Créditos del Equipo de Desarrollo
* Desarrollado por el Equipo de Especialistas en Inteligencia Artificial en colaboración con **Antigravity (AI Coding Assistant)**.
