Sistema de Monitoreo de Fatiga y Somnolencia
Descripción del proyecto
Este proyecto consiste en el desarrollo de una aplicación de visión artificial orientada al monitoreo de signos asociados a la fatiga y la somnolencia de una persona mediante el análisis de imágenes capturadas en tiempo real desde una cámara web mientras conduce un vehículo.
La aplicación utiliza un modelo de detección facial de MediaPipe para localizar distintos puntos de referencia del rostro. A partir de esos puntos, se calculan métricas relacionadas con:
●	La apertura de los ojos.
●	La apertura de la boca.
●	La inclinación vertical de la cabeza.
●	La permanencia de estas condiciones durante varios fotogramas consecutivos.
Con estas mediciones, el sistema construye un índice compuesto y muestra una alerta cuando detecta condiciones que podrían estar relacionadas con fatiga o somnolencia.
________________________________________
Integrantes del equipo
●	Sanchez, Carlos Gabriel
●	Sanchez, Lucia Florencia
________________________________________
Tecnologías utilizadas
El proyecto utiliza las siguientes tecnologías:
●	Python: lenguaje principal del desarrollo.
●	MediaPipe: detección del rostro y obtención de landmarks faciales.
●	OpenCV: dibujo de líneas, puntos, marcos y textos sobre las imágenes.
●	NumPy: operaciones matemáticas y cálculo de distancias.
●	Gradio: interfaz gráfica web y acceso a la cámara.
●	Hugging Face Spaces: plataforma utilizada para desplegar y publicar la aplicación en línea. 
●	urllib: descarga automática del modelo de MediaPipe.
●	Visual Studio Code: entorno de desarrollo.
●	Git y GitHub: control de versiones y almacenamiento del proyecto.
________________________________________
Funcionamiento general
El flujo de procesamiento de la aplicación es el siguiente:
1.	Se obtiene un fotograma desde la cámara web.
2.	El fotograma se convierte al formato utilizado por MediaPipe.
3.	MediaPipe detecta el rostro y sus puntos de referencia.
4.	Se seleccionan puntos específicos de los ojos, la boca, la frente, la nariz y el mentón.
5.	Se calculan las métricas EAR, MAR y la estimación de inclinación de la cabeza.
6.	Las métricas se comparan con umbrales configurables.
7.	Se genera un índice compuesto.
8.	Si el índice supera el valor establecido durante varios fotogramas consecutivos, se muestra una alerta visual.
9.	La imagen procesada se presenta en la interfaz de Gradio.
________________________________________
Métricas utilizadas
EAR — Eye Aspect Ratio
El EAR permite estimar cuánto está abierto un ojo comparando su altura con su ancho.
La fórmula simplificada utilizada es:
EAR = distancia vertical del ojo / distancia horizontal del ojo
Cuando el ojo se cierra, la distancia vertical disminuye y, por lo tanto, el valor EAR se reduce.
El sistema calcula el EAR de ambos ojos y obtiene un promedio.
________________________________________
MAR — Mouth Aspect Ratio
El MAR permite estimar la apertura de la boca.
La fórmula simplificada es:
MAR = altura de la boca / ancho de la boca
Un valor elevado puede indicar que la boca está abierta. Dentro del contexto del proyecto, esta métrica se utiliza como un posible indicador de bostezo.
________________________________________
Inclinación vertical de la cabeza
La inclinación de la cabeza se estima utilizando los puntos correspondientes a:
●	Frente.
●	Nariz.
●	Mentón.
El sistema compara las distancias entre estos puntos para obtener un valor aproximado de inclinación vertical o pitch.
Este cálculo es una aproximación y no representa una medición tridimensional exacta de la postura de la cabeza.
________________________________________
Índice compuesto
El sistema construye un índice compuesto utilizando tres condiciones:
●	Ojos por debajo del umbral EAR: 50 puntos.
●	Boca por encima del umbral MAR: 20 puntos.
●	Inclinación de cabeza por encima del umbral: 30 puntos.
El puntaje máximo es de 100 puntos.
Cuando el índice compuesto alcanza o supera los 50 puntos durante una cantidad determinada de fotogramas consecutivos, el sistema muestra la alerta:
ALERTA: FATIGA
Los valores y umbrales pueden modificarse desde la interfaz gráfica.
________________________________________
Interfaz gráfica
La interfaz fue desarrollada con Gradio.
Permite:
●	Activar la cámara web.
●	Visualizar el fotograma original.
●	Visualizar el fotograma procesado.
●	Observar los ejes calculados sobre los ojos.
●	Observar los puntos seleccionados de la boca.
●	Visualizar el eje vertical de la cabeza.
●	Consultar los valores EAR, MAR, pitch e índice compuesto.
●	Modificar los umbrales mediante controles deslizantes.

Los controles disponibles son:
●	Confianza mínima de detección facial.
●	Umbral EAR.
●	Umbral MAR.
●	Umbral de inclinación de cabeza.
●	Cantidad de fotogramas consecutivos necesarios para activar la alerta.
________________________________________
Estructura del proyecto
Detector de Fatiga/
├── app.py
├── app_2.py
├── face_landmarker.task
└── README.md
Archivos principales
●	app.py: primera versión del detector, basada principalmente en el análisis de los ojos.
●	app_2.py: versión que incorpora ojos, boca, inclinación de cabeza e índice compuesto.
●	face_landmarker.task: modelo utilizado por MediaPipe para detectar landmarks faciales.
●	README.md: documentación general del trabajo.
________________________________________
Requisitos
Las principales dependencias son:
opencv-python
numpy
gradio
mediapipe
python-dateutil

________________________________________
Modelo de MediaPipe
La aplicación utiliza el modelo: Face Landmarker
Si el archivo face_landmarker.task no se encuentra dentro del proyecto, el código se descarga automáticamente desde el repositorio de modelos de MediaPipe.
