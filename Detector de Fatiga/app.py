import cv2
import numpy as np
import gradio as gr
import mediapipe as mp
import urllib.request
import os

# ==========================================
# 1. DESCARGA LOCAL DEL MODELO (Obligatorio en HF)
# ==========================================
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Si el archivo no existe en el servidor, lo descargamos en segundos
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Descarga completa.")

# ==========================================
# 2. CONFIGURACIÓN MODERNA (MEDIAPIPE TASKS)
# ==========================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Índices para los ojos
INDICES_OJO_IZQ = [33, 160, 158, 133, 153, 144]
INDICES_OJO_DER = [362, 385, 387, 263, 373, 380]

# Lista simplificada fija de los labios (boca)
INDICES_BOCA = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88, 178, 87, 14, 317, 402]

EAR_THRESHOLD = 0.22          
CONSECUTIVE_FRAMES = 6        
contador_frames_cerrados = 0

# Configurar apuntando de forma LOCAL al archivo descargado
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# Inicializar el detector globalmente
detector = FaceLandmarker.create_from_options(options)

# ==========================================
# 3. FUNCIÓN MATEMÁTICA (EAR)
# ==========================================
def calcular_ear(landmarks_ojo, width, height):
    puntos = []
    for idx in landmarks_ojo:
        lm = landmarks_ojo[idx]
        puntos.append(np.array([int(lm.x * width), int(lm.y * height)]))
    
    d_vertical_1 = np.linalg.norm(puntos[1] - puntos[5])
    d_vertical_2 = np.linalg.norm(puntos[2] - puntos[4])
    d_horizontal = np.linalg.norm(puntos[0] - puntos[3])
    
    return (d_vertical_1 + d_vertical_2) / (2.0 * d_horizontal)

# ==========================================
# 4. PIPELINE DE PROCESAMIENTO
# ==========================================
def pipeline_detector_fatiga(frame):
    global contador_frames_cerrados
    
    if frame is None:
        return None

    img_out = frame.copy()
    h, w, _ = img_out.shape

    # Convertir el frame de Gradio a Imagen nativa de MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_out)
    
    # Procesar
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        rostro = detection_result.face_landmarks[0]
        
        # 1. Dibujar puntos de los ojos (Azul)
        puntos_ojo_izq = {idx: rostro[idx] for idx in INDICES_OJO_IZQ}
        puntos_ojo_der = {idx: rostro[idx] for idx in INDICES_OJO_DER}
        
        for idx in INDICES_OJO_IZQ + INDICES_OJO_DER:
            lm = rostro[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img_out, (cx, cy), 2, (255, 0, 0), -1)

        # 2. Dibujar puntos de la boca (Verde)
        for idx in INDICES_BOCA:
            lm = rostro[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img_out, (cx, cy), 1, (0, 255, 0), -1)

        # 3. Métrica de Somnolencia (EAR)
        ear_izq = calcular_ear(puntos_ojo_izq, w, h)
        ear_der = calcular_ear(puntos_ojo_der, w, h)
        promedio_ear = (ear_izq + ear_der) / 2.0
        
        cv2.putText(img_out, f"EAR: {promedio_ear:.2f}", (30, h - 40), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)
        
        if promedio_ear < EAR_THRESHOLD:
            contador_frames_cerrados += 1
        else:
            contador_frames_cerrados = 0
        
        # Alerta Crítica (Marco Rojo)
        if contador_frames_cerrados >= CONSECUTIVE_FRAMES:
            cv2.rectangle(img_out, (0, 0), (w, h), (255, 0, 0), 20)
            cv2.putText(img_out, "ALERTA: CONDUCTOR DORMIDO", (int(w*0.1), int(h*0.5)), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0), 3)
    else:
        cv2.putText(img_out, "SISTEMA BUSCANDO OPERADOR...", (30, h - 40), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)
        contador_frames_cerrados = 0

    return img_out

# ==========================================
# 5. INTERFAZ GRÁFICA (GRADIO)
# ==========================================
interface = gr.Interface(
    fn=pipeline_detector_fatiga,
    inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
    title="Monitoreo de Fatiga y Somnolencia en Tiempo Real",
    description="Aplicación desplegada con la API de Tareas de MediaPipe y almacenamiento local."
)

if __name__ == "__main__":
    interface.launch()
