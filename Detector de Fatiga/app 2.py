import cv2
import numpy as np
import gradio as gr
import mediapipe as mp
import urllib.request
import os

# ==========================================
# 1. DESCARGA LOCAL DEL MODELO
# ==========================================
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ==========================================
# 2. CONFIGURACIÓN DE MEDIAPIPE TASKS
# ==========================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Índices detallados para mediciones de ejes y geometría
INDICES_OJO_IZQ = [33, 160, 158, 133, 153, 144]
INDICES_OJO_DER = [362, 385, 387, 263, 373, 380]
# Puntos extremos para los ejes de los ojos [Izquierda, Derecha, Superior, Inferior]
EJES_OJO_IZQ = [33, 133, 159, 145]
EJES_OJO_DER = [362, 263, 386, 374]

# Puntos para el cálculo del MAR (Labios exteriores)
# Izquierda=61, Derecha=291, Superiores=37,72, Inferiores=84,314
INDICES_BOCA = [61, 291, 37, 72, 84, 314]

# Puntos del eje vertical de la cabeza
IDX_FRENTE = 10
IDX_NARIZ = 4
IDX_MENTON = 152

contador_frames_cerrados = 0

# ==========================================
# 3. FUNCIONES MATEMÁTICAS (MÉTRICAS)
# ==========================================
def calcular_ear(rostro, indices_ejes, w, h):
    p_izq = np.array([int(rostro[indices_ejes[0]].x * w), int(rostro[indices_ejes[0]].y * h)])
    p_der = np.array([int(rostro[indices_ejes[1]].x * w), int(rostro[indices_ejes[1]].y * h)])
    p_sup = np.array([int(rostro[indices_ejes[2]].x * w), int(rostro[indices_ejes[2]].y * h)])
    p_inf = np.array([int(rostro[indices_ejes[3]].x * w), int(rostro[indices_ejes[3]].y * h)])
    
    alto = np.linalg.norm(p_sup - p_inf)
    ancho = np.linalg.norm(p_izq - p_der)
    return alto / (ancho + 1e-6), p_izq, p_der, p_sup, p_inf

def calcular_mar(rostro, w, h):
    p_izq = np.array([int(rostro[61].x * w), int(rostro[61].y * h)])
    p_der = np.array([int(rostro[291].x * w), int(rostro[291].y * h)])
    p_sup = np.array([int(rostro[37].x * w), int(rostro[37].y * h)])
    p_inf = np.array([int(rostro[84].x * w), int(rostro[84].y * h)])
    
    alto = np.linalg.norm(p_sup - p_inf)
    ancho = np.linalg.norm(p_izq - p_der)
    return alto / (ancho + 1e-6)

def calcular_pitch_cabeza(rostro):
    frente = np.array([rostro[IDX_FRENTE].x, rostro[IDX_FRENTE].y, rostro[IDX_FRENTE].z])
    nariz = np.array([rostro[IDX_NARIZ].x, rostro[IDX_NARIZ].y, rostro[IDX_NARIZ].z])
    menton = np.array([rostro[IDX_MENTON].x, rostro[IDX_MENTON].y, rostro[IDX_MENTON].z])
    
    dist_superior = np.linalg.norm(frente - nariz)
    dist_inferior = np.linalg.norm(nariz - menton)
    relacion = dist_superior / (dist_inferior + 1e-6)
    return (relacion - 1.0) * 90.0

# ==========================================
# 4. PIPELINE DINÁMICO (CON SLIDERS)
# ==========================================
def pipeline_analisis(frame, conf_deteccion, umbral_ear, umbral_mar, umbral_pitch, frames_alerta):
    global contador_frames_cerrados
    if frame is None:
        return None

    img_out = frame.copy()
    h, w, _ = img_out.shape
    
    # Inicializar dinámicamente según el slider de confianza del usuario
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        min_face_detection_confidence=conf_deteccion,
        num_faces=1
    )
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    color_azul = (0, 0, 255) # RGB 
    color_rojo = (255,0,0)
    
    with FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_out)
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            rostro = detection_result.face_landmarks[0]
            
            # --- 1. CÁLCULO Y DIBUJO DE EJES DE LOS OJOS ---
            ear_i, *pts_i = calcular_ear(rostro, EJES_OJO_IZQ, w, h)
            ear_d, *pts_d = calcular_ear(rostro, EJES_OJO_DER, w, h)
            promedio_ear = (ear_i + ear_d) / 2.0
            
            # Dibujar ejes en cruz (Ancho y Alto) para Ojo Izquierdo
            cv2.line(img_out, tuple(pts_i[0]), tuple(pts_i[1]), color_azul, 1)
            cv2.line(img_out, tuple(pts_i[2]), tuple(pts_i[3]), color_azul, 1)
            # Dibujar ejes en cruz para Ojo Derecho
            cv2.line(img_out, tuple(pts_d[0]), tuple(pts_d[1]), color_azul, 1)
            cv2.line(img_out, tuple(pts_d[2]), tuple(pts_d[3]), color_azul, 1)
            
            # --- 2. DIBUJO DEL EJE VERTICAL DE LA CABEZA ---
            p_frente = (int(rostro[IDX_FRENTE].x * w), int(rostro[IDX_FRENTE].y * h))
            p_menton = (int(rostro[IDX_MENTON].x * w), int(rostro[IDX_MENTON].y * h))
            cv2.line(img_out, p_frente, p_menton, color_azul, 2)
            
            # --- 3. MÉTRICAS RESTANTES ---
            mar_boca = calcular_mar(rostro, w, h)
            pitch_cabeza = calcular_pitch_cabeza(rostro)
            
            # Lógica del Índice Compuesto Dinámico
            score_compuesto = 0
            if promedio_ear < umbral_ear:
                score_compuesto += 50
            if mar_boca > umbral_mar:
                score_compuesto += 20
            if pitch_cabeza > umbral_pitch:
                score_compuesto += 30
            
            # --- 4. PANEL METRICO INFERIOR IZQUIERDO ---
            cv2.putText(img_out, f"EAR (Ojos): {promedio_ear:.2f}", (30, h - 140), font, 0.6, color_azul, 2)
            cv2.putText(img_out, f"MAR (Boca): {mar_boca:.2f}", (30, h - 105), font, 0.6, color_azul, 2)
            cv2.putText(img_out, f"Cabeza (Pitch): {pitch_cabeza:.1f} Grad", (30, h - 70), font, 0.6, color_azul, 2)
            cv2.putText(img_out, f"INDICE COMPUESTO: {score_compuesto}%", (30, h - 30), font, 0.7, color_azul, 2)
            
            # Sistema de Alertas
            if score_compuesto >= 50:
                contador_frames_cerrados += 1
            else:
                contador_frames_cerrados = 0
            
            if contador_frames_cerrados >= frames_alerta:
                cv2.rectangle(img_out, (0, 0), (w, h), color_rojo, 20)
                cv2.putText(img_out, "ALERTA: FATIGA DETECTADA", (int(w*0.15), int(h*0.5)), font, 1.1, color_rojo, 3)
        else:
            cv2.putText(img_out, "BUSCANDO OPERADOR...", (30, h - 30), font, 0.7, color_azul, 2)
            contador_frames_cerrados = 0

    return img_out

# ==========================================
# 5. DISEÑO DE LA INTERFAZ CON SLIDERS (GRADIO)
# ==========================================
with gr.Blocks(title="Laboratorio Avanzado de PDI - Detección de Fatiga") as interfaz_avanzada:
    gr.Markdown("# Sistema Analítico de Monitoreo de Fatiga Multi-Variable")
    gr.Markdown("Calibra los umbrales algebraicos en tiempo real utilizando el panel lateral.")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Entrada de la Cámara Web y Salida de Video
            input_image = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Cámara en Vivo")
            output_image = gr.Image(type="numpy", label="Análisis Cinemático (Ejes)")
            
        with gr.Column(scale=1):
            gr.Markdown("### Panel de Control de Umbrales")
            
            slider_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Confianza Detección Rostro")
            slider_ear = gr.Slider(minimum=0.15, maximum=0.35, value=0.22, step=0.01, label="Umbral Crítico EAR (Ojos)")
            slider_mar = gr.Slider(minimum=0.10, maximum=0.80, value=0.45, step=0.05, label="Umbral Bostezo MAR (Boca)")
            slider_pitch = gr.Slider(minimum=5.0, maximum=30.0, value=15.0, step=1.0, label="Umbral Inclinación Cabeza (Pitch)")
            slider_frames = gr.Slider(minimum=2, maximum=20, value=6, step=1, label="Cuadros Consecutivos (Tiempo)")

    # Vincular los sliders dinámicos directamente al flujo de procesamiento en vivo
    input_image.stream(
        fn=pipeline_analisis,
        inputs=[input_image, slider_conf, slider_ear, slider_mar, slider_pitch, slider_frames],
        outputs=output_image
    )

if __name__ == "__main__":
    interfaz_avanzada.launch()

