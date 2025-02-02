import base64
import logging
import time
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.models import load_model

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MediaPipe configuración
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Modelo preentrenado
try:
    model = load_model("modelo.keras")
    logging.info("Modelo cargado exitosamente.")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    raise e

# Clases
class_names = [
    "again",
    "bad",
    "bathroom",
    "book",
    "busy",
    "do not want",
    "eat",
    "father",
    "fine",
    "finish",
    "forget",
    "go",
    "good",
    "happy",
    "hello",
    "help",
    "how",
    "i",
    "learn",
    "like",
    "meet",
    "milk",
    "more",
    "mother",
    "my",
    "name",
    "need",
    "nice",
    "no",
    "please",
    "question",
    "right",
    "sad",
    "same",
    "see you letter",
    "thank you",
    "want",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "wrong",
    "yes",
    "you",
    "your",
]


# Extraer keypoints
def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])


# Predicción desde una secuencia
def predict_sign_from_sequence(sequence, model):
    sequence = np.expand_dims(sequence, axis=0)
    return model.predict(sequence)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        logging.info("Cliente conectado")

        sequence = []
        sequence_length = 30
        last_prediction_time = 0
        prediction_interval = 0.5

        # Configuración de MediaPipe
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while True:
                try:
                    # Recibir frame en formato base64
                    message = await websocket.receive_text()
                    if not message.startswith("data:image/"):
                        logging.warning("Formato de mensaje inválido")
                        continue

                    image_data = base64.b64decode(message.split(",")[1])
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # Extraer keypoints
                    keypoints, results = extract_keypoints_from_frame(frame, holistic)
                    sequence.append(keypoints)

                    # Limitar tamaño de la secuencia
                    if len(sequence) > sequence_length:
                        sequence = sequence[-sequence_length:]

                    # Predicción a intervalos controlados
                    prediction = None
                    if len(sequence) == sequence_length:
                        current_time = time.time()
                        if current_time - last_prediction_time > prediction_interval:
                            sequence_array = np.array(sequence)
                            prediction = predict_sign_from_sequence(
                                sequence_array, model
                            )
                            last_prediction_time = current_time

                    # Enviar predicción
                    if prediction is not None:
                        predicted_class = class_names[np.argmax(prediction)]
                        confidence = float(np.max(prediction))
                        await websocket.send_json(
                            {"class": predicted_class, "confidence": confidence}
                        )
                    else:
                        await websocket.send_json({"class": None, "confidence": 0.0})
                except Exception as e:
                    logging.error(f"Error procesando el mensaje: {e}", exc_info=True)
                    break
    except WebSocketDisconnect:
        logging.info("Cliente desconectado")
    except Exception as e:
        logging.error(f"Error crítico en WebSocket: {e}", exc_info=True)


# Función para extraer keypoints desde un frame
def extract_keypoints_from_frame(frame, holistic):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    keypoints = extract_keypoints(results)
    return keypoints, results
