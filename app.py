import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Inicializar MediaPipe Holistic y dibujo
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Cargar el modelo entrenado
model = load_model("modelo.keras")

# Nombres de las clases
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


def process_single_frame(frame, holistic):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    keypoints = extract_keypoints(results)
    return keypoints, results


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
    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
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
    return np.concatenate([pose, face, lh, rh])


def predict_sign_from_sequence(sequence, model):
    sequence = np.expand_dims(sequence, axis=0)
    return model.predict(sequence)


def visualize_frame(frame, results, prediction, class_names, confidence_threshold=0.7):
    if prediction is not None:
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]

        if confidence >= confidence_threshold:
            text = f"{class_names[predicted_class]}: {confidence:.2f}"
            color = (0, 255, 0)
        else:
            text = "No se reconoce"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Dibujar landmarks
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )

    return frame


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Configurar MediaPipe Holistic
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as holistic:
    sequence = []  # Secuencia de keypoints
    sequence_length = 30  # Longitud fija de la secuencia

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            break

        keypoints, results = process_single_frame(frame, holistic)
        sequence.append(keypoints)

        if len(sequence) > sequence_length:
            sequence = sequence[
                -sequence_length:
            ]  # Mantener solo los últimos 30 elementos

        prediction = None
        if len(sequence) == sequence_length:
            sequence_array = np.array(sequence)
            prediction = predict_sign_from_sequence(sequence_array, model)

        frame = visualize_frame(frame, results, prediction, class_names)

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
