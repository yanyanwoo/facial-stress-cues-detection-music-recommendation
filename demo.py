import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model

# ==========================================
# 1. Load Mini-Xception Emotion Model
# ==========================================
MODEL_PATH = r"C:\Users\Sam Ocenar\Documents\Comp_vision\models\mini_xception.hdf5"
emotion_model = load_model(MODEL_PATH)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ==========================================
# 2. Mediapipe Face Mesh + Face Detection
# ==========================================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_face_det = mp.solutions.face_detection.FaceDetection(model_selection=0)

LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [300, 293, 334, 296, 336]

# Emotion smoothing buffer
emotion_window = deque(maxlen=8)

# ==========================================
# 3. Music Recommendation Engine
# ==========================================
def recommend_music(emotion):
    mapping = {
        "neutral": "Mellow Acoustic Playlist",
        "happy": "Chill Pop Mix",
        "sad": "Mood Boost Indie Playlist",
        "angry": "Calming Ambient Playlist",
        "fear": "Comforting Piano Playlist",
        "surprise": "Soft Instrumental Playlist",
        "disgust": "Relaxing Ambient Mix"
    }
    return mapping.get(emotion, "Relaxing Ambient Mix")

# ==========================================
# 4. Stress Score based on Eyebrow Motion
# ==========================================
def compute_stress(landmarks, w, h, prev_raise):
    left = (landmarks[70].y - landmarks[105].y) * h
    right = (landmarks[300].y - landmarks[334].y) * h
    avg_raise = (left + right) / 2

    # contraction
    contract = abs((landmarks[70].x - landmarks[300].x) * w)

    # motion
    motion = abs(avg_raise - prev_raise) if prev_raise is not None else 0

    # Weighted sum
    stress_raw = (abs(avg_raise) * 0.4) + (motion * 0.4) + (contract * 0.2)

    # Normalize 0–1 (adjust denominator to your camera)
    stress_norm = np.clip(stress_raw / 50, 0, 1)

    return stress_norm, avg_raise

# ==========================================
# 5. Start Webcam
# ==========================================
cap = cv2.VideoCapture(0)
prev_raise = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 1. Face detection for emotion
    # -----------------------------
    det = mp_face_det.process(rgb)
    current_emotion = "neutral"

    if det.detections:
        box = det.detections[0].location_data.relative_bounding_box

        x = int(box.xmin * w) - 20
        y = int(box.ymin * h) - 20
        bw = int(box.width * w) + 40
        bh = int(box.height * h) + 40

        x = max(0, x)
        y = max(0, y)
        bw = min(w - x, bw)
        bh = min(h - y, bh)

        face_crop = frame[y:y+bh, x:x+bw]

        if face_crop.size > 0:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = gray.astype("float32") / 255.0
            gray = np.expand_dims(gray, axis=[0, -1])

            preds = emotion_model.predict(gray, verbose=0)[0]
            current_emotion = EMOTIONS[np.argmax(preds)]

        emotion_window.append(current_emotion)

    # Smoothed emotion
    if len(emotion_window) > 0:
        smooth_emotion = max(set(emotion_window), key=emotion_window.count)
    else:
        smooth_emotion = current_emotion

    # -----------------------------
    # 2. Eyebrow-based stress score
    # -----------------------------
    fm = mp_face_mesh.process(rgb)
    stress_level = 0

    if fm.multi_face_landmarks:
        lm = fm.multi_face_landmarks[0].landmark
        stress_level, prev_raise = compute_stress(lm, w, h, prev_raise)

        # Draw eyebrow points
        for lid in LEFT_BROW + RIGHT_BROW:
            px = int(lm[lid].x * w)
            py = int(lm[lid].y * h)
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    # -----------------------------
    # 3. Draw Stress Meter Bar
    # -----------------------------
    bar_width = int(300 * stress_level)

    if stress_level < 0.35:
        color = (0, 255, 0)
    elif stress_level < 0.7:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(frame, (20, 20), (320, 60), (40, 40, 40), -1)
    cv2.rectangle(frame, (20, 20), (20 + bar_width, 60), color, -1)
    cv2.putText(frame, "Stress", (20, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -----------------------------
    # 4. Music Recommendation
    # -----------------------------
    music = recommend_music(smooth_emotion)

    cv2.putText(frame, f"Emotion: {smooth_emotion}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Music: {music}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 220), 2)

    # -----------------------------
    # 5. Display Frame
    # -----------------------------
    cv2.imshow("Real-Time Stress Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 