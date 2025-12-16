import cv2
import numpy as np
import mediapipe as mp
import webbrowser
from collections import deque
from tensorflow.keras.models import load_model

# ================================================
# 1. Load Mini-Xception model
# ================================================
MODEL_PATH = r"C:\Users\Sam Ocenar\Documents\Comp_vision\models\mini_xception.hdf5"
emotion_model = load_model(MODEL_PATH)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ================================================
# 2. Spotify Playlists Dictionary
# ================================================
SPOTIFY_LINKS = {
    "neutral":  "https://open.spotify.com/playlist/37i9dQZF1DX2UgsUIg75Vg",
    "happy":    "https://open.spotify.com/playlist/37i9dQZF1DX0UrRvztWcAU",
    "sad":      "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0",
    "angry":    "https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u",
    "fear":     "https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO",
    "surprise": "https://open.spotify.com/playlist/37i9dQZF1DX4AYpKfuYQSj",
    "disgust":  "https://open.spotify.com/playlist/37i9dQZF1DX3PFzdbtx1Us"
}

def recommend_music(emotion):
    return SPOTIFY_LINKS.get(emotion, SPOTIFY_LINKS["neutral"])

# For emotion smoothing
emotion_window = deque(maxlen=8)
last_played_emotion = None

# ================================================
# 3. Mediapipe Face Mesh
# ================================================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_face_det  = mp.solutions.face_detection.FaceDetection(model_selection=0)

LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [300, 293, 334, 296, 336]

# ================================================
# 4. Eyebrow Stress Function
# ================================================
def compute_stress(landmarks, w, h, prev_raise):
    left = (landmarks[70].y - landmarks[105].y) * h
    right = (landmarks[300].y - landmarks[334].y) * h
    avg_raise = (left + right) / 2

    contraction = abs((landmarks[70].x - landmarks[300].x) * w)
    motion = abs(avg_raise - prev_raise) if prev_raise is not None else 0

    stress_raw = (abs(avg_raise) * 0.4) + (motion * 0.4) + (contraction * 0.2)
    stress_norm = np.clip(stress_raw / 50, 0, 1)

    return stress_norm, avg_raise

# ================================================
# 5. Real-Time Webcam Loop
# ================================================
cap = cv2.VideoCapture(0)
prev_raise = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FACE DETECTION FOR EMOTION
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

        face = frame[y:y+bh, x:x+bw]
        if face.size > 0:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = gray.astype("float32") / 255.0
            gray = np.expand_dims(gray, axis=[0, -1])

            preds = emotion_model.predict(gray, verbose=0)[0]
            current_emotion = EMOTIONS[np.argmax(preds)]

        emotion_window.append(current_emotion)

    # SMOOTH EMOTION
    smooth_emotion = max(set(emotion_window), key=emotion_window.count)

    # SPOTIFY AUTO-OPEN WHEN EMOTION CHANGES
    if smooth_emotion != last_played_emotion:
        playlist_url = recommend_music(smooth_emotion)
        webbrowser.open(playlist_url)
        last_played_emotion = smooth_emotion

    # EYEBROW STRESS
    mesh = mp_face_mesh.process(rgb)
    stress_level = 0

    if mesh.multi_face_landmarks:
        lm = mesh.multi_face_landmarks[0].landmark
        stress_level, prev_raise = compute_stress(lm, w, h, prev_raise)

        for lid in LEFT_BROW + RIGHT_BROW:
            px = int(lm[lid].x * w)
            py = int(lm[lid].y * h)
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    # DRAW STRESS BAR
    bar_width = int(300 * stress_level)
    color = (0,255,0) if stress_level < 0.35 else (0,255,255) if stress_level < 0.7 else (0,0,255)

    cv2.rectangle(frame, (20,20), (320,60), (40,40,40), -1)
    cv2.rectangle(frame, (20,20), (20 + bar_width, 60), color, -1)
    cv2.putText(frame, "Stress", (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # DISPLAY TEXT
    cv2.putText(frame, f"Emotion: {smooth_emotion}", (20,110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.putText(frame, "Music: " + playlist_url.split("/")[-1],
                (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,200,220), 2)

    # SHOW FRAME
    cv2.imshow("Real-Time Emotion + Stress + Spotify", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
