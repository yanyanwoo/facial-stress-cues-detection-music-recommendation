import cv2
import numpy as np
import os
import csv
from keras.models import load_model

# Paths
BASE_DIR = r"C:\Users\Sam Ocenar\Documents\Comp_vision\extracted_frames"
MODEL_PATH = r"C:\Users\Sam Ocenar\Documents\Comp_vision\models\mini_xception.hdf5"
OUTPUT_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\emotion_probs.csv"

# Load Mini-Xception
model = load_model(MODEL_PATH, compile=False)

# FER2013 emotion classes
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)
    return reshaped


def run_emotion_model():
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actor", "emotion", "frame"] + EMOTIONS)

        for actor in sorted(os.listdir(BASE_DIR)):
            actor_folder = os.path.join(BASE_DIR, actor)
            if not os.path.isdir(actor_folder):
                continue

            print(f"\n=== Processing {actor} ===")

            for emotion_folder in sorted(os.listdir(actor_folder)):
                emo_path = os.path.join(actor_folder, emotion_folder)
                if not os.path.isdir(emo_path):
                    continue

                print(f" -> {emotion_folder}")

                for frame_name in sorted(os.listdir(emo_path)):
                    if not frame_name.endswith(".jpg"):
                        continue

                    frame_path = os.path.join(emo_path, frame_name)
                    img_in = preprocess_image(frame_path)
                    if img_in is None:
                        continue

                    preds = model.predict(img_in, verbose=0)[0]

                    # Save to CSV
                    row = [actor, emotion_folder, frame_name] + preds.tolist()
                    writer.writerow(row)

                    print("   Saved", frame_name)


if __name__ == "__main__":
    run_emotion_model()
