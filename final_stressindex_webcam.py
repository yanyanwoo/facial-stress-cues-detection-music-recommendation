import pandas as pd
import numpy as np

# === Paths ===
EYEBROW_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\webcam_stress_scores.csv"
EMOTION_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\webcam_emotions.csv"
OUTPUT_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\webcam_final_stress_index.csv"

# === Load CSVs ===
eyebrow_df = pd.read_csv(EYEBROW_CSV)
emotion_df = pd.read_csv(EMOTION_CSV)

print("Eyebrow columns:", eyebrow_df.columns.tolist())
print("Emotion columns:", emotion_df.columns.tolist())

# Ensure column names match expectations
# Eyebrow: ["frame", StressScore]
# Emotion: ["frame", angry, disgust, fear, ...]

# === Compute Emotion Stress Component ===
def compute_emotion_stress(row):
    # Weighted emotion stress model
    # High-stress emotions: angry, fear, sad
    # Medium: disgust
    # Low: neutral, happy, surprise

    return (
        0.40 * row["angry"] +
        0.40 * row["fear"] +
        0.25 * row["sad"] +
        0.10 * row["disgust"] +
        0.05 * row["surprise"] +
        0.00 * row["happy"] +
        0.00 * row["neutral"]
    )

emotion_df["EmotionStress"] = emotion_df.apply(compute_emotion_stress, axis=1)

# === Merge by frame name ===
merged = pd.merge(eyebrow_df, emotion_df, on="frame", how="inner")

# === Normalize ===
merged["StressScore_norm"] = (merged["StressScore"] - merged["StressScore"].min()) / (
        merged["StressScore"].max() - merged["StressScore"].min()
)

merged["EmotionStress_norm"] = (merged["EmotionStress"] - merged["EmotionStress"].min()) / (
        merged["EmotionStress"].max() - merged["EmotionStress"].min()
)

# === Final Stress Index ===
# Eyebrow = 0.6 weight
# Emotion = 0.4 weight
merged["FinalStressIndex"] = (
        0.6 * merged["StressScore_norm"] +
        0.4 * merged["EmotionStress_norm"]
)

# === Save Output ===
merged.to_csv(OUTPUT_CSV, index=False)

print("\nSaved:", OUTPUT_CSV)
print("Preview:\n", merged.head())
