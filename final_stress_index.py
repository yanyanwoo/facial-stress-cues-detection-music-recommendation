import pandas as pd
import numpy as np
import re

# ======================================================
# PATHS
# ======================================================
EMOTION_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\emotion_probs.csv"
EYEBROW_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\eyebrow_stress_scores.csv"
OUTPUT_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\final_stress_index.csv"

# ======================================================
# RAVDESS → FER EMOTION LABEL NORMALIZATION
# ======================================================
emotion_map = {
    "angry": "angry",
    "fearful": "fear",
    "sad": "sad",
    "happy": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "surprise": "surprise",
    "disgust": "disgust"
}

# ======================================================
# LITERATURE-BASED EMOTION STRESS WEIGHTS
# (arousal-driven, not classifier accuracy driven)
# ======================================================
emotion_weights = {
    "angry": 1.0,
    "fear": 1.0,
    "sad": 0.7,
    "disgust": 0.6,
    "surprise": 0.4,
    "neutral": 0.1,
    "happy": 0.0
}

# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def clean_emotion_label(label):
    """
    Normalize RAVDESS emotion labels safely:
    - lowercase
    - remove intensity suffixes
    """
    label = label.lower()
    label = re.sub(r"_intensity\d+", "", label)
    return label

def normalize(series):
    """Safe min-max normalization"""
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

# ======================================================
# FINAL STRESS COMPUTATION (RAVDESS-SAFE)
# ======================================================
def compute_final_stress():

    # --------------------------------------------------
    # Load CSVs
    # --------------------------------------------------
    emo = pd.read_csv(EMOTION_CSV)
    brow = pd.read_csv(EYEBROW_CSV)

    # --------------------------------------------------
    # Merge on actor, emotion folder, frame
    # --------------------------------------------------
    merged = pd.merge(
        brow,
        emo,
        on=["actor", "emotion", "frame"],
        how="inner"
    )

    if merged.empty:
        raise ValueError("Merged dataframe is empty. Check CSV alignment.")

    # --------------------------------------------------
    # CLEAN + MAP EMOTION LABELS
    # --------------------------------------------------
    merged["emotion_clean"] = merged["emotion"].apply(clean_emotion_label)
    merged["emotion_mapped"] = merged["emotion_clean"].map(emotion_map)

    if merged["emotion_mapped"].isna().all():
        raise ValueError("Emotion mapping failed completely. Check folder names.")

    # --------------------------------------------------
    # EMOTION-DERIVED STRESS (PRIMARY SIGNAL)
    # NOTE: DO NOT min-max normalize this
    # Probabilities are already normalized
    # --------------------------------------------------
    merged["EmotionStress"] = sum(
        merged[e] * emotion_weights[e]
        for e in emotion_weights.keys()
    )

    merged["EmotionStress_norm"] = merged["EmotionStress"].clip(0, 1)

    # --------------------------------------------------
    # EYEBROW STRESS (INTENSITY MODULATOR)
    # --------------------------------------------------
    merged["StressScore_norm"] = normalize(merged["StressScore"])

    # --------------------------------------------------
    # FINAL STRESS INDEX
    # Emotion defines stress
    # Eyebrow motion modulates intensity
    # --------------------------------------------------
    merged["FinalStressIndex"] = (
        merged["EmotionStress_norm"] *
        (0.6 + 0.4 * merged["StressScore_norm"])
    )

    # --------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------
    merged.to_csv(OUTPUT_CSV, index=False)

    print("✔ Final stress index generated successfully")
    print("Rows:", len(merged))
    print("Saved to:", OUTPUT_CSV)

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    compute_final_stress()
    print("DONE ✔")
