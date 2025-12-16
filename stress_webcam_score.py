import pandas as pd

INPUT = r"C:\Users\Sam Ocenar\Documents\Comp_vision\webcam_eyebrows.csv"
OUTPUT = r"C:\Users\Sam Ocenar\Documents\Comp_vision\webcam_stress_scores.csv"

df = pd.read_csv(INPUT)

# Normalize values (0–1)
df["raise_norm"] = (df["avg_raise"] - df["avg_raise"].min()) / (df["avg_raise"].max() - df["avg_raise"].min())
df["contraction_norm"] = (df["inner_brow_dist"] - df["inner_brow_dist"].min()) / (df["inner_brow_dist"].max() - df["inner_brow_dist"].min())
df["motion_norm"] = (df["motion_raw"] - df["motion_raw"].min()) / (df["motion_raw"].max() - df["motion_raw"].min())

# Weighted stress components
df["StressScore"] = (
    10 * df["raise_norm"] +
    15 * df["motion_norm"] +
    8 * df["contraction_norm"]
)

df.to_csv(OUTPUT, index=False)
print("Webcam StressScore computed!")
