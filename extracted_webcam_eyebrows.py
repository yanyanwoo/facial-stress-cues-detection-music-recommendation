import mediapipe as mp
import cv2
import csv
import os

# Input frames (webcam extracted)
INPUT_DIR = r"C:\Users\Sam Ocenar\Documents\Comp_vision\webcam_frames"

# Output CSV for features + RAW LANDMARKS
OUTPUT_CSV = r"C:\Users\Sam Ocenar\Documents\Comp_vision\webcam_eyebrows.csv"

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# Eyebrow landmark indices (5 points each)
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [300, 293, 334, 296, 336]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    # CSV HEADER
    writer.writerow([
        "actor", "emotion", "frame",
        # raw eyebrow coordinates (px)
        "lx1", "ly1", "lx2", "ly2", "lx3", "ly3", "lx4", "ly4", "lx5", "ly5",
        "rx1", "ry1", "rx2", "ry2", "rx3", "ry3", "rx4", "ry4", "rx5", "ry5",
        # numerical features
        "left_raise", "right_raise",
        "inner_brow_dist", "avg_raise",
        "contraction_raw", "motion_raw"
    ])

    prev_left = None
    prev_right = None

    for frame_name in sorted(os.listdir(INPUT_DIR)):
        if not frame_name.endswith(".jpg"):
            continue

        path = os.path.join(INPUT_DIR, frame_name)
        img = cv2.imread(path)
        if img is None:
            continue

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_face.process(rgb)

        if not result.multi_face_landmarks:
            continue

        lm = result.multi_face_landmarks[0].landmark

        # =============================
        # 1. EXTRACT RAW LANDMARK COORDS
        # =============================
        left_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_BROW]
        right_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_BROW]

        # =============================
        # 2. COMPUTE FEATURES
        # =============================

        # Vertical raise (using representative top/bottom points)
        left_raise = lm[70].y - lm[105].y
        right_raise = lm[300].y - lm[334].y

        # Inner brow contraction distance
        inner_dist = abs(lm[70].x - lm[300].x)

        # Brow raise average
        avg_raise = (left_raise + right_raise) / 2

        # Contraction metric (inverse)
        contraction_raw = 1 / (inner_dist + 1e-6)

        # Motion across frames
        if prev_left is None:
            motion_raw = 0
        else:
            motion_raw = abs(left_raise - prev_left) + abs(right_raise - prev_right)

        prev_left = left_raise
        prev_right = right_raise

        # =============================
        # 3. WRITE TO CSV
        # =============================
        writer.writerow([
            "webcam_sam", "none", frame_name,

            # left brow raw coordinates
            left_pts[0][0], left_pts[0][1],
            left_pts[1][0], left_pts[1][1],
            left_pts[2][0], left_pts[2][1],
            left_pts[3][0], left_pts[3][1],
            left_pts[4][0], left_pts[4][1],

            # right brow raw coordinates
            right_pts[0][0], right_pts[0][1],
            right_pts[1][0], right_pts[1][1],
            right_pts[2][0], right_pts[2][1],
            right_pts[3][0], right_pts[3][1],
            right_pts[4][0], right_pts[4][1],

            # computed features
            left_raise, right_raise,
            inner_dist, avg_raise,
            contraction_raw, motion_raw
        ])

print("Webcam eyebrow RAW LANDMARKS + FEATURES extracted!")
