import cv2
import mediapipe as mp
import os

# Base directory where your frames are located
BASE_DIR = r"C:\Users\Sam Ocenar\Documents\Comp_vision\extracted_frames"

# Output folder for eyebrow-annotated frames
OUTPUT_DIR = r"C:\Users\Sam Ocenar\Documents\Comp_vision\extracted_eyebrows"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYEBROW = [105, 107, 66, 70, 63]
RIGHT_EYEBROW = [334, 336, 296, 300, 293]

def process_frame(img_path, output_path, face_mesh):
    img = cv2.imread(img_path)
    if img is None:
        print("Could not open:", img_path)
        return

    h, w = img.shape[:2]
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw eyebrow points
            for idx in LEFT_EYEBROW + RIGHT_EYEBROW:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # Save annotated frame
    cv2.imwrite(output_path, img)
    print("Saved:", output_path)


def run_all_subjects():
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for actor in sorted(os.listdir(BASE_DIR)):
            actor_folder = os.path.join(BASE_DIR, actor)
            if not os.path.isdir(actor_folder):
                continue

            print(f"\n=== Processing {actor} ===")

            for emotion_folder in os.listdir(actor_folder):
                emo_path = os.path.join(actor_folder, emotion_folder)
                if not os.path.isdir(emo_path):
                    continue

                print(f" -> {emotion_folder}")

                # Output directory
                out_dir = os.path.join(OUTPUT_DIR, actor, emotion_folder)
                os.makedirs(out_dir, exist_ok=True)

                # Process all frames within this emotion folder
                for frame_name in os.listdir(emo_path):
                    if not frame_name.endswith(".jpg"):
                        continue

                    img_path = os.path.join(emo_path, frame_name)
                    out_path = os.path.join(out_dir, frame_name)

                    process_frame(img_path, out_path, face_mesh)


if __name__ == "__main__":
    run_all_subjects()
