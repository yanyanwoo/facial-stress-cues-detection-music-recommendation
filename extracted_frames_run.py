import os
import cv2
from parse_ravdess import parse_ravdess_filename

# Path to RAVDESS master folder
RAVDESS_PATH = r"C:\Users\Sam Ocenar\Documents\Comp_vision\RAVDESS"
OUTPUT_PATH = r"C:\Users\Sam Ocenar\Documents\Comp_vision\extracted_frames"

def extract_frames_from_video(video_path, output_folder, save_every_n=5):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % save_every_n == 0:
            frame_file = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_file, frame)
            saved += 1

        frame_index += 1

    cap.release()
    print(f"Extracted {saved} frames → {output_folder}")


def process_actor_folder(actor_folder):
    actor_path = os.path.join(RAVDESS_PATH, actor_folder)
    output_actor_path = os.path.join(OUTPUT_PATH, actor_folder)

    for filename in os.listdir(actor_path):
        if filename.endswith(".mp4"):
            video_path = os.path.join(actor_path, filename)
            info = parse_ravdess_filename(filename)

            # Create structured output path using emotion + intensity
            video_output_folder = os.path.join(
                output_actor_path,
                f"{info['emotion']}_intensity{info['inte nsity']}"
            )

            print(f"Processing {video_path} → {video_output_folder}")
            extract_frames_from_video(video_path, video_output_folder)


def run_all_actors():
    for folder in os.listdir(RAVDESS_PATH):
        if folder.startswith("Actor"):
            print(f"\n=== Processing {folder} ===")
            process_actor_folder(folder)


if __name__ == "__main__":
    run_all_actors()
