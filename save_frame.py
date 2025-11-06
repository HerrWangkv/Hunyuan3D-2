import cv2
import sys

# Usage: python save_frame.py <video_path> <frame_number> <output_image_path>

def save_frame(video_path, frame_number, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        if output_path is None:
            output_path = f"frame_{frame_number}.png"
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} saved to {output_path}")
    else:
        print(f"Error: Could not read frame {frame_number}")
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python save_frame.py <video_path> <frame_number> [output_image_path]")
        sys.exit(1)
    video_path = sys.argv[1]
    frame_number = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) == 4 else None
    save_frame(video_path, frame_number, output_path)
