import cv2
import os
import numpy as np
import argparse
import imageio.v2 as imageio

def vstack_videos(video_path1, video_path2, output_path):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate target width and heights while maintaining aspect ratios
    target_width = min(width1, width2)
    
    # Calculate new heights based on aspect ratio
    aspect_ratio1 = width1 / height1
    aspect_ratio2 = width2 / height2
    new_height1 = int(target_width / aspect_ratio1)
    new_height2 = int(target_width / aspect_ratio2)
    
    # Use the higher frame rate for output to maintain quality
    output_fps = max(fps1, fps2)
    
    # Calculate durations
    duration1 = frame_count1 / fps1
    duration2 = frame_count2 / fps2
    output_duration = min(duration1, duration2)
    
    # Read all frames from both videos
    frames1 = []
    frames2 = []
    
    # Read video 1
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        if width1 != target_width:
            frame = cv2.resize(frame, (target_width, new_height1))
        frames1.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Read video 2  
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        if width2 != target_width:
            frame = cv2.resize(frame, (target_width, new_height2))
        frames2.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap1.release()
    cap2.release()
    
    # Create synchronized output frames
    output_frames = []
    output_frame_count = int(output_duration * output_fps)
    
    for i in range(output_frame_count):
        # Calculate time for this output frame
        t = i / output_fps
        
        # Find corresponding frame indices in each video
        frame_idx1 = min(int(t * fps1), len(frames1) - 1)
        frame_idx2 = min(int(t * fps2), len(frames2) - 1)
        
        # Stack the synchronized frames
        stacked = np.vstack((frames1[frame_idx1], frames2[frame_idx2]))
        output_frames.append(stacked)
    
    # Use imageio to write video (same as gt.py)
    with imageio.get_writer(output_path, fps=output_fps) as video_writer:
        for frame in output_frames:
            video_writer.append_data(frame)

def parse_args():
    parser = argparse.ArgumentParser(description="Vertically stack gt and simulation videos.")
    parser.add_argument("--dir", type=str, default="videos")
    return parser.parse_args()

def main():
    args = parse_args()
    for scene_idx in os.listdir(args.dir):
        video_path1 = f"{args.dir}/{scene_idx}/gt_video.mp4"
        video_path2 = f"{args.dir}/{scene_idx}/objects.mp4"
        output_path = f"{args.dir}/{scene_idx}/vstacked_video.mp4"
        vstack_videos(video_path1, video_path2, output_path)

if __name__ == "__main__":
    main()