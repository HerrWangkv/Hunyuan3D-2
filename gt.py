import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import matplotlib.pyplot as plt
import glob
import imageio.v2 as imageio
import argparse
from PIL import Image

def save_img(nusc, sample, save_path):
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    images = []
    for cam in cams:
        sample_data = nusc.get('sample_data', sample['data'][cam])
        img_path = os.path.join(nusc.dataroot, sample_data['filename'])
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    # Arrange images in two rows of three using PIL
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    row_height = max(heights)
    row1_width = sum(widths[:3])
    row2_width = sum(widths[3:])
    final_width = max(row1_width, row2_width)
    final_height = row_height * 2
    final_img = Image.new('RGB', (final_width, final_height))
    # Paste images directly into final_img
    x_offset = 0
    for img in images[:3]:
        final_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    x_offset = 0
    for img in images[3:]:
        final_img.paste(img, (x_offset, row_height))
        x_offset += img.size[0]
    final_img.save(save_path)

def main():
    # Initialize NuScenes dataset
    nusc = NuScenes(version="v1.0-trainval", dataroot="/data/nuscenes", verbose=True)
    val_scenes = [s for s in nusc.scene if s["name"] in splits.val]
    # Get scenes in validation set
    for scene_idx, scene in enumerate(val_scenes):
        print(f"Scene index: {scene_idx}, Scene name: {scene['name']}")
        first_sample_token = scene["first_sample_token"]
        sample = nusc.get("sample", first_sample_token)
        t = 0
        os.makedirs(f"val_videos/{scene_idx}/gt_images", exist_ok=True)
        while sample:
            save_img(
                nusc, sample, f"val_videos/{scene_idx}/gt_images/frame_{t:04d}.png"
            )

            # Move to the next sample
            if sample["next"] == "":
                break
            sample = nusc.get("sample", sample["next"])
            t += 1

        # Generate a video from the saved frames
        frame_paths = sorted(glob.glob(f"val_videos/{scene_idx}/gt_images/frame_*.png"))
        with imageio.get_writer(
            f"val_videos/{scene_idx}/gt_video.mp4", fps=2
        ) as video_writer:
            for frame_path in frame_paths:
                frame = imageio.imread(frame_path)
                video_writer.append_data(frame)

        # Clean up the frame images and remove the folder
        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(f"val_videos/{scene_idx}/gt_images")

if __name__ == "__main__":
    main()
