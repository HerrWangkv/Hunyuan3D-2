import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

import imageio.v2 as imageio
from tqdm import trange
import glob
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from copy import deepcopy
from gsplat.rendering import rasterization
from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from smpl.smpl import SMPL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 - y1 * z0 + z1 * y0 + w1 * x0
    y = x1 * z0 + y1 * w0 - z1 * x0 + w1 * y0
    z = -x1 * y0 + y1 * x0 + z1 * w0 + w1 * z0

    return torch.concat((w, x, y, z), dim=-1)


def initialize_openpose_pipe():
    """Initialize models once to be shared across all objects."""
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
    # )
    pipe.to("cuda")
    return pipe


def rpy2rotations(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw to rotation matrix.
    """
    import numpy as np
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    return torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ]).cuda()

def create_humans_batched(objects_info, openpose_pipe, max_batch_size=16):
    """Create multiple objects in a single batched diffusion run."""
    if not objects_info:
        return []
    # Split into smaller batches if needed
    prompts = []
    openpose_imgs = []
    human_models = []
    for i in range(len(objects_info)):
        obj_info = objects_info[i]
        prompt = obj_info["prompt"]
        prompts.append(prompt)
        if "female" in prompt:
            human_model = SMPL("smpl/SMPL_FEMALE.pkl", betas=torch.rand(1,10))
        else:
            human_model = SMPL("smpl/SMPL_MALE.pkl", betas=torch.rand(1,10))
        human_models.append(human_model)
        openpose_img = human_model.get_openpose_img(azimuth=180)
        openpose_img = Image.fromarray(openpose_img)
        # openpose_img.save(f"{i}.png")
        openpose_imgs.append(openpose_img)
    for i in range(0, len(prompts), max_batch_size):
        batch_prompts = prompts[i:i+max_batch_size]
        batch_prompts = [p + ", facing the camera, high quality, detailed" for p in batch_prompts]
        batch_openpose_imgs = openpose_imgs[i:i+max_batch_size]
        batch_ref_images = openpose_pipe(
            batch_prompts, 
            negative_prompt=["cartoon, blurry, drawing, sketch, toy-like"] * len(batch_prompts),
            num_inference_steps=20,
            image=batch_openpose_imgs,
        ).images
        if i == 0:
            all_ref_images = batch_ref_images
        else:
            all_ref_images += batch_ref_images
    all_created_objects = []
    for i in range(len(objects_info)):
        human_model = human_models[i]
        human_model.paint_mesh(all_ref_images[i])
        obj_info = objects_info[i]
        obj = Object3D(
            size=torch.tensor(obj_info["size"]).cuda(),
            prompt=obj_info["prompt"],
            human_model=human_model,
            pose_key=obj_info["pose_key"],
            inst_token=obj_info["inst_token"],
        )
        all_created_objects.append(obj)
    return all_created_objects


class Object3D:
    poses = {
        "stand_female": np.load("smpl/A1 - Stand_poses.npz")["poses"][
            :, :66
        ],  # 0.00 m/s
        "sway_female": np.load("smpl/A2 - Sway_poses.npz")["poses"][:, :66],  # 0.00 m/s
        "stand_male": np.load("smpl/A1- Stand_poses.npz")["poses"][:, :66],  # 0.00 m/s
        "sway_male": np.load("smpl/A2- Sway_poses.npz")["poses"][:, :66],  # 0.00 m/s
        "walk": np.load("smpl/B3 - walk1_poses.npz")["poses"][:, :66],  # 1.06 m/s
        "jog": np.load("smpl/C3 - run_poses.npz")["poses"][:, :66],  # 3.62 m/s
        "run": np.load("smpl/C3 - Run_poses.npz")["poses"][:, :66],  # 4.62 m/s
    }

    def __init__(
        self,
        size,
        prompt,
        human_model,
        pose_key="walk",
        initial_pose_idx=None,
        inst_token=None,
    ):
        self._size = size.clone().detach().cuda()
        self._size[[0, 1]] = self._size[[1, 0]]
        self._text = prompt
        self._human_model = human_model
        self._pose_key = pose_key

        # Make initial pose index more unique by incorporating instance token
        if initial_pose_idx is not None:
            self._initial_pose_idx = initial_pose_idx
        else:
            # Use hash of instance token to create deterministic but unique starting points
            if inst_token is not None:
                # Create a hash-based seed for this specific instance
                inst_hash = hash(inst_token) % 10000
                local_random = random.Random(inst_hash)
                self._initial_pose_idx = local_random.randint(
                    0, self.poses[self._pose_key].shape[0] - 1
                )
            else:
                self._initial_pose_idx = random.randint(
                    0, self.poses[self._pose_key].shape[0] - 1
                )

        initial_orient = torch.zeros([1, 3], device=DEVICE)  # global_orient
        initial_orient[0] = torch.from_numpy(
            self.poses[self._pose_key][self._initial_pose_idx, :3]
        ).to(DEVICE)
        initial_pose = torch.zeros([1, 69], device=DEVICE)  # 23*3 axis-angle
        initial_pose[0, :63] = torch.from_numpy(
            self.poses[self._pose_key][self._initial_pose_idx, 3:66]
        ).to(DEVICE)
        self._human_model.apply_pose(
            body_pose=initial_pose, global_orient=initial_orient
        )
        self.scale = None
        inst_id = (
            inst_token[-8:] if inst_token else "unknown"
        )  # Show last 8 chars of token
        print(
            f"  {prompt} (ID: {inst_id}) initialized with '{self._pose_key}' pose, starting from motion {self._initial_pose_idx}"
        )

    @property
    def _gs(self):
        return self._human_model.get_gs()

    @property
    def _centeralized_and_scaled_gs(self):
        gs = deepcopy(self._gs)
        if self._pose_key in [
            "stand_to_walk",
            "stand_female",
            "stand_male",
            "sway_female",
            "sway_male",
        ]:
            gs = self.rotate_gs(rpy2rotations(0, 0, -np.pi / 2), gs)
        elif self._pose_key == "walk":
            gs = self.rotate_gs(rpy2rotations(0, 0, 3 * np.pi / 4), gs)
        elif self._pose_key == "jog":
            gs = self.rotate_gs(rpy2rotations(0, 0, -5 * np.pi / 6), gs)
        elif self._pose_key == "run":
            gs = self.rotate_gs(rpy2rotations(0, 0, -3 * np.pi / 4), gs)
        valid_mask = gs["opacities"].squeeze() != 0
        x_min, x_max = gs["xyz"][valid_mask, 0].min(), gs["xyz"][valid_mask, 0].max()
        y_min, y_max = gs["xyz"][valid_mask, 1].min(), gs["xyz"][valid_mask, 1].max()
        z_min, z_max = (
            gs["xyz"][valid_mask, 2].min(),
            gs["xyz"][valid_mask, 2].max(),
        )
        center = torch.tensor(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        ).cuda()
        gs["xyz"] -= center
        if self.scale is None:
            self.scale = self._size[2] / (z_max - z_min)
        gs["xyz"] *= self.scale
        gs["scales"] *= self.scale
        return gs

    def apply_pose(self, delta_t):
        pose_idx = (delta_t + self._initial_pose_idx) % self.poses[
            self._pose_key
        ].shape[0]
        body_pose = torch.zeros([1, 69], device=DEVICE)  # 23*3 axis-angle
        body_pose[0, :63] = torch.from_numpy(
            self.poses[self._pose_key][pose_idx, 3:66]
        ).to(DEVICE)
        global_orient = torch.zeros([1, 3], device=DEVICE)  # global_orient
        global_orient[0] = torch.from_numpy(
            self.poses[self._pose_key][pose_idx, :3]
        ).to(DEVICE)
        self._human_model.apply_pose(body_pose=body_pose, global_orient=global_orient)

    def transform_gs(self, transformation_matrix, delta_t):
        '''
        Args:
            transformation_matrix: 4x4 transformation matrix
        '''
        self.apply_pose(delta_t=delta_t)
        gs = deepcopy(self._centeralized_and_scaled_gs)
        gs = self.rotate_gs(transformation_matrix[:3, :3], gs)
        gs = self.translate_gs(transformation_matrix[:3, 3], gs)
        return gs

    def rotate_gs(self, rotation_matrix, gs):
        '''
        Args:
            rotation_matrix: 3x3 rotation matrix
        '''
        rotated_xyz = gs['xyz'].double() @ rotation_matrix.T
        rotated_rotations = F.normalize(quat_multiply(
            torch.tensor(Quaternion(matrix=rotation_matrix.cpu().numpy()).elements).cuda(),
            gs['rots'],
        ))
        gs['xyz'] = rotated_xyz.to(torch.float32)
        gs['rots'] = rotated_rotations.to(torch.float32)
        return gs

    def translate_gs(self, translation, gs):
        '''
        Args:
            translation: 3 translation vector
        '''
        gs['xyz'] += translation
        return gs

def get_obj_to_cam_front(rotation, translation, cam_front_to_world):
    '''
    Args:
        rotation: 3x3 rotation matrix
        translation: 3 translation vector
        cam_front_to_world: 4x4 camera front to world matrix
    '''
    obj_to_world = np.eye(4)
    obj_to_world[:3, :3] = Quaternion(rotation).rotation_matrix
    obj_to_world[:3, 3] = translation
    obj_to_cam_front = np.linalg.inv(cam_front_to_world) @ obj_to_world
    return torch.tensor(obj_to_cam_front).cuda()

def all_to_camera_front(nusc, cam_calib_tokens):
    '''
    Args:
        cam_calib_tokens: list of camera tokens
    '''
    cam_front_calib_token = cam_calib_tokens["CAM_FRONT"]
    cam_front_calib_data = nusc.get('calibrated_sensor', cam_front_calib_token)
    cam_front_to_ego = np.eye(4)
    cam_front_to_ego[:3, :3] = Quaternion(cam_front_calib_data['rotation']).rotation_matrix
    cam_front_to_ego[:3, 3] = np.array(cam_front_calib_data['translation'])
    ret = {}
    for cam in cam_calib_tokens.keys():
        if cam == "CAM_FRONT":
            ret[cam] = np.eye(4)
        else:
            cam_to_ego = np.eye(4)
            calib_token = cam_calib_tokens[cam]
            calib_data = nusc.get('calibrated_sensor', calib_token)
            cam_to_ego[:3, :3] = Quaternion(calib_data['rotation']).rotation_matrix
            cam_to_ego[:3, 3] = np.array(calib_data['translation'])
            ret[cam] = np.linalg.inv(cam_front_to_ego) @ cam_to_ego
    return ret, cam_front_to_ego

def render_gaussian(gaussian, extrinsics, intrinsics, width=533, height=300):
    if gaussian is None:
        return torch.ones(1, height, width, 3).cuda()
    extrinsics = torch.tensor(extrinsics).float().cuda()
    intrinsics = torch.tensor(intrinsics).float().cuda()
    intrinsics[0] *= width / 1600
    intrinsics[1] *= height / 900
    means = gaussian["xyz"]
    rgbs = gaussian["rgbs"]
    opacities = gaussian["opacities"]
    scales = gaussian["scales"]
    rotations = gaussian["rots"]

    renders, _, _ = rasterization(
        means=means,
        quats=rotations,
        scales=scales,
        opacities=opacities.squeeze(),
        colors=rgbs,
        viewmats=torch.linalg.inv(extrinsics)[None, ...],  # [C, 4, 4]
        Ks=intrinsics[None, ...],  # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=True,
        sparse_grad=False,
        rasterize_mode="classic",
        near_plane=0.1,
        far_plane=10000000000.0,
        render_mode="RGB",
        radius_clip=0.,
        backgrounds=torch.ones(1, 3).cuda(),
    )
    renders = torch.clamp(renders, max=1.0)
    return renders

def render(gs, intrinsics, extrinsics, save_path='render.png'):
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    images = []
    for i in range(len(cams)):
        intrinsic = intrinsics[cams[i]]
        extrinsic = extrinsics[cams[i]]
        img = render_gaussian(gs, extrinsic, intrinsic)
        img = img[0].detach().cpu().numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
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


def _orthonormalize(mat: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto the closest valid rotation matrix using SVD."""
    U, _, Vt = np.linalg.svd(mat)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def interpolate_transform_matrix(
    mat1: torch.Tensor, mat2: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Interpolate between two 4x4 transformation matrices.

    Args:
        mat1, mat2 : torch.Tensor of shape (4,4)
        alpha      : float in [0,1], interpolation parameter
    Returns:
        torch.Tensor of shape (4,4), same dtype/device as inputs
    """
    device = mat1.device
    dtype = mat1.dtype

    # --- Translation ---
    t1 = mat1[:3, 3].cpu().numpy()
    t2 = mat2[:3, 3].cpu().numpy()
    translation = (1 - alpha) * t1 + alpha * t2

    # --- Rotation ---
    r1 = R.from_matrix(mat1[:3, :3].cpu().numpy())
    r2 = R.from_matrix(mat2[:3, :3].cpu().numpy())
    key_times = [0, 1]
    key_rots = R.from_quat([r1.as_quat(), r2.as_quat()])
    slerp = Slerp(key_times, key_rots)
    interp_rot = slerp([alpha]).as_matrix()[0]

    # Re-orthogonalize to avoid pyquaternion ValueError
    interp_rot = _orthonormalize(interp_rot)

    # --- Compose back ---
    result = torch.eye(4, dtype=dtype, device=device)
    result[:3, :3] = torch.tensor(interp_rot, dtype=dtype, device=device)
    result[:3, 3] = torch.tensor(translation, dtype=dtype, device=device)

    return result.double()


def parse_args():
    parser = argparse.ArgumentParser(description="Render 3D objects in a scene.")
    parser.add_argument("--scene-idx", type=int, default=0, help="Index of the scene to render.")
    parser.add_argument(
        "--smpl-lora", type=str, default="lora_ckpts/smpl.pt", help="Path to the smpl LoRA model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=2,
        help="Frame rate for the output video.",
        choices=[2 * i for i in range(1, 61)],
    )
    return parser.parse_args()

def main():
    nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes", verbose=True)
    print("Multiple GPUs not yet supported...")
    args = parse_args()
    # Make seed scene-specific to avoid identical runs
    scene_seed = args.seed + args.scene_idx * 1000
    seed_everything(scene_seed)
    print(f"Using seed {scene_seed} for scene {args.scene_idx}")
    openpose_pipe = initialize_openpose_pipe()
    scene = nusc.scene[args.scene_idx]
    sample = None
    objects = {}
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    # Collect new objects to create in parallel
    humans_to_create = []
    existing_humans = []
    annotations_2hz = {}
    positions = {}
    for t in range(scene["nbr_samples"]):
        sample_token = scene["first_sample_token"] if sample is None else sample["next"]
        sample = nusc.get("sample", sample_token)
        cam_tokens = {cam: sample["data"][cam] for cam in cams}
        cam_data = {cam: nusc.get("sample_data", cam_tokens[cam]) for cam in cam_tokens}
        ego_pose = nusc.get("ego_pose", cam_data["CAM_FRONT"]["ego_pose_token"])
        if t == 0:
            cam_calib_tokens = {
                cam: cam_data[cam]["calibrated_sensor_token"] for cam in cam_tokens
            }
            intrinsics = {
                cam: np.array(
                    nusc.get("calibrated_sensor", cam_calib_tokens[cam])[
                        "camera_intrinsic"
                    ]
                )
                for cam in cam_tokens
            }
            extrinsics, cam_front_to_ego = all_to_camera_front(nusc, cam_calib_tokens)
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
        ego_to_world[:3, 3] = ego_pose["translation"]
        cam_front_to_world = ego_to_world @ cam_front_to_ego
        for _, ann_token in enumerate(sample["anns"]):
            ann = nusc.get("sample_annotation", ann_token)
            if not ann["category_name"].startswith("human."):
                continue
            size = ann["size"]
            inst_token = ann["instance_token"]

            if inst_token not in existing_humans:
                prompt = np.random.choice(["A female real-world pedestrian", "A male real-world pedestrian"])
                # Queue for parallel creation
                humans_to_create.append(
                    {
                        "inst_token": inst_token,
                        "size": size,
                        "prompt": prompt,
                        "pose_key": (
                            np.random.choice(
                                ["stand_female", "sway_female"], p=[0.7, 0.3]
                            )
                            if "female" in prompt
                            else np.random.choice(
                                ["stand_male", "sway_male"], p=[0.7, 0.3]
                            )
                        ),
                    }
                )
                existing_humans.append(inst_token)
            obj_to_cam_front = get_obj_to_cam_front(
                ann["rotation"],
                ann["translation"],
                cam_front_to_world,
            )
            if inst_token in annotations_2hz:
                annotations_2hz[inst_token][t] = obj_to_cam_front
                positions[inst_token][t] = ann["translation"]
            else:
                annotations_2hz[inst_token] = {t: obj_to_cam_front}
                positions[inst_token] = {t: ann["translation"]}
    del existing_humans

    # Create interpolated annotations with higher frame rate
    interpolation_factor = args.hz // 2  # Convert from 2Hz to desired Hz
    annotations_required = {}

    for inst_token in annotations_2hz:
        annotations_required[inst_token] = {}
        time_keys = sorted(annotations_2hz[inst_token].keys())
        velocities = []

        # For each pair of consecutive keyframes, interpolate
        for i in range(len(time_keys) - 1):
            t1, t2 = time_keys[i], time_keys[i + 1]
            mat1 = annotations_2hz[inst_token][t1]
            mat2 = annotations_2hz[inst_token][t2]
            pos1 = np.array(positions[inst_token][t1])
            pos2 = np.array(positions[inst_token][t2])
            velocities.append(
                np.sum((pos2 - pos1) ** 2) ** 0.5 / (t2 - t1) * 2
            )  # annotations are at 2hz
            # Add the first keyframe
            for interp_step in range(interpolation_factor):
                interp_t = t1 * interpolation_factor + interp_step
                if interp_step == 0:
                    # Use exact keyframe
                    annotations_required[inst_token][interp_t] = mat1
                else:
                    # Interpolate
                    alpha = interp_step / interpolation_factor
                    interp_mat = interpolate_transform_matrix(mat1, mat2, alpha)
                    annotations_required[inst_token][interp_t] = interp_mat

        # Add the last keyframe
        if time_keys:
            last_t = time_keys[-1]
            for interp_step in range(interpolation_factor):
                interp_t = last_t * interpolation_factor + interp_step
                if interp_step == 0:
                    annotations_required[inst_token][interp_t] = annotations_2hz[
                        inst_token
                    ][last_t]
                else:
                    # For the last frame, just repeat the last pose
                    annotations_required[inst_token][interp_t] = annotations_2hz[
                        inst_token
                    ][last_t]
        avg_velocity = sum(velocities) / len(velocities) if velocities else 0
        if avg_velocity > 4.0:  # running
            pose_key = "run"
        elif avg_velocity > 2.0:  # jogging
            pose_key = "jog"
        elif avg_velocity > 0.5:  # walking
            pose_key = "walk"
        else:
            continue
        for obj in humans_to_create:
            if obj["inst_token"] == inst_token:
                obj["pose_key"] = pose_key
                break

    created_humans = create_humans_batched(humans_to_create, openpose_pipe)
    print(f"Created {len(created_humans)} human objects for scene {args.scene_idx}")

    # Store created objects and transform them
    for obj, obj_info in zip(created_humans, humans_to_create):
        objects[obj_info["inst_token"]] = obj

    for t in trange(scene["nbr_samples"] * interpolation_factor):
        gs = None
        current_smpl_gs = []
        for inst_token in objects.keys():
            if t not in annotations_required[inst_token]:
                continue
            obj = objects[inst_token]
            obj_to_cam_front = annotations_required[inst_token][t]
            obj_gs = obj.transform_gs(
                obj_to_cam_front, (t * 120) // args.hz
            )  # smpl poses are at 120hz
            current_smpl_gs.append(obj_gs)

        if current_smpl_gs:
            gs = current_smpl_gs[0]
            for obj_gs in current_smpl_gs[1:]:
                for key in gs.keys():
                    gs[key] = torch.vstack([gs[key], obj_gs[key]])
        else:
            gs = None

        # Save the rendered image for the current frame
        os.makedirs(f"videos/{args.scene_idx}/rendered_images", exist_ok=True)
        image_path = f"videos/{args.scene_idx}/rendered_images/frame_{t:04d}.png"
        render(gs, intrinsics, extrinsics, save_path=image_path)

    # Generate a video from the saved frames
    frame_paths = sorted(glob.glob(f"videos/{args.scene_idx}/rendered_images/frame_*.png"))
    with imageio.get_writer(
        f"videos/{args.scene_idx}/objects.mp4", mode="I", fps=args.hz, format="FFMPEG"
    ) as video_writer:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            video_writer.append_data(frame)

    # Clean up the frame images and remove the folder
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(f"videos/{args.scene_idx}/rendered_images")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
