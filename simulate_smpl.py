import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

import imageio.v2 as imageio
from tqdm import trange
import glob
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from copy import deepcopy
from gsplat.rendering import rasterization
from PIL import Image
from scipy.spatial.transform import Rotation as R, Slerp
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from smpl.smpl import SMPL
from hy3dgen.texgen import Hunyuan3DPaintPipeline


def get_device(local_rank=None):
    """Get the appropriate device based on rank."""
    if torch.cuda.is_available():
        if local_rank is not None:
            return f"cuda:{local_rank}"
        return "cuda"
    return "cpu"


def setup_distributed(rank, world_size, master_port=12355):
    """Initialize distributed training with better error handling."""
    import socket

    # Find a free port if the default is not available
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    # Try multiple backends and addresses
    backends = ["nccl", "gloo"]  # NCCL for GPU, Gloo as fallback
    master_addresses = ["127.0.0.1", "localhost", "0.0.0.0"]

    for backend in backends:
        print(f"Trying backend: {backend}")

        for master_addr in master_addresses:
            try:
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = str(master_port)

                # Set additional environment variables for better compatibility
                if backend == "nccl":
                    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
                    os.environ["NCCL_P2P_DISABLE"] = "1"
                    os.environ["NCCL_IB_DISABLE"] = "1"

                print(
                    f"Attempting to initialize {backend} backend on {master_addr}:{master_port}"
                )

                # Initialize with timeout
                dist.init_process_group(
                    backend=backend,
                    rank=rank,
                    world_size=world_size,
                    timeout=torch.distributed.default_pg_timeout,
                )
                torch.cuda.set_device(rank)
                print(
                    f"Successfully initialized distributed training on rank {rank} with {backend} backend at {master_addr}:{master_port}"
                )
                return  # Success, exit the function

            except Exception as e:
                print(
                    f"Failed to initialize {backend} with {master_addr}:{master_port}: {e}"
                )
                continue

        # If current backend failed with all addresses, try with a different port
        if backend == backends[0]:  # Only try different port for first backend
            new_port = find_free_port()
            print(f"Retrying {backend} with a different port: {new_port}")

            for master_addr in master_addresses:
                try:
                    os.environ["MASTER_ADDR"] = master_addr
                    os.environ["MASTER_PORT"] = str(new_port)

                    dist.init_process_group(
                        backend=backend,
                        rank=rank,
                        world_size=world_size,
                        timeout=torch.distributed.default_pg_timeout,
                    )
                    torch.cuda.set_device(rank)
                    print(
                        f"Successfully initialized distributed training on rank {rank} with {backend} backend at {master_addr}:{new_port}"
                    )
                    return  # Success, exit the function

                except Exception as e:
                    print(
                        f"Failed to initialize {backend} with {master_addr}:{new_port}: {e}"
                    )
                    continue

    # If we get here, all attempts failed
    raise RuntimeError(
        "Could not initialize distributed training with any backend/address/port combination"
    )


def seed_everything(seed: int, rank: int = 0):
    """Set random seed for reproducibility across multiple processes."""
    # Make seed different for each process to avoid identical behavior
    process_seed = seed + rank
    random.seed(process_seed)
    os.environ["PYTHONHASHSEED"] = str(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)  # For multi-GPU


def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 - y1 * z0 + z1 * y0 + w1 * x0
    y = x1 * z0 + y1 * w0 - z1 * x0 + w1 * y0
    z = -x1 * y0 + y1 * x0 + z1 * w0 + w1 * z0

    return torch.concat((w, x, y, z), dim=-1)


def initialize_pipelines(device="cuda", local_rank=None):
    """Initialize models once to be shared across all objects."""
    if local_rank is not None:
        device = f"cuda:{local_rank}"

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    )

    openpose_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    openpose_pipe.scheduler = UniPCMultistepScheduler.from_config(
        openpose_pipe.scheduler.config
    )
    # controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
    # openpose_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
    # )
    openpose_pipe.to(device)

    # Initialize paint pipeline directly here to be shared across all objects

    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0-turbo"
    )
    paint_pipeline.config.device = str(device)
    paint_pipeline.render.device = str(device)
    paint_pipeline.load_models()

    return openpose_pipe, paint_pipeline


def rpy2rotations(roll, pitch, yaw, device="cuda"):
    """
    Convert roll, pitch, yaw to rotation matrix.
    """
    import numpy as np
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    ).to(device)


def create_humans_batched(
    objects_info,
    openpose_pipe,
    paint_pipeline,
    max_batch_size=16,
    rank=0,
    world_size=1,
):
    """Create multiple objects in a single batched diffusion run with multi-GPU support."""
    device = get_device(rank) if world_size > 1 else get_device()
    # print(f"Rank {rank}: Using device {device} for human creation")

    # Handle empty objects_info for single GPU mode
    if not objects_info and world_size == 1:
        return []

    # Distribute objects across GPUs more evenly
    # This ensures better load balancing even when objects don't divide evenly
    objects_per_gpu_base = len(objects_info) // world_size if objects_info else 0
    extra_objects = len(objects_info) % world_size if objects_info else 0

    # Calculate start and end indices for this rank
    if objects_info and rank < extra_objects:
        # This rank gets one extra object
        start_idx = rank * (objects_per_gpu_base + 1)
        end_idx = start_idx + objects_per_gpu_base + 1
    elif objects_info:
        # This rank gets the base number of objects
        start_idx = (
            extra_objects * (objects_per_gpu_base + 1)
            + (rank - extra_objects) * objects_per_gpu_base
        )
        end_idx = start_idx + objects_per_gpu_base
    else:
        # No objects to distribute
        start_idx = end_idx = 0

    local_objects_info = objects_info[start_idx:end_idx] if objects_info else []
    # print(
    #     f"Rank {rank}: Processing objects {start_idx}-{end_idx-1} ({len(local_objects_info)} objects)"
    # )

    if not local_objects_info:
        # print(f"Rank {rank}: No objects assigned to this rank (idle GPU)")
        # Create empty local objects list for idle GPUs, but still participate in sharing
        local_created_objects = []
    else:
        # Split into smaller batches if needed
        prompts = []
        openpose_imgs = []
        human_models = []

        for i in range(len(local_objects_info)):
            obj_info = local_objects_info[i]
            prompt = obj_info["prompt"]
            prompts.append(prompt)

            # Create SMPL model on the correct device
            if "female" in prompt:
                human_model = SMPL(
                    "smpl/SMPL_FEMALE.pkl", betas=torch.rand(1, 10), device=device
                )
            else:
                human_model = SMPL(
                    "smpl/SMPL_MALE.pkl", betas=torch.rand(1, 10), device=device
                )

            human_models.append(human_model)
            openpose_img = human_model.get_openpose_img(azimuth=180)
            openpose_img = Image.fromarray(openpose_img)
            openpose_imgs.append(openpose_img)

        local_ref_images = []
        for i in range(0, len(prompts), max_batch_size):
            batch_prompts = prompts[i : i + max_batch_size]
            batch_prompts = [
                p + ", facing the camera, high quality, detailed" for p in batch_prompts
            ]
            batch_openpose_imgs = openpose_imgs[i : i + max_batch_size]

            batch_ref_images = openpose_pipe(
                batch_prompts,
                negative_prompt=["cartoon, blurry, drawing, sketch, toy-like"]
                * len(batch_prompts),
                num_inference_steps=20,
                image=batch_openpose_imgs,
            ).images
            local_ref_images.extend(batch_ref_images)

        local_created_objects = []
        for i in range(len(local_objects_info)):
            human_model = human_models[i]
            human_model.paint_mesh(local_ref_images[i], paint_pipeline)
            obj_info = local_objects_info[i]
            obj = Object3D(
                size=torch.tensor(obj_info["size"]).to(device),
                prompt=obj_info["prompt"],
                human_model=human_model,
                pose_key=obj_info["pose_key"],
                inst_token=obj_info["inst_token"],
                device=device,
            )
            local_created_objects.append(obj)

    # print(f"Rank {rank}: Created {len(local_created_objects)} humans")

    # For multi-GPU: First ensure ALL ranks finish object creation, then share all objects
    if world_size > 1 and dist.is_initialized():
        # CRITICAL: All ranks must finish object creation before any sharing begins
        # print(f"Rank {rank}: Waiting for all ranks to complete object creation...")
        # Specify device_ids to avoid NCCL warning and potential hang
        device_id = torch.cuda.current_device()
        dist.barrier(device_ids=[device_id])
        # print(
        #     f"Rank {rank}: All ranks completed object creation. Starting object sharing..."
        # )

        # Serialize local objects with all necessary data including colors
        local_object_data = []
        for obj in local_created_objects:
            obj_data = {
                "inst_token": obj.inst_token,
                "prompt": obj._text,
                "pose_key": obj._pose_key,
                "size": obj._size.cpu().tolist(),
                "initial_pose_idx": obj._initial_pose_idx,
                "betas": (obj._human_model.betas.cpu().tolist()),
                "colors": (obj._human_model.colors.cpu().tolist()),
                "vertices": (obj._human_model.vertices.tolist()),
                "rest_vertices": (obj._human_model.rest_vertices.tolist()),
            }
            local_object_data.append(obj_data)

        # Now gather all object data from all ranks
        gathered_objects = [None] * world_size
        dist.all_gather_object(gathered_objects, local_object_data)

        # Reconstruct all objects on all ranks (not just rank 0)
        device = get_device(rank)
        all_objects = []

        for rank_objects in gathered_objects:
            if rank_objects:
                for obj_data in rank_objects:
                    # Check if we already have this object locally (to avoid duplication)
                    local_obj = None
                    for local_obj_check in local_created_objects:
                        if local_obj_check.inst_token == obj_data["inst_token"]:
                            local_obj = local_obj_check
                            break

                    if local_obj is not None:
                        # Use the local object (already on correct device and fully textured)
                        all_objects.append(local_obj)
                    else:
                        # Reconstruct object from gathered data
                        if "female" in obj_data["prompt"]:
                            human_model = SMPL(
                                "smpl/SMPL_FEMALE.pkl",
                                betas=(torch.tensor(obj_data["betas"])),
                                device=device,
                            )
                        else:
                            human_model = SMPL(
                                "smpl/SMPL_MALE.pkl",
                                betas=(torch.tensor(obj_data["betas"])),
                                device=device,
                            )

                        human_model.colors = torch.tensor(obj_data["colors"]).to(device)
                        human_model.vertices = np.array(obj_data["vertices"])
                        human_model.rest_vertices = np.array(obj_data["rest_vertices"])

                        # Ensure the mesh is properly updated
                        human_model.mesh.vertices = human_model.vertices
                        colors_uint8 = (human_model.colors.cpu().numpy() * 255).astype(
                            np.uint8
                        )
                        human_model.mesh.visual.vertex_colors = colors_uint8

                        # Reconstruct Object3D
                        obj = Object3D(
                            size=torch.tensor(obj_data["size"]).to(device),
                            prompt=obj_data["prompt"],
                            human_model=human_model,
                            pose_key=obj_data["pose_key"],
                            inst_token=obj_data["inst_token"],
                            device=device,
                            initial_pose_idx=obj_data["initial_pose_idx"],
                        )
                        all_objects.append(obj)

        # print(
        #     f"Rank {rank}: Successfully generated and shared {len(all_objects)} total objects across all ranks"
        # )
        return all_objects
    else:
        # Single GPU mode - just return local objects
        print(f"Single GPU: Created {len(local_created_objects)} objects")
        return local_created_objects


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
        device=None,
    ):
        self.device = device if device is not None else get_device()
        self._size = size.clone().detach().to(self.device)
        self._size[[0, 1]] = self._size[[1, 0]]
        self._text = prompt
        self._human_model = human_model
        self._pose_key = pose_key
        self.inst_token = inst_token  # Store instance token for identification

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

        initial_orient = torch.zeros([1, 3], device=self.device)  # global_orient
        initial_orient[0] = torch.from_numpy(
            self.poses[self._pose_key][self._initial_pose_idx, :3]
        ).to(self.device)
        initial_pose = torch.zeros([1, 69], device=self.device)  # 23*3 axis-angle
        initial_pose[0, :63] = torch.from_numpy(
            self.poses[self._pose_key][self._initial_pose_idx, 3:66]
        ).to(self.device)
        self._human_model.apply_pose(
            body_pose=initial_pose, global_orient=initial_orient
        )
        self.scale = None
        inst_id = (
            inst_token[-8:] if inst_token else "unknown"
        )  # Show last 8 chars of token
        # print(
        #     f"  {prompt} (ID: {inst_id}) initialized with '{self._pose_key}' pose, starting from motion {self._initial_pose_idx}"
        # )

    def to_device(self, device):
        """Move this Object3D and its components to the specified device."""
        self.device = device
        self._size = self._size.to(device)
        # Move human model components to new device
        if hasattr(self._human_model, "device"):
            self._human_model.device = device
        if hasattr(self._human_model, "model"):
            self._human_model.model.to(device)
        if hasattr(self._human_model, "betas"):
            self._human_model.betas = self._human_model.betas.to(device)
        if hasattr(self._human_model, "body_pose"):
            self._human_model.body_pose = self._human_model.body_pose.to(device)
        if hasattr(self._human_model, "global_orient"):
            self._human_model.global_orient = self._human_model.global_orient.to(device)
        # Note: Paint pipeline is no longer part of SMPL class - it's obtained on-demand
        return self

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
            gs = self.rotate_gs(rpy2rotations(0, 0, -np.pi / 2, self.device), gs)
        elif self._pose_key == "walk":
            gs = self.rotate_gs(rpy2rotations(0, 0, 3 * np.pi / 4, self.device), gs)
        elif self._pose_key == "jog":
            gs = self.rotate_gs(rpy2rotations(0, 0, -5 * np.pi / 6, self.device), gs)
        elif self._pose_key == "run":
            gs = self.rotate_gs(rpy2rotations(0, 0, -3 * np.pi / 4, self.device), gs)
        valid_mask = gs["opacities"].squeeze() != 0
        x_min, x_max = gs["xyz"][valid_mask, 0].min(), gs["xyz"][valid_mask, 0].max()
        y_min, y_max = gs["xyz"][valid_mask, 1].min(), gs["xyz"][valid_mask, 1].max()
        z_min, z_max = (
            gs["xyz"][valid_mask, 2].min(),
            gs["xyz"][valid_mask, 2].max(),
        )
        center = torch.tensor(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        ).to(self.device)
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
        body_pose = torch.zeros([1, 69], device=self.device)  # 23*3 axis-angle
        body_pose[0, :63] = torch.from_numpy(
            self.poses[self._pose_key][pose_idx, 3:66]
        ).to(self.device)
        global_orient = torch.zeros([1, 3], device=self.device)  # global_orient
        global_orient[0] = torch.from_numpy(
            self.poses[self._pose_key][pose_idx, :3]
        ).to(self.device)
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


def render_gaussian(gaussian, extrinsics, intrinsics, width=800, height=424):
    if gaussian is None:
        return (
            torch.ones(1, height, width, 3).cuda(),
            torch.zeros(1, height, width, 1).bool().cuda(),
        )
    extrinsics = torch.tensor(extrinsics).float().cuda()
    intrinsics = torch.tensor(intrinsics).float().cuda()
    intrinsics[0] *= width / 1600
    intrinsics[1] *= height / 900
    means = gaussian["xyz"]
    rgbs = gaussian["rgbs"]
    opacities = gaussian["opacities"]
    scales = gaussian["scales"]
    rotations = gaussian["rots"]

    renders, alphas, _ = rasterization(
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
        radius_clip=0.0,
        backgrounds=torch.ones(1, 3).cuda(),
    )
    renders = torch.clamp(renders, max=1.0)
    masks = (alphas > 0).bool()
    return renders, masks


def render(
    gs, intrinsics, extrinsics, save_path="render.png", mask_save_path="mask.png"
):
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    images = []
    masks = []
    for i in range(len(cams)):
        intrinsic = intrinsics[cams[i]]
        extrinsic = extrinsics[cams[i]]
        img, mask = render_gaussian(gs, extrinsic, intrinsic)
        img = img[0].detach().cpu().numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        images.append(img)
        mask = mask[0, ..., 0].detach().cpu().numpy().astype(np.uint8) * 255
        mask = Image.fromarray(mask)
        masks.append(mask)
    # Arrange images in two rows of three using PIL
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    row_height = max(heights)
    row1_width = sum(widths[:3])
    row2_width = sum(widths[3:])
    final_width = max(row1_width, row2_width)
    final_height = row_height * 2
    final_img = Image.new('RGB', (final_width, final_height))
    final_mask = Image.new("L", (final_width, final_height))
    # Paste images directly into final_img
    x_offset = 0
    for img in images[:3]:
        final_img.paste(img, (x_offset, 0))
        final_mask.paste(masks[images.index(img)], (x_offset, 0))
        x_offset += img.size[0]
    x_offset = 0
    for img in images[3:]:
        final_img.paste(img, (x_offset, row_height))
        final_mask.paste(masks[images.index(img)], (x_offset, row_height))
        x_offset += img.size[0]
    final_img.save(save_path)
    final_mask.save(mask_save_path)


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
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel processing.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=12355,
        help="Master port for distributed training.",
    )
    return parser.parse_args()


def main_worker(rank, world_size, args):
    """Main worker function for each GPU process."""
    print(f"Worker started: rank={rank}, world_size={world_size}")
    distributed_success = False

    if world_size > 1:
        print(f"Rank {rank}: Attempting to setup distributed training...")
        try:
            setup_distributed(rank, world_size, args.master_port)
            distributed_success = True
            print(f"Rank {rank}: Successfully setup distributed training")
        except Exception as e:
            print(f"Rank {rank}: Failed to setup distributed training: {e}")
            print(f"Rank {rank}: Continuing with single-GPU mode...")
            world_size = 1  # Fall back to single GPU mode
            rank = 0
    else:
        print(f"Rank {rank}: Running in single-GPU mode (world_size={world_size})")

    seed_everything(args.seed, rank)
    print(f"GPU {rank}: Using seed {args.seed}")

    device = (
        get_device(rank) if distributed_success and world_size > 1 else get_device()
    )

    # Initialize the diffusion pipeline on the appropriate device
    openpose_pipe, paint_pipeline = initialize_pipelines(
        device, rank if distributed_success and world_size > 1 else None
    )

    # Load NuScenes data (only on rank 0 to avoid conflicts)
    if rank == 0:
        nusc = NuScenes(
            version="v1.0-trainval", dataroot="/data/nuscenes", verbose=True
        )
    else:
        nusc = NuScenes(
            version="v1.0-trainval", dataroot="/data/nuscenes", verbose=False
        )
    val_scenes = [s for s in nusc.scene if s["name"] in splits.val]
    for scene_idx, scene in enumerate(val_scenes):
        sample = None
        objects = {}
        cams = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        # Collect new objects to create in parallel
        humans_to_create = []
        existing_humans = []
        annotations_2hz = {}
        positions = {}

        for t in range(scene["nbr_samples"]):
            sample_token = (
                scene["first_sample_token"] if sample is None else sample["next"]
            )
            sample = nusc.get("sample", sample_token)
            cam_tokens = {cam: sample["data"][cam] for cam in cams}
            cam_data = {
                cam: nusc.get("sample_data", cam_tokens[cam]) for cam in cam_tokens
            }
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
                extrinsics, cam_front_to_ego = all_to_camera_front(
                    nusc, cam_calib_tokens
                )
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
                    if ann["category_name"] == "human.pedestrian.police_officer":
                        prompt = "A real-world police officer in uniform"
                    elif ann["category_name"] == "human.pedestrian.construction_worker":
                        prompt = "A real-world construction worker in uniform"
                    else:
                        prompt = np.random.choice(
                            [
                                "A female real-world pedestrian",
                                "A male real-world pedestrian",
                            ]
                        )
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

        # Create humans using distributed processing
        effective_world_size = world_size if distributed_success else 1
        created_humans = create_humans_batched(
            humans_to_create,
            openpose_pipe,
            paint_pipeline,
            16,
            rank if distributed_success else 0,
            effective_world_size,
        )

        if rank == 0:
            print(f"Created {len(created_humans)} human objects for scene {scene_idx}")

        # Store created objects by their instance tokens
        # Each rank only keeps its local objects - no more placeholder objects
        for obj in created_humans:
            # Move object to current device if needed
            current_device = (
                get_device(rank)
                if distributed_success and world_size > 1
                else get_device()
            )
            if obj.device != current_device:
                obj.to_device(current_device)
            objects[obj.inst_token] = obj

        # print(f"Rank {rank}: Ready to render with {len(objects)} objects")

        # Synchronize all ranks before starting distributed rendering
        if distributed_success and world_size > 1 and dist.is_initialized():
            # print(
            #     f"Rank {rank}: All ranks synchronized and ready for distributed rendering"
            # )
            # Specify device_ids to avoid NCCL warning and potential hang
            device_id = torch.cuda.current_device()
            dist.barrier(device_ids=[device_id])

        # Now ALL ranks participate in distributed frame rendering
        # Each rank renders its assigned subset of frames
        total_frames = scene["nbr_samples"] * interpolation_factor

        # Distribute frames across all GPUs for parallel rendering
        frames_per_gpu_base = (
            total_frames // world_size
            if distributed_success and world_size > 1
            else total_frames
        )
        extra_frames = (
            total_frames % world_size if distributed_success and world_size > 1 else 0
        )

        if distributed_success and world_size > 1:
            # Calculate frame range for this rank
            if rank < extra_frames:
                start_frame = rank * (frames_per_gpu_base + 1)
                end_frame = start_frame + frames_per_gpu_base + 1
            else:
                start_frame = (
                    extra_frames * (frames_per_gpu_base + 1)
                    + (rank - extra_frames) * frames_per_gpu_base
                )
                end_frame = start_frame + frames_per_gpu_base

            # print(
            #     f"Rank {rank}: Rendering frames {start_frame}-{end_frame-1} ({end_frame - start_frame} frames)"
            # )
        else:
            start_frame = 0
            end_frame = total_frames
            print(f"Single GPU: Rendering all {total_frames} frames")

        # Create output directory
        os.makedirs(f"val_videos/{scene_idx}/rendered_images", exist_ok=True)
        os.makedirs(f"val_videos/{scene_idx}/rendered_masks", exist_ok=True)

        # Render assigned frames
        # Render assigned frames - each rank renders its subset
        for t in trange(start_frame, end_frame, desc=f"Rank {rank} rendering"):
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
            image_path = f"val_videos/{scene_idx}/rendered_images/frame_{t:04d}.png"
            mask_path = f"val_videos/{scene_idx}/rendered_masks/frame_{t:04d}.png"
            render(
                gs,
                intrinsics,
                extrinsics,
                save_path=image_path,
                mask_save_path=mask_path,
            )

        # Synchronize all processes after rendering
        if distributed_success and world_size > 1 and dist.is_initialized():
            # Specify device_ids to avoid NCCL warning and potential hang
            device_id = torch.cuda.current_device()
            dist.barrier(device_ids=[device_id])

        # Only rank 0 assembles the final video from all frames
        if rank == 0:
            print(f"Assembling video from all {total_frames} frames...")
            frame_paths = sorted(
                glob.glob(f"val_videos/{scene_idx}/rendered_images/frame_*.png")
            )
            mask_paths = sorted(
                glob.glob(f"val_videos/{scene_idx}/rendered_masks/frame_*.png")
            )

            # Check that all frames were rendered
            if len(frame_paths) != total_frames:
                print(
                    f"Warning: Expected {total_frames} frames but found {len(frame_paths)}"
                )
            if len(mask_paths) != total_frames:
                print(
                    f"Warning: Expected {total_frames} masks but found {len(mask_paths)}"
                )

            with imageio.get_writer(
                f"val_videos/{scene_idx}/rgbs.mp4",
                mode="I",
                fps=args.hz,
                format="FFMPEG",
            ) as video_writer:
                for frame_path in frame_paths:
                    frame = imageio.imread(frame_path)
                    video_writer.append_data(frame)

            # Clean up the frame images and remove the folder
            for frame_path in frame_paths:
                os.remove(frame_path)
            os.rmdir(f"val_videos/{scene_idx}/rendered_images")
            print(f"✓ Video saved: val_videos/{scene_idx}/rgbs.mp4")
            with imageio.get_writer(
                f"val_videos/{scene_idx}/masks.mp4",
                mode="I",
                fps=args.hz,
                format="FFMPEG",
            ) as mask_writer:
                for mask_path in mask_paths:
                    mask = imageio.imread(mask_path)
                    mask_writer.append_data(mask)
            # Clean up the mask images and remove the folder
            for mask_path in mask_paths:
                os.remove(mask_path)
            os.rmdir(f"val_videos/{scene_idx}/rendered_masks")
            print(f"✓ Mask video saved: val_videos/{scene_idx}/masks.mp4")

        # Synchronize all processes after video assembly
        if distributed_success and world_size > 1 and dist.is_initialized():
            # Specify device_ids to avoid NCCL warning and potential hang
            device_id = torch.cuda.current_device()
            dist.barrier(device_ids=[device_id])

    # Clean up distributed processing after all scenes are processed
    if distributed_success and world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main entry point that handles single or multi-GPU execution."""
    args = parse_args()

    print(f"Requested GPUs: {args.num_gpus}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check if multi-GPU is requested and available
    if args.num_gpus > 1 and torch.cuda.device_count() >= args.num_gpus:
        print(f"Starting multi-GPU processing with {args.num_gpus} GPUs...")
        try:
            mp.spawn(
                main_worker, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True
            )
        except Exception as e:
            print(f"Multi-GPU processing failed: {e}")
            print("Falling back to single-GPU processing...")
            main_worker(0, 1, args)
    else:
        if args.num_gpus > 1:
            print(
                f"Warning: Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available."
            )
            print("Falling back to single-GPU processing...")
        else:
            print("Using single-GPU processing...")
        main_worker(0, 1, args)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
