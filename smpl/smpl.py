import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import smplx
import math
import copy
import cv2
import trimesh
from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.rembg import BackgroundRemover
import warnings
import sys
import os
import logging
import imageio.v2 as imageio
from gsplat.rendering import rasterization
from smpl.cam_utils import get_single_cam, get_default_cam_sequences
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# -----------------------------------------------------
# Helpers: normals, per-vertex & per-face Gaussians
# -----------------------------------------------------
def compute_vertex_normals(vertices_np, faces_np):
    v = torch.tensor(vertices_np, dtype=torch.float32)
    f = torch.tensor(faces_np, dtype=torch.long)
    normals = torch.zeros_like(v)
    tris = v[f]  # (F,3,3)
    tri_normals = torch.linalg.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for i in range(3):
        normals.index_add_(0, f[:, i], tri_normals)
    normals = torch.nn.functional.normalize(normals, dim=1).cpu().numpy()
    return normals

def build_vertex_gaussians(vertices_np, faces_np, min_scale=0.002, max_scale=0.05, device="cuda"):
    V = vertices_np.shape[0]

    # --- avg edge lengths (vectorized) ---
    edges = np.concatenate(
        [faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]], axis=0
    )
    edge_vecs = vertices_np[edges[:, 0]] - vertices_np[edges[:, 1]]
    edge_lens = np.linalg.norm(edge_vecs, axis=1)
    sum_len = np.bincount(edges[:, 0], weights=edge_lens, minlength=V) + np.bincount(
        edges[:, 1], weights=edge_lens, minlength=V
    )
    count_len = np.bincount(edges[:, 0], minlength=V) + np.bincount(
        edges[:, 1], minlength=V
    )
    avg_len = sum_len / (count_len + 1e-9)
    avg_len = avg_len / 2.0
    avg_len[count_len == 0] = 0.005

    # --- scales ---
    v_scales = np.stack([avg_len, avg_len, avg_len * 0.2], axis=1).astype(np.float32)
    v_scales = np.clip(v_scales, min_scale, max_scale)

    # --- normals ---
    normals = compute_vertex_normals(vertices_np, faces_np)

    # --- rotation frames ---
    z = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    tmp = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (V, 1))
    mask = np.abs((z * tmp).sum(axis=1)) > 0.9
    tmp[mask] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y = np.cross(z, tmp)
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-9
    x = np.cross(y, z)
    Rmat = np.stack([x, y, z], axis=2)
    quat_xyzw = R.from_matrix(Rmat).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return (
        torch.tensor(vertices_np.astype(np.float32), device=device),
        torch.tensor(v_scales, device=device),
        torch.tensor(quat_wxyz, device=device),
    )

SMPL_TO_BODY_MAPPING = [
    24,  # Nose
    12,  # Neck
    16,  # RShoulder
    18,  # RElbow
    20,  # RWrist
    17,  # LShoulder
    19,  # LElbow
    21,  # LWrist
    # 0, # Pelvis
    1,  # RHip
    4,  # RKnee
    7,  # RAnkle
    2,  # LHip
    5,  # LKnee
    8,  # LAnkle
    26,  # LEye
    25,  # REye
    28,  # LEar
    27,  # REar
]

# OpenPose Body connection pairs for skeleton drawing
BODY_PAIRS = [
    [2, 3],
    [2, 6],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 9],
    [9, 10],
    [10, 11],
    [2, 12],
    [12, 13],
    [13, 14],
    [2, 1],
    [1, 15],
    [15, 17],
    [1, 16],
    [16, 18],
]
BODY_COLORS = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]


def convert_smpl_to_body(smpl_joints):
    """
    Convert SMPL joints (45,3) to OpenPose Body format (18,3)

    Args:
        smpl_joints: numpy array of shape (45, 3) - SMPL joint positions

    Returns:
        body_joints: numpy array of shape (18, 3) - Body joint positions
    """
    # Initialize Body joints with NaN (missing joints)
    body_joints = smpl_joints[SMPL_TO_BODY_MAPPING]
    return body_joints


def determine_valid_joint_based_on_ndc(joints_ndc):
    """
    Determine self-occluded joints based on NDC z-values heuristically.
    Joints are indexed as:
    [0]Nose, [1]Neck, [2]RShoulder, [3]RElbow, [4]RWrist,
    [5]LShoulder, [6]LElbow, [7]LWrist,
    [8]RHip, [9]RKnee, [10]RAnkle,
    [11]LHip, [12]LKnee, [13]LAnkle,
    [14]LEye, [15]REye, [16]LEar, [17]REar
    """
    visible = np.ones(len(joints_ndc), dtype=bool)
    z = joints_ndc[:, 2]

    # Reference points
    neck_z = z[1]
    nose_z = z[0]

    # --- Face ---
    eye_mean_z = 1 / 2 * (z[14] + z[15])
    if eye_mean_z < nose_z:
        visible[14] = False  # LEye
        visible[15] = False  # REye
        visible[0] = False
    if z[16] < neck_z and neck_z < z[17]:
        visible[17] = False  # REar
    elif z[17] < neck_z and neck_z < z[16]:
        visible[16] = False  # LEar

    # --- Arms ---
    if z[2] < neck_z and neck_z < z[5]:
        if z[3] < neck_z and neck_z < z[6]:
            if z[4] < z[7]:
                visible[5] = False
                visible[6] = False
                visible[7] = False
    elif z[5] < neck_z and neck_z < z[2]:
        if z[6] < neck_z and neck_z < z[3]:
            if z[7] < z[4]:
                visible[2] = False
                visible[3] = False
                visible[4] = False

    return visible


def smpl_to_openpose(
    smpl_joints,
    full_proj_transform,
    image_width,
    image_height,
    draw_skeleton=True,
):
    """
    Convert SMPL joints to OpenPose v2 Body format with size-dependent thickness.
    """

    # Convert inputs to numpy
    if isinstance(smpl_joints, torch.Tensor):
        smpl_joints = smpl_joints.detach().cpu().numpy()
    if isinstance(full_proj_transform, torch.Tensor):
        full_proj_transform = full_proj_transform.detach().cpu().numpy()
    full_proj_transform = full_proj_transform.squeeze()

    # SMPL -> Body
    body_3d = convert_smpl_to_body(smpl_joints)

    # Project to 2D
    joints_h = np.concatenate([body_3d, np.ones((len(body_3d), 1))], axis=1)
    proj = joints_h @ full_proj_transform
    w_coords = np.where(np.abs(proj[:, 3:4]) < 1e-8, 1e-8, proj[:, 3:4])
    joints_ndc = proj[:, :3] / w_coords
    x_pix = (joints_ndc[:, 0] * 0.5 + 0.5) * image_width
    y_pix = (joints_ndc[:, 1] * 0.5 + 0.5) * image_height

    x_valid = np.logical_and(x_pix >= 0, x_pix < image_width)
    y_valid = np.logical_and(y_pix >= 0, y_pix < image_height)
    visible = np.logical_and(x_valid, y_valid)
    valid = visible  # np.logical_and(visible, determine_valid_joint_based_on_ndc(joints_ndc))
    # Full Body 2D coords
    body_2d = np.stack([x_pix, y_pix], axis=1)

    # Create image
    img = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Draw joints
    for i, color in enumerate(BODY_COLORS):
        if valid[i]:
            pt = body_2d[i].astype(int)
            cv2.circle(img, tuple(pt), 4, color, -1)
            # cv2.putText(
            #     img,
            #     str(SMPL_TO_BODY_MAPPING[i]),
            #     (pt[0] + 6, pt[1] - 6),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     color,
            #     1,
            #     cv2.LINE_AA,
            # )
    if draw_skeleton:
        for (i, j), color in zip(BODY_PAIRS, BODY_COLORS):
            pt1 = body_2d[i - 1].astype(int)
            pt2 = body_2d[j - 1].astype(int)
            if valid[i - 1] and valid[j - 1]:
                mY = np.mean([pt1[0], pt2[0]])
                mX = np.mean([pt1[1], pt2[1]])
                length = ((pt1 - pt2) ** 2).sum() ** 0.5
                angle = math.degrees(math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1
                )
                cv2.fillConvexPoly(img, polygon, [int(float(c) * 0.6) for c in color])

    return img, body_2d

def get_uv_color_from_texture(new_mesh, points):
    tex_image = np.array(new_mesh.visual.material.image)
    H, W = tex_image.shape[:2]
    uv = new_mesh.visual.uv
    faces = new_mesh.faces

    # nearest points & faces
    result = new_mesh.nearest.on_surface(points)
    closest_points = result[0]
    face_indices = result[2]

    def barycentric(p, tri):
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]
        v2 = p - tri[0]
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00*d11 - d01*d01
        v = (d11*d20 - d01*d21)/denom
        w = (d00*d21 - d01*d20)/denom
        u = 1 - v - w
        return u, v, w
    colors = []
    for i, p in enumerate(points):
        f_idx = face_indices[i]
        tri_vert_indices = faces[f_idx]
        tri_vertices = new_mesh.vertices[tri_vert_indices]

        # compute barycentric coordinates
        u0, v0, w0 = barycentric(closest_points[i], tri_vertices)
        tri_uvs = uv[tri_vert_indices]
        uv_p = u0*tri_uvs[0] + v0*tri_uvs[1] + w0*tri_uvs[2]

        # convert UV to pixel coords
        px = int(uv_p[0]*(W-1))
        py = int((1-uv_p[1])*(H-1))
        color = tex_image[py, px]
        colors.append(color)
    return np.array(colors)


class SMPL:

    def __init__(
        self,
        model_path,
        betas=None,
        device="cuda",
    ):
        self.device = device

        self.model = smplx.create(
            model_path,
            model_type="smpl",
            batch_size=1,
        ).to(device)
        self.device = device
        self.betas = (
            torch.zeros([1, 10], device=device) if betas is None else betas.to(device)
        )
        self.body_pose = torch.zeros([1, 69], device=device)  # 23*3 axis-angle
        with torch.no_grad():
            out = self.model(
                betas=self.betas,
                body_pose=self.body_pose,
                return_verts=True,
            )
        # better for Hunyuan3DPaint
        self.rest_vertices = out.vertices[0].detach().cpu().numpy()  # (6890, 3)

        # Store global orientation as instance variable
        self.global_orient = torch.tensor(
            [[math.pi / 2, 0.0, 0.0]], device=device
        )  # stand up

        out = self.model(
            betas=self.betas,
            body_pose=self.body_pose,
            global_orient=self.global_orient,
            return_verts=True,
            return_joints=True,
        )
        self.vertices = out.vertices[0].detach().cpu().numpy()  # (V,3)
        self.joints = out.joints[0].detach()
        self.faces = self.model.faces.astype(np.int64)  # (13776, 3)
        self.mesh = trimesh.Trimesh(
            vertices=self.rest_vertices, faces=self.faces
        )
        self.build_initial_vertex_gaussians()

    def get_openpose_img(self, azimuth=0, img_size=512):
        cam = get_single_cam(azimuth=azimuth, img_size=img_size)
        openpose_img, _ = smpl_to_openpose(
            self.joints,
            cam["full_proj_transform"].squeeze(),
            image_width=int(cam["image_width"]),
            image_height=int(cam["image_height"]),
        )
        return openpose_img

    def paint_mesh(self, ref_image: Image.Image, paint_pipeline):
        """Paint the mesh with a reference image using the provided paint pipeline.

        Args:
            ref_image: Reference image to use for texturing
            paint_pipeline: Hunyuan3D paint pipeline instance. If None, will get one for current device.
        """
        if ref_image.mode == 'RGB':
            rembg = BackgroundRemover()
            ref_image = rembg(ref_image)

        old_mesh = copy.deepcopy(self.mesh)

        # Suppress all output (warnings, logging, prints)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress logging
            logging.disable(logging.CRITICAL)
            # Redirect stdout and stderr to devnull
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    new_mesh = paint_pipeline(old_mesh, ref_image)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    logging.disable(logging.NOTSET)  # Re-enable logging

        colors = get_uv_color_from_texture(new_mesh, self.rest_vertices)
        self.mesh.visual.vertex_colors = colors.astype(np.uint8)
        self.colors = torch.from_numpy(colors / 255).to(self.device)

    def build_initial_vertex_gaussians(self):
        means, scales, quats = build_vertex_gaussians(
            self.vertices, self.faces, device=self.device
        )
        self.means = means
        self.opacities = torch.ones_like(self.means[:, :1])
        self.scales = scales
        self.quats = quats

    def apply_pose(
        self, body_pose=None, global_orient=None
    ):
        """
        Pose by linearly blending joint rotation matrices and translations, then transform
        per-vertex Gaussians (means + covariances).
        Args:
            body_pose: (1,69) axis-angle
            global_orient: (1,3)
            transl: (1,3)
        """
        self.body_pose = body_pose if body_pose is not None else self.body_pose
        self.global_orient = (
            global_orient if global_orient is not None else self.global_orient
        )
        with torch.no_grad():
            out = self.model(
                betas=self.betas,
                body_pose=self.body_pose,
                global_orient=self.global_orient,
                return_verts=True,
                return_joints=True,
            )

        # posed vertex positions (global)
        self.vertices = out.vertices[0].detach().cpu().numpy()  # (V,3)
        self.joints = out.joints[0].detach()
        self.means, self.scales, self.quats = build_vertex_gaussians(
            self.vertices, self.faces, device=self.device
        )
        self.mesh.vertices = self.vertices

    def get_gs(self):
        return {
            "xyz": self.means.float(),
            "scales": self.scales.float(),
            "rots": self.quats.float(),
            "rgbs": self.colors.float(),
            "opacities": self.opacities.float(),
        }

    def show(self):
        self.mesh.show()

    def render_gsplat(self, cam_info, background=None):
        """
        Render scene of splats using gsplat rasterization.
        
        Args:
            cam_info: Camera information dict containing transformation matrices
            width: Image width
            height: Image height  
            background: Background color tensor (3,), defaults to white
            
        Returns:
            Rendered image tensor (height, width, 3)
        """
        if background is None:
            background = torch.ones(3, device=self.device)
        else:
            background = background.to(self.device)

        width = int(cam_info["image_width"])
        height = int(cam_info["image_height"])

        # Get Gaussian splatting parameters
        gs = self.get_gs()
        means = gs["xyz"]
        rgbs = gs["rgbs"]
        opacities = gs["opacities"]
        scales = gs["scales"]
        rotations = gs["rots"]

        # Get camera matrices from cam_info
        c2w = torch.from_numpy(cam_info["c2w"]).to(self.device)
        w2c = torch.linalg.inv(c2w)  # world-to-camera

        # Convert to view matrix (gsplat expects camera-to-world, we have world-to-camera)
        viewmat = w2c.unsqueeze(0)  # [1, 4, 4]

        # Create intrinsic matrix from projection matrix
        # For orthographic projection used in SMPL, we need to construct K matrix
        fovx = cam_info.get("FoVx", 1.0)
        fovy = cam_info.get("FoVy", 1.0)
        focal_x = width / (2 * torch.tan(torch.tensor(fovx / 2)))
        focal_y = height / (2 * torch.tan(torch.tensor(fovy / 2)))
        K = torch.tensor([
            [focal_x, 0, width / 2],
            [0, focal_y, height / 2], 
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, 3, 3]
        # Render using gsplat
        renders, _, _ = rasterization(
            means=means,
            quats=rotations,
            scales=scales,
            opacities=opacities.squeeze(),
            colors=rgbs,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="classic",
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
            radius_clip=0.0,
            backgrounds=background.unsqueeze(0),
        )

        # Clamp and return the rendered image
        rendered_image = torch.clamp(renders[0], 0.0, 1.0)
        return rendered_image

    def orbit_video(self, out_path="smpl_orbit.mp4", fps=30, img_size=512):
        cam_sequences = get_default_cam_sequences(img_size=img_size)
        frames = []
        for i, cam_info in enumerate(cam_sequences):
            # Render current frame using gsplat
            rendered_img = self.render_gsplat(cam_info)
            # Convert to numpy array for video writing
            frame = (rendered_img.cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)

        # Save video using imageio
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Orbit video saved to {out_path}")

    def visualize(self, poses_file, out_path="smpl_visualization.mp4", fps=120, img_size=512):
        """
        Visualize SMPL model with different poses.
        
        Args:
            poses_file: Path to .npz file containing 'poses' array of shape (N, 66)
        """
        cam = get_single_cam(azimuth=0, img_size=img_size)
        frames = []
        poses = np.load(poses_file)["poses"][:, :66]
        for i, pose in enumerate(poses):
            body_pose = torch.zeros([1, 69], device=self.device)
            body_pose[0, :63] = torch.from_numpy(pose[3:66]).to(self.device)
            global_orient = torch.zeros([1, 3], device=self.device)
            global_orient[0] = torch.from_numpy(pose[:3]).to(self.device)
            self.apply_pose(body_pose=body_pose, global_orient=global_orient)
            rendered_img = self.render_gsplat(cam)
            frame = (rendered_img.cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Visualization video saved to {out_path}")


if __name__ == "__main__":
    print("Testing SMPL with gsplat rendering...")

    # Initialize SMPL model
    smpl = SMPL("smpl/SMPL_NEUTRAL.pkl", betas=torch.rand(1, 10))
    openpose_img = smpl.get_openpose_img(azimuth=180)
    Image.fromarray(openpose_img).save("smpl_openpose_test.png")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
    )

    openpose_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    openpose_pipeline.scheduler = UniPCMultistepScheduler.from_config(
        openpose_pipeline.scheduler.config
    )
    openpose_pipeline.to("cuda")
    ref_image = openpose_pipeline(
        prompt="A male real-world pedestrian, facing the camera, high quality, detailed",
        negative_prompt="cartoon, blurry, drawing, sketch, toy-like",
        num_inference_steps=20,
        image=Image.fromarray(openpose_img),
    ).images[0]
    ref_image.save("smpl_ref_image.png")

    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0-turbo"
    )
    smpl.paint_mesh(ref_image, paint_pipeline)

    smpl.orbit_video("smpl_orbit_test.mp4", fps=30)
    smpl.visualize("smpl/Form 1_poses.npz", out_path="smpl_visualization_test.mp4", fps=120)
