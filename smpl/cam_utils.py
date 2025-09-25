# GaussianCube
import numpy as np
import torch
import math
import random
from typing import Union
from torch import Tensor
from numpy import ndarray


def dot(x: Union[Tensor, ndarray], y: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """dot product (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        y (Union[Tensor, ndarray]): y, [..., C]

    Returns:
        Union[Tensor, ndarray]: x dot y, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)
    
def length(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """length of an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: length, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))
    
def safe_normalize(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """normalize an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """

    return x / length(x, eps)

def look_at(campos, target, opengl=True):
    """construct pose rotation matrix by look-at.

    Args:
        campos (np.ndarray): camera position, float [3]
        target (np.ndarray): look at target, float [3]
        opengl (bool, optional): whether use opengl camera convention (forward direction is target --> camera). Defaults to True.

    Returns:
        np.ndarray: the camera pose rotation matrix, float [3, 3], normalized.
    """
   
    if not opengl:
        # forward is camera --> target
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # forward is target --> camera
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    """construct a camera pose matrix orbiting a target with elevation & azimuth angle.

    Args:
        elevation (float): elevation in (-90, 90), from +y to -y is (-90, 90)
        azimuth (float): azimuth in (-180, 180), from +z to +x is (0, 90)
        radius (float, optional): camera radius. Defaults to 1.
        is_degree (bool, optional): if the angles are in degree. Defaults to True.
        target (np.ndarray, optional): look at target position. Defaults to None.
        opengl (bool, optional): whether to use OpenGL camera convention. Defaults to True.

    Returns:
        np.ndarray: the camera pose matrix, float [4, 4]
    """
    
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos
    return T

def load_cam(c2w, fovx, orig_image_size=512):
    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    c2w[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    fovy = focal2fov(fov2focal(fovx, orig_image_size), orig_image_size)

    R = R
    T = T
    FoVx = fovx
    FoVy = fovy

    image_width = orig_image_size
    image_height = orig_image_size

    zfar = 100.0
    znear = 0.01

    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0

    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return {"FoVx": fovx, "FoVy": fovy, "image_width": orig_image_size, "image_height": orig_image_size, "world_view_transform": world_view_transform, "projection_matrix": projection_matrix, "full_proj_transform": full_proj_transform, "camera_center": camera_center, "c2w": c2w}


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_single_cam(azimuth=0, target=None, img_size=512):
    elevation = -20
    cam_radius = 2.0
    fovx = 1.0

    convert_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).astype(np.float32)
    cam_poses = orbit_camera(elevation, azimuth, target=target, radius=cam_radius, opengl=True)
    cam_poses = convert_mat @ cam_poses
    cam = load_cam(c2w=cam_poses, fovx=fovx, orig_image_size=img_size)
    return cam

def get_default_cam_sequences(target=None, img_size=512):
    azimuth = np.arange(-180, 180, 6, dtype=np.int32)
    cams = []
    for azi in azimuth:
        cams.append(get_single_cam(azi, target=target, img_size=img_size))
    return cams
