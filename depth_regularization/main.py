from proj_utils import create_depth_maps, project_points_to_cameras, find_optimal_offset_scale, calculate_smoothness_loss
import cv2
import numpy as np
import os
from PIL import Image
from typing import NamedTuple

dataset_path = "dataset"
depth_maps = create_depth_maps(dataset_path)

cam_intrinsics, cam_extrinsics, projected_points, transformed_points, rgb, error = project_points_to_cameras(dataset_path)

extrinsics = {}
for cam in cam_extrinsics.values():
    extrinsics[cam.name] = cam

intrinsic = cam_intrinsics[1]

reliability = 1/error

min = np.percentile(reliability, 0)
max = np.percentile(reliability, 70)

weight = (reliability - min) / (max - min)
weight = np.clip(weight, 0, 1).reshape(-1)

#weight  = np.ones_like(weight)

depth_adjusted = find_optimal_offset_scale(weight, extrinsics, depth_maps, 
                              projected_points, transformed_points, intrinsic)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    depth: np.array
    depth_map_name: str

    width: int
    height: int

for img_name in depth_adjusted:

    print("Saving adjusted depth map for: ", img_name)

    depth_map = depth_adjusted[img_name] * (2 ** 15)
    depth_map = depth_map.astype(np.uint16)

    depth_map_image = Image.fromarray(depth_map)
    
    caminfo = CameraInfo(0, np.eye(3), np.zeros(3), 0, 0, None, None, None, depth_map_image, "tula", None, None)

    depth_map_image.save(os.path.join(dataset_path, "depth", img_name.replace(".jpg", ".png")))
    