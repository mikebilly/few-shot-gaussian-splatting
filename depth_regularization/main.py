from proj_utils import create_depth_maps, project_points_to_cameras, find_optimal_offset_scale
import cv2
import numpy as np
import os

dataset_path = "dataset"
depth_maps = create_depth_maps(dataset_path)

cam_intrinsics, cam_extrinsics, projected_points, transformed_points, rgb, error = project_points_to_cameras(dataset_path)

extrinsics = {}
for cam in cam_extrinsics.values():
    extrinsics[cam.name] = cam

intrinsic = cam_intrinsics[1]

reliability = 1/error

min = np.percentile(reliability, 5)
max = np.percentile(reliability, 95)

weight = (reliability - min) / (max - min)
weight = np.clip(weight, 0, 1).reshape(-1)

depth_adjusted = find_optimal_offset_scale(weight, extrinsics, depth_maps, 
                              projected_points, transformed_points, intrinsic)