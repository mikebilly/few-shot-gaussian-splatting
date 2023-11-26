from depth_regularization.proj_utils import create_depth_maps_zoe, project_points_to_cameras, find_optimal_offset_scale
import numpy as np
import os
from PIL import Image

def preprocess_depth(dataset_path):

    depth_maps = create_depth_maps_zoe(dataset_path, save=True)

    cam_intrinsics, cam_extrinsics, projected_points, transformed_points, rgb, error = project_points_to_cameras(dataset_path)

    extrinsics = {}
    for cam in cam_extrinsics.values():
        extrinsics[cam.name] = cam

    intrinsic = cam_intrinsics[1]

    reliability = 1/error

    min = np.percentile(reliability, 0)
    max = np.percentile(reliability, 80)

    weight = (reliability - min) / (max - min)
    weight = np.clip(weight, 0, 1).reshape(-1)

    depth_adjusted = find_optimal_offset_scale(weight, extrinsics, depth_maps, 
                                projected_points, transformed_points, intrinsic)

    os.makedirs(os.path.join(dataset_path, "depth_adjusted"), exist_ok=True)

    for img_name in depth_adjusted:

        print("Saving adjusted depth map for: ", img_name)
        
        range = 200
        
        depth_map = depth_adjusted[img_name] * 2**16 / range
        depth_map = np.clip(depth_map, 0, 2**16 - 1)
        depth_map = depth_map.astype(np.uint16)

        depth_map_image = Image.fromarray(depth_map)
        
        depth_map_image.save(os.path.join(dataset_path, "depth_adjusted", img_name.split(".")[0] + ".png"))

if __name__ == "__main__":
    dataset_path = "dataset"
    preprocess_depth(dataset_path)