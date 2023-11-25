import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
import numpy as np
from PIL import Image
from scene.colmap_loader import read_points3D_binary, read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

def create_depth_maps(dataset_path, save=False):
    repo = "isl-org/ZoeDepth"
    # Zoe_N
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True).to("cuda")
    depth_maps = {}

    os.makedirs(os.path.join(dataset_path,"depth"), exist_ok=True)

    for image in os.listdir(os.path.join(dataset_path,"images")):
        # if it is not an image, skip
        if not (image.lower().endswith(".jpg") or image.endswith(".png")):
            continue
        img = Image.open(os.path.join(dataset_path, "images", image))
        depth = model_zoe_n.infer_pil(img)
        depth_maps[image] = depth

        print("Created depth map for: ", image.split(".")[0])
        
        if (save):
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            depth = (depth_norm).astype(np.uint8)
            depth = Image.fromarray(depth)
            depth.save(os.path.join(dataset_path, "depth", image.split(".")[0]+".png"))
            
    
    return depth_maps

def create_extrinsic_matrix(extrinsic):
    """
    Create a 4x4 extrinsic matrix from a quaternion and translation vector.
    """
    rotation_matrix = qvec2rotmat(extrinsic.qvec)
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = extrinsic.tvec
    return extrinsic_matrix

def create_intrinsic_matrix(intrinsic):

    intrinsic_matrix = np.eye(4)
    intrinsic_matrix[0, 0] = intrinsic.params[0]
    intrinsic_matrix[1, 1] = intrinsic.params[1]
    intrinsic_matrix[0, 2] = intrinsic.params[2]
    intrinsic_matrix[1, 2] = intrinsic.params[3]
    
    return intrinsic_matrix

def project_points(points, intrinsics):
    # Assuming points is an Nx4 numpy array and intrinsics is a 4x4 matrix
    projected_points = intrinsics @ points.T
    projected_points /= projected_points[2, :]
    return projected_points.T

def project_points_to_cameras(dataset_path):
    
    # Point cloud
    points3D_path = os.path.join(dataset_path,"sparse","0", "points3D.bin")
    xyz, rgb, errors = read_points3D_binary(points3D_path)

    # Cameras
    cameras_intrinsic_file = os.path.join(dataset_path, "sparse/0", "cameras.bin")
    cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.bin")

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    intrinsic = cam_intrinsics[1]
    intrinsic_matrix = create_intrinsic_matrix(intrinsic)

    projected_points = {}
    transformed_points = {}

    for extrinsic in cam_extrinsics.values():

        points_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        extrinsic_matrix = create_extrinsic_matrix(extrinsic)
        transformed = extrinsic_matrix @ points_homogeneous.T
        transformed = transformed.T

        transformed_points[extrinsic.id] = transformed
        
        projected = project_points(transformed, intrinsic_matrix)

        projected_points[extrinsic.id] = projected

        print("Projected points for camera: ", extrinsic.id)
    
    return cam_intrinsics, cam_extrinsics, projected_points, transformed_points, rgb, errors


def depth_error(x, weight, transformed_points, depth_map_points):
    scale, offset = x
    return (weight *(transformed_points[:, 2] - (depth_map_points * scale + offset))**2).mean()

def find_optimal_offset_scale(weight, extrinsics, depth_maps, 
                              projected_points, transformed_points, intrinsic, 
                              samples=150, ranges=[(0.5, 15), (0, 5)], discard_extreme=True):
    
    adjusted_depth_maps = {}

    for img_name in depth_maps:

        id = extrinsics[img_name].id
        print(id)
        projected = projected_points[id][:,:2]
        transformed_points_id = transformed_points[id]

        # Scale projected points to depth map size
        scale_ratio = intrinsic.width / depth_maps[img_name].shape[1]
        projected = projected / scale_ratio

        index = np.zeros((projected.shape[0], 2), dtype=np.int32)
        index[:, 1] = np.int32(np.clip(projected[:, 0], 0, intrinsic.width - 1))
        index[:, 0] = np.int32(intrinsic.height - 1 - np.clip(projected[:, 1], 0, intrinsic.height - 1))

        depth_map_points = depth_maps[img_name][index[:, 0], index[:, 1]]

        # Discard extreme values
        if discard_extreme:
            diff = transformed_points_id[:, 2] - depth_map_points
            high_thres = np.percentile(diff, 90)
            transformed_points_id = transformed_points_id[diff < high_thres]
            depth_map_points = depth_map_points[diff < high_thres]
            weight_new = weight[diff < high_thres]
        else:
            weight_new = weight

        scale = np.linspace(ranges[0][0], ranges[0][1], samples)
        offset = np.linspace(ranges[1][0], ranges[1][1], samples)

        scale_m, offset_m = np.meshgrid(scale, offset)

        Z = np.zeros((samples, samples))

        for i in range(samples):
            for j in range(samples):
                Z[i, j] = depth_error([scale_m[i, j], offset_m[i, j]], weight_new, transformed_points_id, depth_map_points)
        
        min_index = np.argmin(Z)
        
        i_min, j_min = np.unravel_index(min_index, Z.shape)

        # Get the corresponding scale and offset values
        optimal_scale = scale_m[i_min, j_min]
        optimal_offset = offset_m[i_min, j_min]

        diff = transformed_points_id[:, 2] - (depth_map_points * optimal_scale + optimal_offset)
        
        print(f"Optimal scale {optimal_scale} and offset {optimal_offset} for image {img_name}")
        print(f"Got an error of {Z.min()}, average difference between the images: {diff.mean()}")

        adjusted_depth_maps[img_name] = depth_maps[img_name] * optimal_scale + optimal_offset


    return adjusted_depth_maps

def get_smoothness_loss(depth_map):
    
    tensor = False
    if isinstance(depth_map, torch.Tensor):
        tensor = True
        depth_map = depth_map.detach().cpu().numpy().squeeze()
    
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    edges = cv2.Canny((depth_norm*255).astype(np.uint8), 10, 50)
    mask = 255 - cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    dm_down = np.roll(depth_map, 1, axis=0)
    dm_up = np.roll(depth_map, -1, axis=0)
    dm_right = np.roll(depth_map, 1, axis=1)
    dm_left = np.roll(depth_map, -1, axis=1)

    diff =  np.square(depth_map - dm_down)
    diff += np.square(depth_map - dm_up)
    diff += np.square(depth_map - dm_right)
    diff += np.square(depth_map - dm_left)

    smoothness = diff * mask / 255

    if (tensor):
        return torch.from_numpy(smoothness).cuda().mean()
    else:
        return smoothness.mean()

if __name__ == "__main__":

    dataset_path = "dataset"
    #create_depth_maps(dataset_path, save=True)
    cam_intrinsics, cam_extrinsics, projected_points, transformed_points, rgb, errors = project_points_to_cameras(dataset_path)
    intrinsic = cam_intrinsics[1]

    # Visualize
    for extrinsic, projected, color in zip(cam_extrinsics.values(), projected_points.values(), rgb):
        img = cv2.imread(os.path.join(dataset_path, "images", extrinsic.name))
        scale_ratio = intrinsic.width / img.shape[1]
        projected = projected / scale_ratio

        img = np.zeros_like(img)

        for point, color in zip(projected, rgb):
            x, y = point[0], point[1]
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (int(x), int(y)), 2, (color[0], color[1],  color[2]), -1)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dataset_path, "projected_debug", extrinsic.name), img)