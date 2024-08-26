import os
import numpy as np
import open3d as o3d
import json
import cv2
from tqdm import tqdm
import pandas as pd
import scipy.io as scio
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import open3d
from glob import glob
from scipy.io import loadmat


def distort_coordinates(coords, camera_matrix, dist_coeffs):
    """
    Apply distortion to a grid of coordinates.
    coords should be an array of shape (N, 1, 2) and of type np.float32.
    """
    k1, k2, p1, p2, k3 = dist_coeffs
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

    x = (coords[:, :, 0] - cx) / fx
    y = (coords[:, :, 1] - cy) / fy

    r = np.sqrt(x**2 + y**2)

    x_distorted = x * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6) + 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_distorted = y * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6) + p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y

    x_distorted = x_distorted * fx + cx
    y_distorted = y_distorted * fy + cy

    distorted_coords = np.concatenate((x_distorted[..., np.newaxis], y_distorted[..., np.newaxis]), axis=-1)

    return np.round(distorted_coords)

def align_depth2rgb(rgb_file, depth_file, rgb_camera_matrix, rgb_distortion_coeffs, \
                    depth_camera_matrix, depth_distortion_coeffs, rotation_matrix, translation_vector):
    
    # Capture depth images
    rgb_image = cv2.imread(rgb_file)
    depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    h, w = depth_image.shape[:2]
    
    # Undistort the depth image
    depth_img_undistorted  = cv2.undistort(depth_image, depth_camera_matrix, depth_distortion_coeffs[0])

    # Create a meshgrid of pixel indices
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)

    # Transform pixel indices to normalized depth camera coordinates
    X = (xv.flatten() - depth_camera_matrix[0, 2]) / depth_camera_matrix[0, 0]
    Y = (yv.flatten() - depth_camera_matrix[1, 2]) / depth_camera_matrix[1, 1]
    Z = depth_img_undistorted * 0.001  # Assume depth is in millimeters, convert to meters
    points = np.stack([X, Y, np.ones_like(X)], axis=-1) * Z.flatten()[:,np.newaxis]

    # Transform the point cloud to the RGB camera's coordinate system
    points_transformed = (rotation_matrix @ points.T + translation_vector).T

    # Project the point cloud onto the RGB image plane
    uvs = points_transformed[:, :2] / (points_transformed[:, 2:3] + 1e-10)
    uvs[:, 0] = uvs[:, 0] * rgb_camera_matrix[0, 0] + rgb_camera_matrix[0, 2]
    uvs[:, 1] = uvs[:, 1] * rgb_camera_matrix[1, 1] + rgb_camera_matrix[1, 2]
    uvs = np.round(uvs).astype(int)

    # apply the distortion of RGB camera
    uvs = distort_coordinates(uvs[:, np.newaxis, :2], rgb_camera_matrix, rgb_distortion_coeffs[0]).reshape(-1,2).astype(int)

    # Create an aligned depth image
    aligned_depth = np.zeros_like(rgb_image[:, :, 0], dtype=np.float32)
    valid = (uvs[:,0] >= 0) & (uvs[:,0] < rgb_image.shape[1]) & (uvs[:,1] >= 0) & (uvs[:,1] < rgb_image.shape[0])
    aligned_depth[uvs[:,1][valid], uvs[:,0][valid]] = points_transformed[:, 2][valid]

    return aligned_depth

    
def main():

    root_dir = '/mnt/data/lawrence/ThermalHands/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        ego_calib_file = os.path.join(path, 'egocentric', 'egocentric_calib.json')
        # ther_calib_file = os.path.join(path, 'egocentric', 'fisheye_calib.mat')
        thermal_path = os.path.join(root_dir, clip, 'egocentric', 'thermal')
        depth_path = os.path.join(root_dir, clip, 'egocentric', 'depth')
        align_path = os.path.join(save_dir, clip, 'align_depth_t')
        thermal_files = sorted(os.listdir(thermal_path))
        depth_files = sorted(os.listdir(depth_path))
        if not os.path.exists(align_path):
            os.makedirs(align_path)
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        # ther_calib = loadmat(ther_calib_file)
        # Load the calibrated camera matrices and distortion coefficients
        # These matrices should have been obtained during camera calibration
        # rgb_camera_matrix = np.array(ego_calib['RGB_Camera_matrix'])  # Intrinsic matrix for RGB camera
        # rgb_distortion_coeffs = np.array(ego_calib['RGB_dist'])  # Distortion coefficients for RGB camera

        depth_camera_matrix = np.array(ego_calib['rgb_mtx'])  # Intrinsic matrix for depth camera
        depth_distortion_coeffs = np.array(ego_calib['rgb_dist'])  # Distortion coefficients for depth camera
        thermal_camera_matrix = np.array(ego_calib['thermal_mtx']) # Intrinsic matrix for thermal camera
        thermal_distortion_coeffs = np.array(ego_calib['thermal_dist'])  # Distortion coefficients for thermal camera
        rotation_matrix = np.array(ego_calib['thermal2rgb_rotation_matrix'])  # Extrinsic rotation matrix
        translation_vector = np.array(ego_calib['thermal2rgb_translation_vector'])  # Extrinsic translation vector

        # transform_matrix = np.eye(4)
        # transform_matrix[:3,:3] = rotation_matrix
        # transform_matrix[:3, 3] = translation_vector.T
        # transform_matrix = np.linalg.inv(transform_matrix)
        # rotation_matrix = transform_matrix[:3,:3]
        # translation_vector = transform_matrix[:3, 3].T[:,np.newaxis]
        for i in tqdm(range(len(depth_files))):
            thermal_file = os.path.join(thermal_path, thermal_files[0])
            depth_file = os.path.join(depth_path, depth_files[i])
            aligned_depth = align_depth2rgb(thermal_file, depth_file, thermal_camera_matrix, thermal_distortion_coeffs, \
                depth_camera_matrix, depth_distortion_coeffs, rotation_matrix, translation_vector)
            output_path = os.path.join(align_path, os.path.basename(depth_file).split('.')[-2] + '.png')
            plt.imshow(aligned_depth, cmap='jet', vmin=0, vmax=3)
            # plt.colorbar()
            # plt.title('Depth Camera Image [m]')
            plt.axis('off')  # Turn off axis labels
            plt.savefig(output_path)
            plt.close()
            plt.clf()


   
if __name__ == '__main__':
    main()
