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
import open3d
from glob import glob

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
    depth_img_undistorted  = cv2.undistort(depth_image, depth_camera_matrix, depth_distortion_coeffs)

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
    points_transformed = (rotation_matrix @ points.T + translation_vector/1000).T

    # Project the point cloud onto the RGB image plane
    uvs = points_transformed[:, :2] / points_transformed[:, 2:3]
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

    ##################### Calibrate between the mocap and the VR platform #####################
    root_dir = '/mnt/data/lawrence/ThermalHands/'
    save_dir = '/mnt/data/TherHandsPro/'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(save_dir, clip)
        ego_calib_file = os.path.join(root_dir, clip, 'egocentric', 'egocentric_calib.json')

        # Load the calibrated camera matrices and distortion coefficients
        # These matrices should have been obtained during camera calibration
        rgb_camera_matrix = np.array(ego_calib['RGB Camera matrix'])  # Intrinsic matrix for RGB camera
        rgb_distortion_coeffs = np.array(ego_calib['RGB dist'])  # Distortion coefficients for RGB camera
        depth_camera_matrix = np.array(ego_calib['IR Camera matrix'])  # Intrinsic matrix for depth camera
        depth_distortion_coeffs = np.array(ego_calib['IR dist'])  # Distortion coefficients for depth camera
        rotation_matrix = np.array(ego_calib['rgb2ir_rmatrix'])  # Extrinsic rotation matrix
        translation_vector = np.array(ego_calib['rgb2ir_tvecs'])  # Extrinsic translation vector

        # read the infered egocentric marker info
        ego_marker_path = os.path.join(path, 'ego_marker_n')
        ego_marker_files = sorted(os.listdir(ego_marker_path))

        # depth files list 
        depth_path = os.path.join(root_dir, clip, 'egocentric', 'depth')
        depth_files = sorted(os.listdir(depth_path))

        # read the annotataed allocentric marker info
        allo_marker_file = os.path.join(path, 'allo_marker.json')
        with open(allo_marker_file, 'r') as json_file:
            allo_markers = json.load(json_file)['markers']
        num_am = np.array(allo_markers).shape[0]

        for i in tqdm(range(len(ego_marker_files))):
            ego_marker_file = ego_marker_files[i]
            with open(os.path.join(ego_marker_path, ego_marker_file), 'r') as json_file:
                json_info = json.load(json_file)
                ego_markers = json_info['markers']
                ego_ir_file = json_info['image_path']
            ego_depth_file = os.path.join(depth_path, depth_files[i])
            ego_rgb_file = ego_ir_file.replace('/ir', '/rgb')

            # keep the first num_am markers from the left from egocentric view
            em = np.array(ego_markers)
            sort_indices = np.argsort(em[:, 0])
            em = em[sort_indices][:num_am].astype('int')

            # find the correpondence pixel on depth image
            aligned_depth_img = align_depth2rgb(ego_rgb_file, ego_depth_file, rgb_camera_matrix, rgb_distortion_coeffs, \
                                    depth_camera_matrix, depth_distortion_coeffs, rotation_matrix, translation_vector)
            em_3d = np.hstack((em, aligned_depth_img[em[:,1], em[:,0]][:,np.newaxis]))
            
            # calibrate between the RGB and the mocap 08_A
            mocap_to_cam = np.eye(4)
            retval, rvec, tvec = cv2.solvePnP(em_3d, allo_markers, camera_matrix, dist_coeffs)
            rot_mat, _ = cv2.Rodrigues(rvec)
            mocap_to_cam[:3,:3] = rot_mat
            mocap_to_cam[:3, 3] = tvec.squeeze(1)

            #  read the mocap configuration file
            # mocap_file = glob(os.path.join(path, 'meshes', '*.csd'))
            # mocap_config = np.genfromtxt(mocap_file, dtype=str)
            # mocap_names = [config[0] for config in mocap_config]
            # mocap_params = np.array([config[1:] for config in mocap_config], dtype=float)

      

            # apply the solvePnP 
            # mocap_to_cam = np.eye(4)
            # retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            # rot_mat, _ = cv2.Rodrigues(rvec)
            # mocap_to_cam[:3,:3] = rot_mat
            # mocap_to_cam[:3, 3] = tvec.squeeze(1)


if __name__ == '__main__':
    main()