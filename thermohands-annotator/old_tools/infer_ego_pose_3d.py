import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import time
import torch
import json

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

def project_to_2d(keypoints_3d, camera_matrix):

    uvs = keypoints_3d[:, :2] / keypoints_3d[:, 2:3]
    uvs[:, 0] = uvs[:, 0] * camera_matrix[0, 0] + camera_matrix[0, 2]
    uvs[:, 1] = uvs[:, 1] * camera_matrix[1, 1] + camera_matrix[1, 2]
    uvs = np.round(uvs).astype(int)

    return uvs
    
def convert_to_3d(keypoints_2d, depths, camera_matrix):
    # Extracting camera parameters
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

    # Vectorized computation
    X = ((keypoints_2d[:,0] - cx) * depths) / fx
    Y = ((keypoints_2d[:,1] - cy) * depths) / fy
    Z = depths

    # Combine X, Y, Z to get 3D keypoints
    keypoints_3d = np.vstack((X, Y, Z))

    return keypoints_3d.T

def draw_3d_keypoints(vis_path, ego_pose_l_3d, ego_pose_r_3d):

    # Draw lines connecting hand landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (0, 9), (0, 13), (0, 17)
    ]
    conn_color = [
        'red', 'red', 'red', 'red',
        'green', 'green', 'green',
        'blue', 'blue', 'blue',
        'purple', 'purple', 'purple',
        'cyan', 'cyan', 'cyan',
        'green', 'blue', 'purple', 'cyan'
    ]

    # Creating a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot
    # ax.scatter(ego_pose_l_3d[:,0], ego_pose_l_3d[:,1], ego_pose_l_3d[:,2])
    # ax.scatter(ego_pose_r_3d[:,0], ego_pose_r_3d[:,1], ego_pose_r_3d[:,2])
    for (start, end), col in zip(connections, conn_color):
        start_point = ego_pose_l_3d[start]
        end_point = ego_pose_l_3d[end]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col, linewidth = 2)
        start_point = ego_pose_r_3d[start]
        end_point = ego_pose_r_3d[end]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=col, linewidth = 2)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_xlim([-0.2, 0.3])
        ax.set_ylim([-0.2, 0.3])
        ax.set_zlim([0.1, 0.6])
        ax.view_init(azim=115)
    
    # save the plot
    plt.savefig(vis_path)
    plt.close()
    plt.clf()

def draw_2d_keypoints(output_path, keypoints_l, keypoints_r, image):

    keypoints_l = keypoints_l.tolist()
    keypoints_r = keypoints_r.tolist()
    # draw keypoints and connections
    for keypoint in keypoints_l:
        cv2.circle(image, (keypoint[0], keypoint[1]), 5, (0, 255, 0), -1)
    for keypoint in keypoints_r:
        cv2.circle(image, (keypoint[0], keypoint[1]), 5, (0, 255, 0), -1)
    # Draw lines connecting hand landmarks
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20)
    ]
    for connection in connections:
        point1 = keypoints_l[connection[0]]
        point2 = keypoints_l[connection[1]]
        cv2.line(image, tuple(point1), tuple(point2), (0, 255, 0), 2)
        point1 = keypoints_r[connection[0]]
        point2 = keypoints_r[connection[1]]
        cv2.line(image, tuple(point1), tuple(point2), (0, 255, 0), 2)
    # save image with projected keypoints
    cv2.imwrite(output_path, image)

def limit_2d_coordinates(keypoint_2d_l, keypoint_2d_r, shape):

    keypoint_2d_l[:,1] = np.maximum(keypoint_2d_l[:,1], 0)
    keypoint_2d_l[:,0] = np.maximum(keypoint_2d_l[:,0], 0)
    keypoint_2d_r[:,1] = np.maximum(keypoint_2d_r[:,1], 0)
    keypoint_2d_r[:,0] = np.maximum(keypoint_2d_r[:,0], 0)
    keypoint_2d_l[:,1] = np.minimum(keypoint_2d_l[:,1], shape[0]-1)
    keypoint_2d_l[:,0] = np.minimum(keypoint_2d_l[:,0], shape[1]-1)
    keypoint_2d_r[:,1] = np.minimum(keypoint_2d_r[:,1], shape[0]-1)
    keypoint_2d_r[:,0] = np.minimum(keypoint_2d_r[:,0], shape[1]-1)

    return keypoint_2d_l, keypoint_2d_r

def process_3d_pose(depth_file, thermal_file, ir_file, pose_file, ego_calib, save_path, vis_path, save_t_path, vis_t_path, vis_2d_t_path):

    # camera parameters
    ir_camera_matrix = np.array(ego_calib['ir_mtx'])  # Intrinsic matrix for RGB camera
    ir_distortion_coeffs = np.array(ego_calib['ir_dist'])  # Distortion coefficients for RGB camera
    r2t_rotation = np.array(ego_calib['rgb2thermal_rotation_matrix'])
    r2t_translation = np.array(ego_calib['rgb2thermal_translation_vector'])
    d2r_rotation = np.array(ego_calib['depth2rgb_rotation_matrix'])
    d2r_translation = np.array(ego_calib['depth2rgb_translation_vector'])
    th_camera_matrix = np.array(ego_calib['thermal_mtx']) 
    th_distortion_coeffs = np.array(ego_calib['thermal_dist'])
    
    d2t_translation = d2r_translation + r2t_translation
    # read the depth image
    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    depth_img_undist = cv2.undistort(depth_img, ir_camera_matrix, ir_distortion_coeffs)
    # read the thermal image
    thermal_img = cv2.imread(thermal_file)
    # read the ir image
    ir_img = cv2.imread(ir_file)
    # read the ego pose file
    with open(pose_file, 'r') as json_file:
        ego_pose_info = json.load(json_file)
    left_hand_index = ego_pose_info['left_hand_index'][0]
    hand_landmarks = ego_pose_info['hand_landmarks']
    ego_pose_l = np.array(hand_landmarks[left_hand_index])
    ego_pose_r = np.array(hand_landmarks[1-left_hand_index])
    # undistort the depth image and 2d keypoints coordinates
    depth_img_undist  = depth_img_undist * 0.001
    ego_pose_l, ego_pose_r = limit_2d_coordinates(ego_pose_l, ego_pose_r, depth_img_undist.shape)
    ego_pose_l_3d = convert_to_3d(ego_pose_l, depth_img_undist[ego_pose_l[:,1], ego_pose_l[:,0]], ir_camera_matrix) 
    ego_pose_r_3d = convert_to_3d(ego_pose_r, depth_img_undist[ego_pose_r[:,1], ego_pose_r[:,0]], ir_camera_matrix) 
    
    # project the 3d keypoints to the ir image
    # uvs_l = project_to_2d(ego_pose_l_3d, ir_camera_matrix)
    # uvs_r = project_to_2d(ego_pose_r_3d, ir_camera_matrix)
    # uv_l = distort_coordinates(uvs_l[:, np.newaxis, :2], ir_camera_matrix, ir_distortion_coeffs[0]).reshape(-1,2).astype(int)
    # uv_r = distort_coordinates(uvs_r[:, np.newaxis, :2], ir_camera_matrix, ir_distortion_coeffs[0]).reshape(-1,2).astype(int)
    # uv_l, uv_r = limit_2d_coordinates(uv_l, uv_r, ir_img.shape)
    # output_path = os.path.join(vis_2d_path, ir_file.split('/')[-1].split('.')[-2] + '.png')
    # draw_2d_keypoints(output_path, uv_l, uv_r, ir_img)

    # visualize 3d ego hand pose
    output_path = os.path.join(vis_path, os.path.basename(depth_file))
    draw_3d_keypoints(output_path, ego_pose_l_3d, ego_pose_r_3d)

    # transform the 3d keypoints to the thermal camera coordinates
    d2r_T = np.eye(4)
    r2t_T = np.eye(4)
    d2r_T[:3, :3] = d2r_rotation
    d2r_T[:3,  3] = d2r_translation[0]
    r2t_T[:3, :3] = r2t_rotation
    r2t_T[:3,  3] = r2t_translation[0]
    ego_pose_l_3d_h = np.hstack((ego_pose_l_3d, np.ones((ego_pose_l_3d.shape[0], 1)))).T
    ego_pose_r_3d_h = np.hstack((ego_pose_r_3d, np.ones((ego_pose_r_3d.shape[0], 1)))).T
    ego_pose_l_3d_t = ((r2t_T) @ (d2r_T) @ ego_pose_l_3d_h)[:3,:].T
    ego_pose_r_3d_t = ((r2t_T) @ (d2r_T) @ ego_pose_r_3d_h)[:3,:].T
    output_path = os.path.join(vis_t_path, os.path.basename(depth_file))
    draw_3d_keypoints(output_path, ego_pose_l_3d_t, ego_pose_r_3d_t)

    # project the 3d keypoints to the thermal image
    uvs_l = project_to_2d(ego_pose_l_3d_t, th_camera_matrix)
    uvs_r = project_to_2d(ego_pose_r_3d_t, th_camera_matrix)
    uv_l = distort_coordinates(uvs_l[:, np.newaxis, :2], th_camera_matrix, th_distortion_coeffs[0]).reshape(-1,2).astype(int)
    uv_r = distort_coordinates(uvs_r[:, np.newaxis, :2], th_camera_matrix, th_distortion_coeffs[0]).reshape(-1,2).astype(int)
    uv_l, uv_r = limit_2d_coordinates(uv_l, uv_r, thermal_img.shape)
    output_path = os.path.join(vis_2d_t_path, thermal_file.split('/')[-1].split('.')[-2] + '.png')
    draw_2d_keypoints(output_path, uv_l, uv_r, thermal_img)

    # Create a dictionary to store the hand pose results
    output_path = os.path.join(save_path, os.path.splitext(os.path.basename(depth_file))[0] + ".json")
    hand_pose_data = {"left": ego_pose_l_3d.tolist(), 'right': ego_pose_r_3d.tolist()}
    with open(output_path, "w") as json_file:
        json.dump(hand_pose_data, json_file, indent=4)
    # Create a dictionary to store the hand pose results in thermal coordiantes
    output_path = os.path.join(save_t_path, os.path.splitext(os.path.basename(depth_file))[0] + ".json")
    hand_pose_data = {"left": ego_pose_l_3d_t.tolist(), 'right': ego_pose_r_3d_t.tolist()}
    with open(output_path, "w") as json_file:
        json.dump(hand_pose_data, json_file, indent=4)


def main():

    root_dir = '/mnt/data/MultimodalEgoHands/subject_13/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_13/'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        ego_calib_file = '/mnt/data/MultimodalEgoHands/calibration/ego_calib.json'
        depth_path = os.path.join(path, 'egocentric', 'depth')
        thermal_path = os.path.join(path, 'egocentric', 'thermal')
        ir_path = os.path.join(path, 'egocentric', 'ir')
        pose_path = os.path.join(save_dir, clip, 'ego', 'pose_2d')
        mask_path = os.path.join(save_dir, clip, 'ego', 'mask_2d')
        save_3d_path = os.path.join(save_dir, clip, 'ego', 'pose_3d')
        vis_3d_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_3d')
        save_3d_t_path = os.path.join(save_dir, clip, 'ego', 'pose_3d_t')
        vis_3d_t_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_3d_t')
        vis_2d_t_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_2d_t')
        if not os.path.exists(vis_3d_path):
            os.makedirs(vis_3d_path)
        if not os.path.exists(save_3d_path):
            os.makedirs(save_3d_path)
        if not os.path.exists(vis_3d_t_path):
            os.makedirs(vis_3d_t_path)
        if not os.path.exists(save_3d_t_path):
            os.makedirs(save_3d_t_path)
        if not os.path.exists(vis_2d_t_path):
            os.makedirs(vis_2d_t_path)
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        depth_files = sorted(glob(depth_path + '/'+ '*.png'))
        thermal_files = sorted(glob(thermal_path + '/'+ '*.tiff'))
        ir_files = sorted(glob(ir_path + '/'+ '*.png'))
        pose_files = sorted(glob(pose_path + '/'+ '*.json'))
        for depth_file, thermal_file, ir_file, pose_file in tqdm(zip(depth_files, thermal_files, ir_files, pose_files), total = len(depth_files)):
            process_3d_pose(depth_file, thermal_file, ir_file, pose_file, ego_calib, save_3d_path, vis_3d_path, save_3d_t_path, vis_3d_t_path, vis_2d_t_path)

if __name__ == "__main__":
    main()
