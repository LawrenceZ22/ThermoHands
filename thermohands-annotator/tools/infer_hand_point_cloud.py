import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd
import open3d as o3d
    
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

def draw_hand_pcd(vis_pc_path, left_pcd, right_pcd):

    fig = plt.figure(figsize=(12, 6))
    # Create first subplot for the first point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(left_pcd[:,0], left_pcd[:,1], left_pcd[:,2], s=0.5, color='blue')
    ax1.scatter(right_pcd[:,0], right_pcd[:,1], right_pcd[:,2], s=0.5, color='blue')
    ax1.set_title('Left - Hand Point Cloud')
    ax1.xaxis.pane.set_facecolor('white')  # Change x-axis pane color
    ax1.yaxis.pane.set_facecolor('white')  # Change y-axis pane color
    ax1.zaxis.pane.set_facecolor('white')  # Change z-axis pane color
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.view_init(elev = -10, azim = -70)

    # Create second subplot for the second point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(right_pcd[:,0], right_pcd[:,1], right_pcd[:,2], s=0.5, color='red')
    ax2.set_title('Right - Hand Point Cloud')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    ax2.view_init(elev = -10, azim = -60)

    # save figure
    plt.savefig(vis_pc_path, dpi = 200)
    plt.close()
    plt.clf()


def process_hand_pcd(depth_file, mask_file, depth_exo_file, mask_exo_file, ego_calib, exo_calib, transform, save_hpc_l_path, save_hpc_r_path, vis_pc_path):

    transform = np.array(transform).reshape(4,4)
    transform_inv = np.linalg.inv(transform)

    ego_camera_matrix = np.array(ego_calib['ir_mtx'])  # Intrinsic matrix for RGB camera
    ego_distortion_coeffs = np.array(ego_calib['ir_dist'])  # Distortion coefficients for RGB camera
    exo_camera_matrix = np.array(exo_calib['ir_mtx'])  # Intrinsic matrix for RGB camera
    exo_distortion_coeffs = np.array(exo_calib['ir_dist'])  # Distortion coefficients for RGB camera

    # read the depth image
    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    depth_img_undist = cv2.undistort(depth_img, ego_camera_matrix, ego_distortion_coeffs)
    depth_img_undist  = depth_img_undist * 0.001
    depth_exo_img = cv2.imread(depth_exo_file, cv2.IMREAD_UNCHANGED)
    depth_exo_img_undist = cv2.undistort(depth_exo_img, exo_camera_matrix, exo_distortion_coeffs)
    depth_exo_img_undist  = depth_exo_img_undist * 0.001
    
    # read the ego mask file and get the point cloud
    ego_mask = np.array(pd.read_csv(mask_file, header=None))
    ego_mask_l = np.where(ego_mask == 1)
    ego_mask_r = np.where(ego_mask == 2)
    hand_left_coord = np.vstack((ego_mask_l[1],ego_mask_l[0])).T
    hand_right_coord = np.vstack((ego_mask_r[1],ego_mask_r[0])).T
    hand_left_pcd = convert_to_3d(hand_left_coord, depth_img_undist[hand_left_coord[:,1], hand_left_coord[:,0]], ego_camera_matrix)
    hand_right_pcd = convert_to_3d(hand_right_coord, depth_img_undist[hand_right_coord[:,1], hand_right_coord[:,0]], ego_camera_matrix)

    # read the exo mask file and get the point cloud
    exo_mask = np.array(pd.read_csv(mask_exo_file, header=None))
    exo_mask_l = np.where(exo_mask == 2)
    exo_mask_r = np.where(exo_mask == 1)
    hand_left_exo_coord = np.vstack((exo_mask_l[1],exo_mask_l[0])).T
    hand_right_exo_coord = np.vstack((exo_mask_r[1],exo_mask_r[0])).T
    # delete depth = 0 pixels
    hand_left_exo_coord = hand_left_exo_coord[depth_exo_img_undist[hand_left_exo_coord[:,1], hand_left_exo_coord[:,0]]>0.2]
    hand_right_exo_coord = hand_right_exo_coord[depth_exo_img_undist[hand_right_exo_coord[:,1], hand_right_exo_coord[:,0]]>0.2]
    hand_left_exo_pcd = convert_to_3d(hand_left_exo_coord, depth_exo_img_undist[hand_left_exo_coord[:,1], hand_left_exo_coord[:,0]], exo_camera_matrix)
    hand_right_exo_pcd = convert_to_3d(hand_right_exo_coord, depth_exo_img_undist[hand_right_exo_coord[:,1], hand_right_exo_coord[:,0]], exo_camera_matrix)

    hand_left_exo_pcd = (transform_inv[:3,:3] @ hand_left_exo_pcd.T + transform_inv[:3, 3][:, np.newaxis]).T
    hand_right_exo_pcd = (transform_inv[:3,:3] @ hand_right_exo_pcd.T + transform_inv[:3, 3][:,np.newaxis]).T
    hand_left_pcd_agg = np.vstack((hand_left_pcd, hand_left_exo_pcd))
    hand_right_pcd_agg = np.vstack((hand_right_pcd, hand_right_exo_pcd))
    
    # left_point_cloud = o3d.geometry.PointCloud()
    # left_point_cloud.points = o3d.utility.Vector3dVector(hand_left_pcd_agg)
    # right_point_cloud = o3d.geometry.PointCloud()
    # right_point_cloud.points = o3d.utility.Vector3dVector(hand_right_pcd_agg)
    # left_pcd , _ = left_point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    # right_pcd , _ = right_point_cloud.remove_radius_outlier(nb_points=16, radius=0.05)
    # hand_left_pcd_agg = np.asarray(left_pcd.points)
    # hand_right_pcd_agg = np.asarray(right_pcd.points)

    # save and draw
    output_path = os.path.join(save_hpc_l_path, os.path.splitext(os.path.basename(depth_file))[0] + ".bin")
    hand_left_pcd_agg.astype(np.float32).tofile(output_path)
    # hand_left_pcd.astype(np.float32).tofile(output_path)
    output_path = os.path.join(save_hpc_r_path, os.path.splitext(os.path.basename(depth_file))[0] + ".bin")
    hand_right_pcd_agg.astype(np.float32).tofile(output_path)
    # hand_right_pcd.astype(np.float32).tofile(output_path)
    vis_path = os.path.join(vis_pc_path, os.path.splitext(os.path.basename(depth_file))[0] + ".png")
    draw_hand_pcd(vis_path, hand_left_pcd_agg, hand_right_pcd_agg)
   
    
def infer_hand_pcd(root_dir, save_dir):

    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        ego_calib_file = '/mnt/data/MultimodalEgoHands/calibration/ego_calib.json'
        exo_calib_file = "/mnt/data/MultimodalEgoHands/calibration/exo_calib.json"
        depth_path = os.path.join(path, 'egocentric', 'depth')
        depth_exo_path = os.path.join(path, 'exocentric', 'depth')
        transform_path = os.path.join(save_dir, clip, 'exo_ego_transform.csv')
        mask_path = os.path.join(save_dir, clip, 'ego', 'mask_2d')
        mask_exo_path = os.path.join(save_dir, clip, 'exo', 'mask_2d')
        save_hpc_l_path = os.path.join(save_dir, clip, 'ego', 'hand_pcd_L')
        save_hpc_r_path = os.path.join(save_dir, clip, 'ego', 'hand_pcd_R')
        vis_pc_path = os.path.join(save_dir, clip, 'vis_ego', 'hand_pcd')
        if not os.path.exists(save_hpc_l_path):
            os.makedirs(save_hpc_l_path)
        if not os.path.exists(save_hpc_r_path):
            os.makedirs(save_hpc_r_path)
        if not os.path.exists(vis_pc_path):
            os.makedirs(vis_pc_path)
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        with open(exo_calib_file, 'r') as file:
            exo_calib = json.load(file)
        transforms = np.genfromtxt(transform_path, delimiter=',').tolist()
        depth_files = sorted(glob(depth_path + '/'+ '*.png'))
        mask_files = sorted(glob(mask_path + '/'+ '*.csv'))
        depth_exo_files = sorted(glob(depth_exo_path + '/'+ '*.png'))
        mask_exo_files = sorted(glob(mask_exo_path + '/'+ '*.csv'))
       
        for depth_file, mask_file, depth_exo_file, mask_exo_file, transform in tqdm(zip(depth_files, mask_files, depth_exo_files, mask_exo_files, transforms), total = len(depth_files), desc = clip):
             process_hand_pcd(depth_file, mask_file, depth_exo_file, mask_exo_file, ego_calib, exo_calib, transform, save_hpc_l_path, save_hpc_r_path, vis_pc_path)

if __name__ == "__main__":

    root_dir = '/mnt/data/MultimodalEgoHands/subject_01/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_01/'
    infer_hand_pcd(root_dir, save_dir)
