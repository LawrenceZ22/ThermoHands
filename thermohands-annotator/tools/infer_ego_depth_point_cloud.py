import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

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
    ax1.scatter(left_pcd[::4,0], left_pcd[::4,1], left_pcd[::4,2], s=0.5, color='blue')
    ax1.set_title('Left - Hand Point Cloud')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    # ax1.view_init(elev = 60)

    # Create second subplot for the second point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(right_pcd[::4,0], right_pcd[::4,1], right_pcd[::4,2], s=0.5, color='red')
    ax2.set_title('Right - Hand Point Cloud')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    # ax2.view_init(elev = 60)

    # save figure
    plt.savefig(vis_pc_path, dpi = 200)
    plt.close()
    plt.clf()


def process_depth_to_pc(depth_file, ego_calib, save_pc_path, vis_pc_path):

    ir_camera_matrix = np.array(ego_calib['ir_mtx'])  # Intrinsic matrix for RGB camera
    ir_distortion_coeffs = np.array(ego_calib['ir_dist'])  # Distortion coefficients for RGB camera
    # read the depth image
    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    depth_img_undist = cv2.undistort(depth_img, ir_camera_matrix, ir_distortion_coeffs)
    depth_img_undist  = depth_img_undist * 0.001
    y_coords, x_coords = np.meshgrid(range(depth_img_undist.shape[0]), range(depth_img_undist.shape[1]), indexing='ij')
    pixel_coordinates = np.vstack((x_coords.flatten(),y_coords.flatten())).T
    point_cloud = convert_to_3d(pixel_coordinates, depth_img_undist[pixel_coordinates[:,1], pixel_coordinates[:,0]], ir_camera_matrix)
    # read the ego mask file
    # ego_mask = np.array(pd.read_csv(mask_file, header=None))
    # ego_mask_l = np.where(ego_mask == 1)
    # ego_mask_r = np.where(ego_mask == 2)
    # hand_left_coord = np.vstack((ego_mask_l[1],ego_mask_l[0])).T
    # hand_right_coord = np.vstack((ego_mask_r[1],ego_mask_r[0])).T
    # hand_left_pcd = convert_to_3d(hand_left_coord, depth_img_undist[hand_left_coord[:,1], hand_left_coord[:,0]], ir_camera_matrix)
    # hand_right_pcd = convert_to_3d(hand_right_coord, depth_img_undist[hand_right_coord[:,1], hand_right_coord[:,0]], ir_camera_matrix)
    output_path = os.path.join(save_pc_path, os.path.splitext(os.path.basename(depth_file))[0] + ".bin")
    point_cloud.astype(np.float32).tofile(output_path)
    # output_path = os.path.join(save_hpc_l_path, os.path.splitext(os.path.basename(depth_file))[0] + ".bin")
    # hand_left_pcd.astype(np.float32).tofile(output_path)
    # output_path = os.path.join(save_hpc_r_path, os.path.splitext(os.path.basename(depth_file))[0] + ".bin")
    # hand_right_pcd.astype(np.float32).tofile(output_path)
    # vis_path = os.path.join(vis_pc_path, os.path.splitext(os.path.basename(depth_file))[0] + ".png")
    # draw_hand_pcd(vis_path, point_cloud, point_cloud)
   
    
def infer_ego_point_cloud(root_dir, save_dir):

    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        ego_calib_file = '/mnt/data/MultimodalEgoHands/calibration/ego_calib.json'
        depth_path = os.path.join(path, 'egocentric', 'depth')
        save_pc_path = os.path.join(save_dir, clip, 'ego', 'depth_pcd')
        vis_pc_path = os.path.join(save_dir, clip, 'vis_ego', 'depth_pcd')
        if not os.path.exists(save_pc_path):
            os.makedirs(save_pc_path)
        if not os.path.exists(vis_pc_path):
            os.makedirs(vis_pc_path)
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        depth_files = sorted(glob(depth_path + '/'+ '*.png'))
        for depth_file in tqdm(depth_files, total = len(depth_files), desc = clip):
            process_depth_to_pc(depth_file, ego_calib, save_pc_path, vis_pc_path)

if __name__ == "__main__":
    root_dir = '/mnt/data/MultimodalEgoHands/subject_03/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_03/'
    infer_ego_point_cloud(root_dir, save_dir)
