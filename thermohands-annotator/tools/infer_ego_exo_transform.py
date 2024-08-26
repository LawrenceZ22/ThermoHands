import os
import numpy as np
import json
import cv2
from tqdm import tqdm
from glob import glob

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

def limit_2d_coordinates(keypoint_2d_l, shape):

    keypoint_2d_l[:,1] = np.maximum(keypoint_2d_l[:,1], 0)
    keypoint_2d_l[:,0] = np.maximum(keypoint_2d_l[:,0], 0)
    keypoint_2d_l[:,1] = np.minimum(keypoint_2d_l[:,1], shape[0]-1)
    keypoint_2d_l[:,0] = np.minimum(keypoint_2d_l[:,0], shape[1]-1)

    return keypoint_2d_l

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

def init_the_transform(transforms, ego_calib, exo_calib, marker_2d_file, marker_exo_file, depth_file):

    ir_camera_matrix = np.array(ego_calib['ir_mtx'])  # Intrinsic matrix for RGB camera
    exo_camera_matrix = np.array(exo_calib['ir_mtx'])
    exo_camera_dist = np.array(exo_calib['ir_dist'])
    with open(marker_2d_file, 'r') as json_file:
        marker_2d = json.load(json_file)['markers']
    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    depth_img = depth_img * 0.001
    marker_2d = np.array(marker_2d).astype(np.int32)
    marker_2d = limit_2d_coordinates(marker_2d, depth_img.shape)
    marker_3d = convert_to_3d(marker_2d, depth_img[marker_2d[:,1], marker_2d[:,0]], ir_camera_matrix) 
    with open(marker_exo_file, 'r') as json_file:
        marker_exo = json.load(json_file)['markers']
    marker_exo = np.array(marker_exo).astype(np.float32)
    _, rotation_vector, translation_vector = cv2.solvePnP(marker_3d, marker_exo, exo_camera_matrix, exo_camera_dist)
    # Convert Rotation Vector to Rotation Matrix
    rotation_matrix,_ = cv2.Rodrigues(rotation_vector)
    # Create 4x4 Transformation Matrix
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector.T
    transformation_matrix[3, 3] = 1

    transforms.append(transformation_matrix.flatten())

    return transforms, marker_3d

def infer_one_tranform(transforms, marker_3d, odometry, exo_calib, vis_proj_path, exo_rgb_file):


    exo_camera_matrix = np.array(exo_calib['ir_mtx'])
    exo_camera_dist = np.array(exo_calib['ir_dist'])
    transformation_matrix = transforms[-1].reshape(4,4) @ odometry
    transforms.append(transformation_matrix.flatten())
    # inv_transform = np.linalg.inv(transformation_matrix)
    # visualize 3d markers projected onto the exo image
    marker_3d_trans = transformation_matrix[:3, :3] @ marker_3d.T + transformation_matrix[:3, 3][:, np.newaxis]
    marker_2d_proj = project_to_2d(marker_3d_trans.T, exo_camera_matrix)
    marker_2d_proj = distort_coordinates(marker_2d_proj[:, np.newaxis, :], exo_camera_matrix, exo_camera_dist[0]).reshape(-1,2).astype(int)
    # Load the IR image
    exo_rgb = cv2.imread(exo_rgb_file)
    marker_2d_proj = limit_2d_coordinates(marker_2d_proj, exo_rgb.shape)
    for marker in marker_2d_proj:
        x = int(marker[0])
        y = int(marker[1])
        cv2.rectangle(exo_rgb, (x-3, y-3), (x + 3, y + 3), (0, 255, 0), 2)
    # Save the annotated image to the output directory - marker
    output_path = os.path.join(vis_proj_path, os.path.basename(exo_rgb_file))
    cv2.imwrite(output_path, exo_rgb)

    return transforms
    

def infer_transform_matrix(root_dir, save_dir, kiss_icp_dir):

    
    clips = sorted(os.listdir(root_dir))
    kiss_icp_res = sorted(os.listdir(kiss_icp_dir))
    for i, clip in enumerate(clips): 
        kiss_icp_path = os.path.join(kiss_icp_dir, kiss_icp_res[i], 'depth_pcd_poses.npy')
        exo_rgb_path = os.path.join(root_dir, clip, 'exocentric', 'rgb')
        vis_proj_path = os.path.join(save_dir, clip, 'vis_exo', 'marker_2d_proj')
        vis_odom_path = os.path.join(save_dir, clip, 'vis_ego', 'odometry')
        marker_2d_path = os.path.join(save_dir, clip, 'ego', 'marker_2d_kf')
        marker_exo_path  = os.path.join(save_dir, clip, 'exo', 'marker_2d_anno')
        depth_path = os.path.join(root_dir, clip, 'egocentric', 'depth')
        marker_2d_files = sorted(glob(marker_2d_path + '/'+ '*.json'))
        marker_exo_files = sorted(glob(marker_exo_path + '/'+ '*.json'))
        depth_files = sorted(glob(depth_path + '/'+ '*.png'))
        exo_rgb_files = sorted(glob(exo_rgb_path + '/'+ '*.png'))
        if not os.path.exists(vis_proj_path):
            os.makedirs(vis_proj_path)
        if not os.path.exists(vis_odom_path):
            os.makedirs(vis_odom_path)
        #  read the egocentric calibration file
        ego_calib_file = '/mnt/data/MultimodalEgoHands/calibration/ego_calib.json'
        exo_calib_file = '/mnt/data/MultimodalEgoHands/calibration/exo_calib.json'
        # Open the file in read mode ('r')
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        with open(exo_calib_file, 'r') as file:
            exo_calib = json.load(file)
        odometry = np.load(kiss_icp_path)
        # assert len(marker_2d_files) == len(marker_exo_files)
        assert odometry.shape[0] == len(depth_files)
        transforms = []
        transforms, marker_3d = init_the_transform(transforms, ego_calib, exo_calib, marker_2d_files[0], marker_exo_files[0], depth_files[0])
        for j in tqdm(range(len(depth_files)), desc = clip, total = len(depth_files)):
            transforms = infer_one_tranform(transforms, marker_3d, odometry[j], exo_calib, vis_proj_path, exo_rgb_files[j])
        
        output_path = os.path.join(save_dir, clip, 'exo_ego_transform.csv')
        np.savetxt(output_path, np.array(transforms), fmt='%f', delimiter=',')

        # # visualizing the egocentric odometry
        # num_frames = len(transforms)
        # transforms = [transform.reshape((4,4)) for transform in transforms]
        # odometry = np.zeros((num_frames,4,4))
        # ego_loc = np.zeros((num_frames, 3))
        # odometry[0] = np.eye(4)
        # for i in range(1, num_frames):
        #     odometry[i] = np.linalg.inv(transforms[i-1]) @ transforms[i]
        # for i in range(1, num_frames):
        #     ego_loc[i] = (odometry[i] @ np.hstack((ego_loc[i-1], np.ones(1))))[:3]
        # for i in tqdm(range(num_frames)):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot(ego_loc[:i,0], ego_loc[:i,1], ego_loc[:i,2], color = (254/255, 129/255, 125/255), linewidth = 2)
        #     ax.set_xlabel('X axis')
        #     ax.set_ylabel('Y axis')
        #     ax.set_zlabel('Z axis')
        #     ax.set_xlim(-0.1,0.1)
        #     ax.set_ylim(-0.1,0.1)
        #     ax.set_zlim(-0.1,0.1)
        #     x_locator = MultipleLocator(0.02)
        #     y_locator = MultipleLocator(0.02)
        #     z_locator = MultipleLocator(0.02)
        #     ax.xaxis.set_major_locator(x_locator)
        #     ax.yaxis.set_major_locator(y_locator)
        #     ax.zaxis.set_major_locator(z_locator)
        #     ax.set_title('Egocentric Camera Trajectory - Frame {}'.format(i))
        #     output_path = os.path.join(vis_odom_path, '{}.png'.format(str(i).zfill(5)))
        #     plt.savefig(output_path, dpi = 300)
        #     plt.close()
        #     plt.clf()




        
        


if __name__ == '__main__':

    root_dir = '/mnt/data/MultimodalEgoHands/subject_03/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_03/'
    kiss_icp_dir = '/mnt/data/fangqiang/TherHandsPro/subject_03_odom/'
    infer_transform_matrix(root_dir, save_dir, kiss_icp_dir)