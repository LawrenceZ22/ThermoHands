import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Function to triangulate points
def triangulate_points(proj_matrix1, proj_matrix2, points1, points2):
    # Convert points to homogeneous coordinates
    points4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1, points2)
    # Convert from homogeneous coordinates to 3D
    points3d = points4d[:3, :] / points4d[3, :]
    return points3d.T

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
        ax.set_xlim([-0.1, 0.4])
        ax.set_ylim([-0.2, 0.3])
        ax.set_zlim([0.2, 0.7])
        ax.xaxis.pane.fill = False  # Enable filling for the x-axis pane (bottom)
        ax.yaxis.pane.fill = False # Enable filling for the y-axis pane (left side)
        ax.zaxis.pane.fill = False  # Enable filling for the z-axis pane (back side)

        ax.xaxis.pane.set_facecolor('white')  # Change x-axis pane color
        ax.yaxis.pane.set_facecolor('white')  # Change y-axis pane color
        ax.zaxis.pane.set_facecolor('white')  # Change z-axis pane color

        ax.view_init(elev = -10, azim = -70)
    
    # save the plot
    plt.savefig(vis_path, dpi = 200)
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


def process_3d_pose(transform, rgb_file, pose_file, pose_exo_file, ego_calib, exo_calib, save_path, vis_path, vis_2d_path):

    # read the calibration results
    ego_cam_matrix = np.array(ego_calib['ir_mtx']) 
    exo_cam_matrix = np.array(exo_calib['ir_mtx'])
    ego_dist = np.array(ego_calib['ir_dist']) 
    exo_dist = np.array(exo_calib['ir_dist'])

    rgb_img = cv2.imread(rgb_file)
    rgb_img = cv2.undistort(rgb_img, ego_cam_matrix, ego_dist)

    # read the ego pose and 3d pose file
    with open(pose_file, 'r') as json_file:
        ego_pose_info = json.load(json_file)
    left_hand_index = ego_pose_info['left_hand_index'][0]
    hand_landmarks = ego_pose_info['hand_landmarks']
    ego_pose_l = np.array(hand_landmarks[left_hand_index])
    ego_pose_r = np.array(hand_landmarks[1-left_hand_index])
   
   # read the exo pose file (the left hand corresponds to the right one in ego view)
    with open(pose_exo_file, 'r') as json_file:
        exo_pose_info = json.load(json_file)
    left_hand_index = exo_pose_info['left_hand_index'][0]
    hand_landmarks = exo_pose_info['hand_landmarks']
    exo_pose_l = np.array(hand_landmarks[1-left_hand_index])
    exo_pose_r = np.array(hand_landmarks[left_hand_index])
    
    # Get the triangulated 3D keypoints
    ego_project_matrix = ego_cam_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    exo_project_matrix = exo_cam_matrix @ transform.reshape(4,4)[:3]
    kps_3d_l = triangulate_points(ego_project_matrix, exo_project_matrix, ego_pose_l.T.astype(np.float), exo_pose_l.T.astype(np.float))
    kps_3d_r = triangulate_points(ego_project_matrix, exo_project_matrix, ego_pose_r.T.astype(np.float), exo_pose_r.T.astype(np.float))
    
    # project the 3d keypoints to the ir image
    # uvs_l = project_to_2d(kps_3d_l , ego_cam_matrix)
    # uvs_r = project_to_2d(kps_3d_r, ego_cam_matrix)
    # uv_l = distort_coordinates(uvs_l[:, np.newaxis, :2], ir_camera_matrix, ir_distortion_coeffs[0]).reshape(-1,2).astype(int)
    # uv_r = distort_coordinates(uvs_r[:, np.newaxis, :2], ir_camera_matrix, ir_distortion_coeffs[0]).reshape(-1,2).astype(int)
    # uv_l, uv_r = limit_2d_coordinates(uv_l, uv_r, ir_img.shape)
    # output_path = os.path.join(vis_2d_path, rgb_file.split('/')[-1].split('.')[-2] + '.png')
    # draw_2d_keypoints(output_path, uvs_l, uvs_r, rgb_img)

    # visualize 3d ego hand pose
    output_path = os.path.join(vis_path, os.path.basename(rgb_file))
    draw_3d_keypoints(output_path, kps_3d_l, kps_3d_r)


    # Create a dictionary to store the hand pose results
    output_path = os.path.join(save_path, os.path.splitext(os.path.basename(rgb_file))[0] + ".json")
    hand_pose_data = {"left": kps_3d_l.tolist(), 'right': kps_3d_r.tolist()}
    with open(output_path, "w") as json_file:
        json.dump(hand_pose_data, json_file, indent=4)


def infer_3D_kps(root_dir, save_dir):

    
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        ego_calib_file = '/mnt/data/MultimodalEgoHands/calibration/ego_calib.json'
        exo_calib_file = "/mnt/data/MultimodalEgoHands/calibration/exo_calib.json"
        transform_path = os.path.join(save_dir, clip, 'exo_ego_transform.csv')
        pose_path = os.path.join(save_dir, clip, 'ego', 'pose_2d')
        pose_exo_path = os.path.join(save_dir, clip, 'exo', 'pose_2d')
        save_3d_path = os.path.join(save_dir, clip, 'ego', 'pose_3d_tri')
        vis_3d_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_3d_tri')
        vis_2d_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_2d_tri')
        rgb_path = os.path.join(path, 'egocentric', 'rgb')
        if not os.path.exists(vis_3d_path):
            os.makedirs(vis_3d_path)
        if not os.path.exists(vis_2d_path):
            os.makedirs(vis_2d_path)
        if not os.path.exists(save_3d_path):
            os.makedirs(save_3d_path)
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        with open(exo_calib_file, 'r') as file:
            exo_calib = json.load(file)
        rgb_files = sorted(glob(rgb_path + '/'+ '*.png'))
        transforms = np.genfromtxt(transform_path, delimiter=',')
        pose_files = sorted(glob(pose_path + '/'+ '*.json'))
        pose_exo_files = sorted(glob(pose_exo_path + '/'+ '*.json'))
        for i, rgb_file, pose_file, pose_exo_file in tqdm(zip(range(len(pose_files)), rgb_files, pose_files, pose_exo_files), total = len(pose_files)):
            process_3d_pose(transforms[i], rgb_file, pose_file, pose_exo_file, ego_calib, exo_calib, save_3d_path, vis_3d_path, vis_2d_path)

if __name__ == "__main__":
    root_dir = '/mnt/data/MultimodalEgoHands/subject_01/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_01/'
    infer_3D_kps(root_dir, save_dir)
