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


def compute_error(transform, pre_file, pose_ego_file, pose_exo_file, ego_calib, exo_calib):

    # read the calibration results
    ego_cam_matrix = np.array(ego_calib['ir_mtx']) 
    exo_cam_matrix = np.array(exo_calib['ir_mtx'])
    ego_dist = np.array(ego_calib['ir_dist']) 
    exo_dist = np.array(exo_calib['ir_dist'])

    # read the ego pose and 3d pose file
    with open(pose_ego_file, 'r') as json_file:
        ego_pose_info = json.load(json_file) 
    assert len(ego_pose_info['points']) == 42
    ego_pose_l = np.array(ego_pose_info['points'][:21])
    ego_pose_r = np.array(ego_pose_info['points'][21:])
   
    # read the exo pose file (the left hand corresponds to the right one in ego view)
    with open(pose_exo_file, 'r') as json_file:
        exo_pose_info = json.load(json_file) 
    assert len(exo_pose_info['points']) == 42
    exo_pose_l = np.array(exo_pose_info['points'][21:])
    exo_pose_r = np.array(exo_pose_info['points'][:21])
    exo_pose_l_d = distort_coordinates(exo_pose_l[:, np.newaxis, :2], exo_cam_matrix, exo_dist[0]).squeeze(1)
    exo_pose_r_d = distort_coordinates(exo_pose_r[:, np.newaxis, :2], exo_cam_matrix, exo_dist[0]).squeeze(1)
    
    
    # Get the triangulated 3D keypoints
    ego_project_matrix = ego_cam_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    exo_project_matrix = exo_cam_matrix @ transform.reshape(4,4)[:3]
    kps_3d_l = triangulate_points(ego_project_matrix, exo_project_matrix, ego_pose_l.T.astype(np.float), exo_pose_l_d.T.astype(np.float))
    kps_3d_r = triangulate_points(ego_project_matrix, exo_project_matrix, ego_pose_r.T.astype(np.float), exo_pose_r_d.T.astype(np.float))
    
    # read the predicted 3d pose file
    with open(pre_file, 'r') as json_file:
        pre_info = json.load(json_file) 
    kps_pre_l = np.array(pre_info['kps3D_L'])
    kps_pre_r = np.array(pre_info['kps3D_R'])

    error_l = np.linalg.norm(kps_3d_l - kps_pre_l, axis=1)
    error_r = np.linalg.norm(kps_3d_r - kps_pre_r, axis=1)

    return error_l.mean(), error_r.mean()


def main():

    clip_names = ['subject_21/pour_water', 'subject_01_gesture/swipe']
    error_all = []
    error_a_l = []
    error_a_r = []
    for clip in clip_names:
        gt_dir = '/mnt/data/fangqiang/mannual-anno-hand/' + clip
        pre_dir = '/mnt/data/fangqiang/TherHandsPro/' + clip
        ego_calib_file = '/mnt/data/MultimodalEgoHands/calibration/ego_calib.json'
        exo_calib_file = "/mnt/data/MultimodalEgoHands/calibration/exo_calib.json"
        transform_path = os.path.join(pre_dir, 'exo_ego_transform.csv')
        pre_path = os.path.join(pre_dir, 'gt_info')
        gt_ego_path = os.path.join(gt_dir, 'ego_anno')
        gt_exo_path = os.path.join(gt_dir, 'exo_anno')
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        with open(exo_calib_file, 'r') as file:
            exo_calib = json.load(file)
        pre_files = sorted(glob(pre_path + '/'+ '*.json'))
        transforms = np.genfromtxt(transform_path, delimiter=',')
        gt_ego_files = sorted(glob(gt_ego_path + '/'+ '*.json'))
        gt_exo_files = sorted(glob(gt_exo_path + '/'+ '*.json'))
        error_l_ls = []
        error_r_ls = []
        error_ls = []
        for i, pre_file, gt_ego_file, gt_exo_file in tqdm(zip(range(len(pre_files)), pre_files, gt_ego_files, gt_exo_files), total = len(pre_files)):
            error_l, error_r = compute_error(transforms[i], pre_file, gt_ego_file, gt_exo_file, ego_calib, exo_calib)
            error_l_ls.append(error_l)
            error_r_ls.append(error_r)
            error_ls.append((error_l + error_r)/2)
        
        print(clip)
        print('The avg. MEPE (cm) for the left hand is {}'.format(np.array(error_l_ls).mean()*100))
        print('The std. MEPE (cm) for the left hand is {}'.format(np.array(error_l_ls).std()*100))
        print('The avg. MEPE (cm) for the right hand is {}'.format(np.array(error_r_ls).mean()*100))
        print('The std. MEPE (cm) for the right hand is {}'.format(np.array(error_r_ls).std()*100))
        print('The avg. MEPE (cm) for the two hands is {}'.format(np.array(error_ls).mean()*100))
        print('The std. MEPE (cm) for the two hand is {}'.format(np.array(error_ls).std()*100))
        print('Num. of frames: {}'.format(len(pre_files)))
    
        error_all.extend(error_ls)
        error_a_r.extend(error_r_ls)
        error_a_l.extend(error_l_ls)
    print('The avg. MEPE (cm) for the left hand is {}'.format(np.array(error_a_l).mean()*100))
    print('The std. MEPE (cm) for the left hand is {}'.format(np.array(error_a_l).std()*100))
    print('The avg. MEPE (cm) for the right hand is {}'.format(np.array(error_a_r).mean()*100))
    print('The std. MEPE (cm) for the right hand is {}'.format(np.array(error_a_r).std()*100))
    print('The avg. MEPE (cm) for the two hands is {}'.format(np.array(error_all).mean()*100))
    print('The std. MEPE (cm) for the two hand is {}'.format(np.array(error_all).std()*100))
    print('Num. of frames: {}'.format(len(error_all)))
    


       

if __name__ == "__main__":

    main()
