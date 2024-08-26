import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch
from manopth.manolayer import ManoLayer
import matplotlib.path as mplPath

   
def infer_mask_gt(gt_file, img_file, thermal_file, ego_calib, vis_mask_path, vis_mask_path_t):

    ncomps = 45
    torch.set_default_tensor_type('torch.DoubleTensor')
    mano_right = ManoLayer(
        mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True, side='right')
    mano_left = ManoLayer(
        mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', use_pca=False, ncomps=ncomps, flat_hand_mean=True, side='left')
    with open(gt_file, 'r') as json_file:
        load_hand = json.load(json_file)
    hand_pose = {}
    hand_pose["left_pose"] = load_hand['poseCoeff_L']
    hand_pose["left_tran"] = load_hand['trans_L']
    hand_pose["left_shape"] = load_hand['beta_L']
    hand_pose["right_pose"] = load_hand['poseCoeff_R']
    hand_pose["right_tran"] = load_hand['trans_R']
    hand_pose["right_shape"] = load_hand['beta_R']

    random_pose = torch.tensor(hand_pose["left_pose"])
    random_tran = torch.tensor(hand_pose["left_tran"])
    random_shape = torch.tensor(hand_pose["left_shape"])
    hand_verts, _ = mano_left(random_pose, random_shape)
    left_verts_scaled = hand_verts/1000.0 + random_tran

    random_pose = torch.tensor(hand_pose["right_pose"])
    random_tran = torch.tensor(hand_pose["right_tran"])
    random_shape = torch.tensor(hand_pose["right_shape"])
    hand_verts, _ = mano_right(random_pose, random_shape)
    right_verts_scaled = hand_verts/1000.0 + random_tran

    ego_cam_mtx = np.array(ego_calib['ir_mtx'])  # Intrinsic matrix for IR camera
    ego_dist_coeffs = np.array(ego_calib['ir_dist'])  # Distortion coefficients for IR camera
    th_cam_mtx = np.array(ego_calib['thermal_mtx'])  # Intrinsic matrix for thermal camera
    th_dist_coeffs = np.array(ego_calib['thermal_dist'])  # Distortion coefficients for thermal camera
    d2t_rot = torch.from_numpy(np.array(ego_calib['depth2thermal_rotation_matrix']))
    d2t_tran = torch.from_numpy(np.array(ego_calib['depth2thermal_translation_vector']))
    image = cv2.imread(img_file)
    image = cv2.undistort(image, ego_cam_mtx, ego_dist_coeffs)
    # distorted
    th_img = cv2.imread(thermal_file)

    vertices_l = (left_verts_scaled[0]).transpose(1,0)
    vertices_l_t = (d2t_rot) @ vertices_l + d2t_tran.T

    vertices_l = vertices_l[:2] / vertices_l[2]
    vertices_l[0] = vertices_l[0] * ego_cam_mtx[0, 0] + ego_cam_mtx[0, 2]
    vertices_l[1] = vertices_l[1] * ego_cam_mtx[1, 1] + ego_cam_mtx[1, 2]
    triangles_l = mano_left.th_faces.numpy()
    vertices_l = vertices_l.T.numpy().astype(np.int32)
   
    vertices_r = (right_verts_scaled[0]).transpose(1,0)
    vertices_r_t = d2t_rot @ vertices_r + d2t_tran.T

    vertices_r = vertices_r[:2] / vertices_r[2]
    vertices_r[0] = vertices_r[0] * ego_cam_mtx[0, 0] + ego_cam_mtx[0, 2]
    vertices_r[1] = vertices_r[1] * ego_cam_mtx[1, 1] + ego_cam_mtx[1, 2]
    triangles_r = mano_right.th_faces.numpy()
    vertices_r = vertices_r.T.numpy().astype(np.int32)
    mask_agg = np.zeros(image.shape[:2])

    # mask on the thermal image
    vertices_l_t = vertices_l_t[:2] / vertices_l_t[2]
    vertices_l_t[0] = vertices_l_t[0] * th_cam_mtx[0, 0] + th_cam_mtx[0, 2]
    vertices_l_t[1] = vertices_l_t[1] * th_cam_mtx[1, 1] + th_cam_mtx[1, 2]
    triangles_l = mano_left.th_faces.numpy()
    vertices_l_t = vertices_l_t.T.numpy().astype(np.int32)
    vertices_r_t = vertices_r_t[:2] / vertices_r_t[2]
    vertices_r_t[0] = vertices_r_t[0] * th_cam_mtx[0, 0] + th_cam_mtx[0, 2]
    vertices_r_t[1] = vertices_r_t[1] * th_cam_mtx[1, 1] + th_cam_mtx[1, 2]
    triangles_r = mano_right.th_faces.numpy()
    vertices_r_t = vertices_r_t.T.numpy().astype(np.int32)
    mask_agg_t = np.zeros(th_img.shape[:2])

    # Draw each triangle
    for triangle in triangles_l:
        # Get the vertices for the current triangle
        pts = np.array([vertices_l[triangle[0]], vertices_l[triangle[1]], vertices_l[triangle[2]]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Reshape for fillPoly function
        # Draw the triangle
        cv2.fillPoly(mask_agg, [pts], 255)  # 255 is the fill color
    for triangle in triangles_r:
        # Get the vertices for the current triangle
        pts = np.array([vertices_r[triangle[0]], vertices_r[triangle[1]], vertices_r[triangle[2]]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Reshape for fillPoly function
        # Draw the triangle
        cv2.fillPoly(mask_agg, [pts], 255)  # 255 is the fill color
    output_path = os.path.join(vis_mask_path, os.path.basename(img_file))
    cv2.imwrite(output_path, mask_agg)


    # Draw each triangle
    for triangle in triangles_l:
        # Get the vertices for the current triangle
        pts = np.array([vertices_l_t[triangle[0]], vertices_l_t[triangle[1]], vertices_l_t[triangle[2]]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Reshape for fillPoly function
        # Draw the triangle
        cv2.fillPoly(mask_agg_t, [pts], 255)  # 255 is the fill color
    for triangle in triangles_r:
        # Get the vertices for the current triangle
        pts = np.array([vertices_r_t[triangle[0]], vertices_r_t[triangle[1]], vertices_r_t[triangle[2]]], np.int32)
        pts = pts.reshape((-1, 1, 2))  # Reshape for fillPoly function
        # Draw the triangle
        cv2.fillPoly(mask_agg_t, [pts], 255)  # 255 is the fill color
    output_path = os.path.join(vis_mask_path_t, os.path.basename(thermal_file).split('.')[-2] + '.png')
    cv2.imwrite(output_path, mask_agg_t)



def infer_hand_mask_gt(root_dir, save_dir):

    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        clip_path = os.path.join(save_dir, clip)
        ego_calib_file = '/mnt/data/MultimodalEgoHands/calibration/ego_calib_opencv.json'
        gt_path = os.path.join(clip_path, 'gt_info')
        save_mask_path = os.path.join(save_dir, clip, 'gt_mask_info')
        vis_mask_path = os.path.join(save_dir, clip, 'gt_mask_vis')
        vis_mask_path_t = os.path.join(save_dir, clip, 'gt_mask_vis_t')
        img_path = os.path.join(root_dir, clip, 'egocentric', 'rgb')
        thermal_path = os.path.join(root_dir, clip, 'egocentric', 'thermal')
        rgb_files = sorted(glob(img_path + '/'+ '*.png'))
        thermal_files = sorted(glob(thermal_path + '/'+ '*.tiff'))
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)
        if not os.path.exists(vis_mask_path):
            os.makedirs(vis_mask_path)
        if not os.path.exists(vis_mask_path_t):
            os.makedirs(vis_mask_path_t)
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        gt_files = sorted(glob(gt_path + '/' + '*.json'))

        for gt_file, rgb_file, thermal_file in tqdm(zip(gt_files, rgb_files, thermal_files), total = len(gt_files), desc = clip):
             infer_mask_gt(gt_file, rgb_file, thermal_file, ego_calib, vis_mask_path, vis_mask_path_t)

if __name__ == "__main__":

    root_dir = '/mnt/data/MultimodalEgoHands/subject_01/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_01/'
    infer_hand_mask_gt(root_dir, save_dir)
