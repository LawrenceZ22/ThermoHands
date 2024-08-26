import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torch.optim as optim
from manopth.manolayer import ManoLayer
import pandas as pd
import torch
from tools.optimize_utils import *
from torch.optim import lr_scheduler



def optimize_frame_pose(index, transform, mask_file, hpc_L_file, hpc_R_file, pose_file, pose_3d_file, mask_exo_file, pose_exo_file, rgb_file, rgb_exo_file,\
                         ego_calib, exo_calib, hand_limit, save_path_dict, init_theta=None, init_beta=None, init_th_trans=None, fix_pose = False, fix_shape = False):

    # read the calibration results
    ego_cam_matrix = np.array(ego_calib['ir_mtx']) 
    exo_cam_matrix = np.array(exo_calib['ir_mtx'])
    ego_dist = np.array(ego_calib['ir_dist']) 
    exo_dist = np.array(exo_calib['ir_dist'])
  
    # read the ego pose and 3d pose file
    with open(pose_file, 'r') as json_file:
        ego_pose_info = json.load(json_file)
    left_hand_index = ego_pose_info['left_hand_index'][0]
    hand_landmarks = ego_pose_info['hand_landmarks']
    ego_pose_l = np.array(hand_landmarks[left_hand_index])
    ego_pose_r = np.array(hand_landmarks[1-left_hand_index])
    with open(pose_3d_file, 'r') as json_file:
        ego_pose_3d_info = json.load(json_file)
    ego_pose_l_3d = np.array(ego_pose_3d_info['left'])
    ego_pose_r_3d = np.array(ego_pose_3d_info['right'])

    # read the exo pose file (the left hand corresponds to the right one in ego view)
    with open(pose_exo_file, 'r') as json_file:
        exo_pose_info = json.load(json_file)
    left_hand_index = exo_pose_info['left_hand_index'][0]
    hand_landmarks = exo_pose_info['hand_landmarks']
    exo_pose_l = np.array(hand_landmarks[1-left_hand_index])
    exo_pose_r = np.array(hand_landmarks[left_hand_index])
    
    # read the ego mask file
    ego_mask = np.array(pd.read_csv(mask_file, header=None))
    ego_mask_l = np.where(ego_mask == 1, 1, 0)
    ego_mask_r = np.where(ego_mask == 2, 1, 0)

    # read the exo mask file
    exo_mask = np.array(pd.read_csv(mask_exo_file, header=None))
    exo_mask_l = np.where(exo_mask == 2, 1, 0)
    exo_mask_r = np.where(exo_mask == 1, 1, 0)

    # read the hand point cloud file
    hpc_l = np.fromfile(hpc_L_file, dtype=np.float32).reshape(-1, 3)
    hpc_r = np.fromfile(hpc_R_file, dtype=np.float32).reshape(-1, 3)

    # convert to GPU
    ego_pose_l = torch.from_numpy(ego_pose_l).to('cuda')
    ego_pose_r = torch.from_numpy(ego_pose_r).to('cuda')
    ego_pose_l_3d = torch.from_numpy(ego_pose_l_3d).to('cuda')
    ego_pose_r_3d = torch.from_numpy(ego_pose_r_3d).to('cuda')
    exo_pose_l = torch.from_numpy(exo_pose_l).to('cuda')
    exo_pose_r = torch.from_numpy(exo_pose_r).to('cuda')
    ego_mask_l = torch.from_numpy(ego_mask_l).to('cuda')
    ego_mask_r = torch.from_numpy(ego_mask_r).to('cuda')
    exo_mask_l = torch.from_numpy(exo_mask_l).to('cuda')
    exo_mask_r = torch.from_numpy(exo_mask_r).to('cuda')
    hand_limit_max = torch.from_numpy(np.array(hand_limit['maxThetaVals'])).to('cuda')
    hand_limit_min = torch.from_numpy(np.array(hand_limit['minThetaVals'])).to('cuda')
    ego_cam_matrix = torch.from_numpy(ego_cam_matrix).to('cuda')
    exo_cam_matrix = torch.from_numpy(exo_cam_matrix).to('cuda')
    ego_dist = torch.from_numpy(ego_dist).to('cuda')
    exo_dist = torch.from_numpy(exo_dist).to('cuda')
    transform = torch.from_numpy(transform).to('cuda').reshape(4,4)
    hpc_l = torch.from_numpy(hpc_l).to('cuda')
    hpc_r = torch.from_numpy(hpc_r).to('cuda')
  
    # Initialize MANO layer
    mano_layer_l = ManoLayer(
                    mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', flat_hand_mean=True, side = 'left', ncomps=45, use_pca=False).cuda()
    mano_layer_r = ManoLayer(
                    mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', flat_hand_mean=True, side = 'right', ncomps=45, use_pca=False).cuda()
    # optimize for two hands
    # Initialize the variable you want to optimize  
    optim_list_l = []
    optim_list_r = []
    if init_theta:
        theta_l = init_theta[0].cuda().requires_grad_()
        theta_r = init_theta[1].cuda().requires_grad_()
    else:
        theta_l = torch.zeros(1, 48, requires_grad=True, device = 'cuda')
        theta_r = torch.zeros(1, 48, requires_grad=True, device = 'cuda')
    if init_beta:
        beta_l = init_beta[0].cuda().requires_grad_()
        beta_r = init_beta[1].cuda().requires_grad_()
    else:
        beta_l = torch.zeros(1, 10, requires_grad=True, device = 'cuda')
        beta_r = torch.zeros(1, 10, requires_grad=True, device = 'cuda')
    # init trans for MANO model
    if init_th_trans:
        th_trans_l = init_th_trans[0].cuda().requires_grad_()
        th_trans_r = init_th_trans[1].cuda().requires_grad_()
    else:
        th_trans_l = ego_pose_l_3d.mean(dim=0).unsqueeze(0).requires_grad_()
        th_trans_r = ego_pose_r_3d.mean(dim=0).unsqueeze(0).requires_grad_()
    if not fix_pose:
        optim_list_l.append(theta_l)
        optim_list_r.append(theta_r)
        optim_list_l.append(th_trans_l)
        optim_list_r.append(th_trans_r)
    if not fix_shape:
        optim_list_l.append(beta_l)
        optim_list_r.append(beta_r)
    # Training loop
    if index == -1:
        num_epochs = 500
        init_lr = 0.1
        step_size = 50
    else:
        num_epochs = 300  # 60
        init_lr = 0.1   #0.05
        step_size = 50
    # Create an AdamW optimize    
    optimizer_l = optim.Adam(optim_list_l, lr=init_lr)
    optimizer_r = optim.Adam(optim_list_r, lr=init_lr)
    scheduler_l = lr_scheduler.StepLR(optimizer_l, step_size=step_size, gamma=0.9)
    scheduler_r = lr_scheduler.StepLR(optimizer_r, step_size=step_size, gamma=0.9)
    loss_ls = {
        'slih_l_ls': [],
        'slih_r_ls': [],
        'jo2d_l_ls': [],
        'jo2d_r_ls': [],
        'jo3d_l_ls': [],
        'jo3d_r_ls': [],
        'mesh_l_ls': [],
        'mesh_r_ls': [],
        'limit_l_ls' : [],
        'limit_r_ls' : [],
        'reg_l_ls': [],
        'reg_r_ls': [],
        'shape_l_ls': [],
        'shape_r_ls': []
    }

    for epoch in tqdm(range(num_epochs)):
        hand_verts_l, hand_joints_l = mano_layer_l(theta_l, beta_l, th_trans_l)
        hand_verts_r, hand_joints_r = mano_layer_r(theta_r, beta_r, th_trans_r)
        # Forward pass: Compute the loss
        silh_loss_l = silhouette_error(ego_mask_l, exo_mask_l, hand_verts_l, transform, ego_cam_matrix, exo_cam_matrix)
        silh_loss_r = silhouette_error(ego_mask_r, exo_mask_r, hand_verts_r, transform, ego_cam_matrix, exo_cam_matrix)
        jo2d_loss_l = joint_2d_error(ego_pose_l, exo_pose_l, hand_joints_l, transform, ego_cam_matrix, exo_cam_matrix)
        jo2d_loss_r = joint_2d_error(ego_pose_r, exo_pose_r, hand_joints_r, transform, ego_cam_matrix, exo_cam_matrix)
        jo3d_loss_l = joint_3d_error(ego_pose_l_3d, hand_joints_l)
        jo3d_loss_r = joint_3d_error(ego_pose_r_3d, hand_joints_r)
        mesh_loss_l = mesh_surface_error(hpc_l, hand_verts_l)
        mesh_loss_r = mesh_surface_error(hpc_r, hand_verts_r)
        limit_loss_l = joint_limit_error(theta_l, hand_limit_max, hand_limit_min)
        limit_loss_r = joint_limit_error(theta_r, hand_limit_max, hand_limit_min)
        reg_loss_l = reg_pose_error(theta_l)
        reg_loss_r = reg_pose_error(theta_r)
        shape_loss_l = reg_shape_error(beta_l)
        shape_loss_r = reg_shape_error(beta_r)
        if not fix_shape:
            loss_l =  silh_loss_l + jo2d_loss_l + 0.2 * shape_loss_l +   1e2 * limit_loss_l + 1e2 * mesh_loss_l + 1e3 * jo3d_loss_l
            loss_r =  silh_loss_r + jo2d_loss_r + 0.2 * shape_loss_r +   1e2 * limit_loss_r + 1e2 * mesh_loss_r + 1e3 * jo3d_loss_r
            # loss_l =  silh_loss_l   + 0.2 * shape_loss_l +  limit_loss_l 
            # loss_r =  silh_loss_r   + 0.2 * shape_loss_r +  limit_loss_r 
            # loss_l =  silh_loss_l +  jo2d_loss_l  +  1e2 * jo3d_loss_l + 1e2 * mesh_loss_l 
            # loss_r =  silh_loss_r +  jo2d_loss_r  +  1e2 * jo3d_loss_r + 1e2 * mesh_loss_r 
        else:
            loss_l =  silh_loss_l  + jo2d_loss_l  + 1e2 * limit_loss_l + 1e2 * mesh_loss_l + 1e3 * jo3d_loss_l
            loss_r =  silh_loss_r  + jo2d_loss_r  + 1e2 * limit_loss_r + 1e2 * mesh_loss_r + 1e3 * jo3d_loss_r
            # loss_l =    silh_loss_l + limit_loss_l 
            # loss_r =    silh_loss_r + limit_loss_r 
            # loss_l =  silh_loss_l +  jo2d_loss_l  +  1e2 * jo3d_loss_l + 1e2 * mesh_loss_l 
            # loss_r =  silh_loss_r +  jo2d_loss_r  +  1e2 * jo3d_loss_r + 1e2 * mesh_loss_r 
        loss_ls['slih_l_ls'].append(silh_loss_l.item())
        loss_ls['slih_r_ls'].append(silh_loss_r.item())
        loss_ls['jo2d_l_ls'].append(jo2d_loss_l.item())
        loss_ls['jo2d_r_ls'].append(jo2d_loss_r.item())
        loss_ls['jo3d_l_ls'].append(jo3d_loss_l.item())
        loss_ls['jo3d_r_ls'].append(jo3d_loss_r.item())
        loss_ls['mesh_l_ls'].append(mesh_loss_l.item())
        loss_ls['mesh_r_ls'].append(mesh_loss_r.item())
        loss_ls['limit_l_ls'].append(limit_loss_l.item())
        loss_ls['limit_r_ls'].append(limit_loss_r.item())
        loss_ls['reg_l_ls'].append(reg_loss_l.item())
        loss_ls['reg_r_ls'].append(reg_loss_r.item())
        
        loss_ls['shape_l_ls'].append(shape_loss_l.item())
        loss_ls['shape_r_ls'].append(shape_loss_r.item())

        # Backward pass: Compute gradients
        
        loss_l.backward()
        loss_r.backward()
        # Update the variable using the optimizer
        optimizer_l.step()
        optimizer_r.step()
        # Update learning rate
        scheduler_l.step()
        scheduler_r.step()
        optimizer_l.zero_grad()
        optimizer_r.zero_grad()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss_l: {loss_l.item()}, Loss_r: {loss_r.item()}, lr: {scheduler_l.get_last_lr()[0]},' )
    
    loss_path = os.path.join(save_path_dict['loss'], os.path.splitext(os.path.basename(rgb_file))[0] + '_loss.png')
    draw_losses(loss_ls, loss_path)
    theta_l = theta_l.cpu().detach()
    theta_r = theta_r.cpu().detach()
    beta_l = beta_l.cpu().detach()
    beta_r = beta_r.cpu().detach()
    th_trans_l = th_trans_l.cpu().detach()
    th_trans_r = th_trans_r.cpu().detach()
    dp_path = os.path.join(save_path_dict['ego'], os.path.splitext(os.path.basename(rgb_file))[0] + '_ego.png')
    dp_3d_path = os.path.join(save_path_dict['ego_3D'], os.path.splitext(os.path.basename(rgb_file))[0] + '_ego.png')
    dp_exo_path = os.path.join(save_path_dict['exo'], os.path.splitext(os.path.basename(rgb_exo_file))[0] + '_exo.png')
    left_path = os.path.join(save_path_dict['left_3D'], os.path.splitext(os.path.basename(rgb_file))[0] + '_l.png')
    right_path = os.path.join(save_path_dict['right_3D'], os.path.splitext(os.path.basename(rgb_file))[0] + '_r.png')
    info_path = os.path.join(save_path_dict['info'], os.path.splitext(os.path.basename(rgb_file))[0] + '.json')
    # display_hand({
    #         'verts': hand_verts_l.cpu().detach(),
    #         'joints': hand_joints_l.cpu().detach()
    # },
    #     mano_faces=mano_layer_l.th_faces.cpu().detach(), alpha= 0.9, save_path=left_path)
    # display_hand({
    #         'verts': hand_verts_r.cpu().detach(),
    #         'joints': hand_joints_r.cpu().detach()
    # },
    #     mano_faces=mano_layer_r.th_faces.cpu().detach(), alpha = 0.9, save_path=right_path)
    display_hand_2d(hand_joints_l[0], hand_joints_r[0], torch.eye(4).cuda(), ego_cam_matrix, ego_dist, rgb_file, dp_path)
    # display_hand_2d(hand_joints_l[0], hand_joints_r[0], transform, exo_cam_matrix, exo_dist, rgb_exo_file, dp_exo_path)
    # draw_3d_keypoints(dp_3d_path, hand_joints_l[0].cpu().detach().numpy(), hand_joints_r[0].cpu().detach().numpy())

    info = {'kps3D_L': (hand_joints_l[0]/1000).cpu().detach().numpy().tolist(), 
            'kps3D_R': (hand_joints_r[0]/1000).cpu().detach().numpy().tolist(), 
            'poseCoeff_L': theta_l.numpy().tolist(),
            'poseCoeff_R': theta_r.numpy().tolist(),
            'beta_L': beta_l.numpy().tolist(), 
            'beta_R': beta_r.numpy().tolist(), 
            'trans_L': th_trans_l.numpy().tolist(), 
            'trans_R': th_trans_r.numpy().tolist(), 
            }
    with open(info_path, "w") as json_file:
        json.dump(info, json_file, indent=4)

    # torch.cuda.empty_cache() 
    return [theta_l, theta_r], [beta_l, beta_r], [th_trans_l, th_trans_r]

def hand_optimiztion(root_dir, save_dir, device = 1):

    torch.cuda.set_device(device)

    # Initialize the path
    clips = sorted(os.listdir(root_dir))
    hand_limit_file = '/home/fangqiang/thermal-hand/hand_limits.json'
    for clip in clips[:1]:
        init_beta = None
        init_theta = None
        init_th_trans = None
        # load all types of files
        rgb_path = os.path.join(root_dir, clip, 'egocentric', 'rgb')
        rgb_exo_path = os.path.join(root_dir, clip, 'exocentric', 'rgb')
        mask_path = os.path.join(save_dir, clip, 'ego', 'mask_2d')
        hpc_L_path = os.path.join(save_dir, clip, 'ego', 'hand_pcd_L')
        hpc_R_path = os.path.join(save_dir, clip, 'ego', 'hand_pcd_R')
        pose_path = os.path.join(save_dir, clip, 'ego', 'pose_2d')
        pose_3d_path = os.path.join(save_dir, clip, 'ego', 'pose_3d_tri')
        pose_exo_path = os.path.join(save_dir, clip, 'exo', 'pose_2d')
        mask_exo_path = os.path.join(save_dir, clip, 'exo', 'mask_2d')
        transform_path = os.path.join(save_dir, clip, 'exo_ego_transform.csv')
        save_path_dict = {'ego': os.path.join(save_dir, clip, 'gt_pose_ego'), 
                          'exo': os.path.join(save_dir, clip, 'gt_pose_exo'),
                          'left_3D': os.path.join(save_dir, clip, 'gt_pose_left3D'),
                          'right_3D': os.path.join(save_dir, clip, 'gt_pose_right3D'),
                          'ego_3D': os.path.join(save_dir, clip, 'gt_pose_ego_3D'),
                          'info':  os.path.join(save_dir, clip, 'gt_info'),
                          'loss': os.path.join(save_dir, clip, 'gt_loss'),
                          }
        for key in save_path_dict:
            if not os.path.exists(save_path_dict[key]):
                os.makedirs(save_path_dict[key])
       
        ego_calib_file = "/mnt/data/MultimodalEgoHands/calibration/ego_calib.json"
        exo_calib_file = "/mnt/data/MultimodalEgoHands/calibration/exo_calib.json"
        rgb_files = sorted(glob(rgb_path + '/'+ '*.png'))
        rgb_exo_files = sorted(glob(rgb_exo_path + '/'+ '*.png'))
        mask_files = sorted(glob(mask_path + '/'+ '*.csv'))
        hpc_L_files = sorted(glob(hpc_L_path + '/'+ '*.bin'))
        hpc_R_files = sorted(glob(hpc_R_path + '/'+ '*.bin'))
        pose_files = sorted(glob(pose_path + '/'+ '*.json'))
        pose_3d_files = sorted(glob(pose_3d_path + '/'+ '*.json'))
        mask_exo_files = sorted(glob(mask_exo_path + '/'+ '*.csv'))
        pose_exo_files = sorted(glob(pose_exo_path + '/'+ '*.json'))
        # Load the calibrated camera matrices and distortion coefficients
        # These matrices should have been obtained during camera calibration
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        with open(exo_calib_file, 'r') as file:
            exo_calib = json.load(file)
        with open(hand_limit_file, 'r') as file:
            hand_limit = json.load(file)
        transforms = np.genfromtxt(transform_path, delimiter=',')
        # rgb_camera_matrix = np.array(ego_calib['RGB_Camera_matrix'])  # Intrinsic matrix for RGB camera
        # rgb_distortion_coeffs = np.array(ego_calib['RGB_dist'])  # Distortion coefficients for RGB camera
        # depth_camera_matrix = np.array(ego_calib['IR_Camera_matrix'])  # Intrinsic matrix for depth camera
        # depth_distortion_coeffs = np.array(ego_calib['IR_dist'])  # Distortion coefficients for depth camera
        # rotation_matrix = np.array(ego_calib['rgb2ir_rmatrix'])  # Extrinsic rotation matrix
        # translation_vector = np.array(ego_calib['rgb2ir_tvecs'])  # Extrinsic translation vector
        if not init_beta:
            init_theta, init_beta, init_th_trans = optimize_frame_pose(-1, transforms[0], mask_files[0], hpc_L_files[0], hpc_R_files[0], pose_files[0], pose_3d_files[0], mask_exo_files[0], pose_exo_files[0], rgb_files[0], rgb_exo_files[0], ego_calib, exo_calib, hand_limit, save_path_dict, None, None, None, fix_pose = False, fix_shape = False)
        # _, init_beta, _ = optimize_frame_pose(0, transforms[0], mask_files[0], hpc_L_files[0], hpc_R_files[0], pose_files[0], pose_3d_files[0], mask_exo_files[0], pose_exo_files[0], rgb_files[0], rgb_exo_files[0], ego_calib, exo_calib, hand_limit, save_path_dict, init_theta, init_beta, init_th_trans, fix_pose = True, fix_shape = False)
        for i in tqdm(range(len(rgb_files)), desc = root_dir.split('/')[-1] + clip):
            init_theta, _, init_th_trans = optimize_frame_pose(i, transforms[i], mask_files[i], hpc_L_files[i], hpc_R_files[i], pose_files[i], pose_3d_files[i], mask_exo_files[i], pose_exo_files[i], rgb_files[i], rgb_exo_files[i], ego_calib, exo_calib, hand_limit, save_path_dict, init_theta, init_beta, init_th_trans, fix_pose=False, fix_shape = True)
            # init_theta, _, init_th_trans = optimize_frame_pose(i, transforms[i], mask_files[i], hpc_L_files[i], hpc_R_files[i], pose_files[i], pose_3d_files[i], mask_exo_files[i], pose_exo_files[i], rgb_files[i], rgb_exo_files[i], ego_calib, exo_calib, hand_limit, save_path_dict, None, init_beta, None, fix_pose=False, fix_shape = True)
            # if i%5 == 0: 
            torch.cuda.empty_cache()
if __name__ == "__main__":
    root_dir = '/mnt/data/MultimodalEgoHands/subject_03/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_03/'
    hand_optimiztion(root_dir, save_dir)
