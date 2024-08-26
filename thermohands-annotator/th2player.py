import os
import yaml
import subprocess
import torch
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm
import argparse
from glob import glob
import copy
import moviepy.video.io.ImageSequenceClip
import time
from scipy.spatial.transform import Rotation as R
from manopth.manolayer import ManoLayer

def viz_hand_meshed(left_hand, right_hand, vis_path, cam_param):

    read_view = True
    save_view = (not read_view)
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='hand', width=640, height=480)  

    for i, left, right in tqdm(zip(range(len(left_hand)), left_hand, right_hand)):
        
        left.compute_vertex_normals()
        right.compute_vertex_normals()
        vis.add_geometry(left)
        vis.add_geometry(right)
        ctr = vis.get_view_control()
        if read_view:
            # param = o3d.io.read_pinhole_camera_parameters('/home/fangqiang/thermal-hand/origin_hand_view.json')
            # ctr.convert_from_pinhole_camera_parameters(param)      
            ctr.convert_from_pinhole_camera_parameters(cam_param, True) 
        vis.poll_events()
        vis.update_renderer()
        # vis.run()
        vis.capture_screen_image(os.path.join(vis_path, '{}.png'.format(str(i).zfill(5))))
        if save_view:
            param = ctr.convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('/home/fangqiang/thermal-hand/origin_hand_view.json', param)
            break
        vis.clear_geometries()

    vis.destroy_window()

    
def load_hand_meshes(hand_path=[]):
    """
    load pointcloud by path and down sample (if True) based on voxel_size 
    """
    left_hand = [o3d.geometry.TriangleMesh()for _ in range(len(hand_path))]
    right_hand = [o3d.geometry.TriangleMesh()for _ in range(len(hand_path))]
    ncomps = 45
    torch.set_default_tensor_type('torch.DoubleTensor')
    mano_right = ManoLayer(
        mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', ncomps=45, flat_hand_mean=True, side='right', use_pca=False)
    mano_left = ManoLayer(
        mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', ncomps=45, flat_hand_mean=True, side='left', use_pca=False)
    for idx in tqdm(range(len(hand_path))):

        hand_pose = {}
        with open(hand_path[idx], 'r') as json_file:
            load_hand = json.load(json_file)
        hand_pose["left_pose"] = load_hand['poseCoeff_L']
        hand_pose["left_tran"] = load_hand['trans_L']
        hand_pose["left_shape"] = load_hand['beta_L']
        hand_pose["right_pose"] = load_hand['poseCoeff_R']
        hand_pose["right_tran"] = load_hand['trans_R']
        hand_pose["right_shape"] = load_hand['beta_R']

        random_pose = torch.tensor(hand_pose["left_pose"])
        random_tran = torch.tensor(hand_pose["left_tran"])
        random_shape = torch.tensor(hand_pose["left_shape"])

        hand_verts, hand_joints = mano_left(random_pose, random_shape)
        hand_verts_scaled = hand_verts/1000.0 + random_tran
        triangles = mano_left.th_faces
        mesh_lh = o3d.geometry.TriangleMesh()
        mesh_lh.vertices = o3d.utility.Vector3dVector(
            hand_verts_scaled.detach().numpy()[0])
        mesh_lh.triangles = o3d.utility.Vector3iVector(
            np.asarray(triangles)[:len(triangles):])
        # Skin color
        mesh_lh.vertex_colors = o3d.utility.Vector3dVector(np.reshape(
            [209/255., 163/255., 164/255.]*np.shape(mesh_lh.vertices)[0], (-1, 3)))

        random_pose = torch.tensor(hand_pose["right_pose"])
        random_tran = torch.tensor(hand_pose["right_tran"])
        random_shape = torch.tensor(hand_pose["right_shape"])
        hand_verts, hand_joints = mano_right(random_pose, random_shape)
        hand_verts_scaled = hand_verts/1000.0 + random_tran
        hand_joints_scaled = hand_joints/1000.0 + random_tran
        triangles = mano_right.th_faces
        mesh_rh = o3d.geometry.TriangleMesh()
        mesh_rh.vertices = o3d.utility.Vector3dVector(
            hand_verts_scaled.detach().numpy()[0])
        mesh_rh.triangles = o3d.utility.Vector3iVector(
            np.asarray(triangles)[:len(triangles):])
        # Skin color
        mesh_rh.vertex_colors = o3d.utility.Vector3dVector(np.reshape(
            [209/255., 163/255., 164/255.]*np.shape(mesh_rh.vertices)[0], (-1, 3)))

        left_hand[idx] = mesh_lh
        right_hand[idx] = mesh_rh

    return left_hand, right_hand

def get_camera_param(ego_calib):

    # Create PinholeCameraParameters object
    camera_params = o3d.camera.PinholeCameraParameters()

    # Set the intrinsic parameters
    ins_mtx = np.array(ego_calib['ir_mtx'])
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_params.intrinsic.set_intrinsics(width=640, height=480, fx=ins_mtx[0,0], fy=ins_mtx[1,1], cx=ins_mtx[0,2], cy=ins_mtx[1,2])

    # Set the extrinsic parameters (assuming R is the rotation matrix and t is the translation vector)
    camera_params.extrinsic = np.eye(4) # Start with an identity matrix

    return camera_params

def main():

    sub_ls = ['01_kitchen', '04_kitchen']
    # sub_ls = sorted(['01_gesture', '02_gesture', '04_gesture', '06_gesture', '13_gesture',\
    #                 '21_gesture', '23_gesture','24_gesture', '25_gesture', '26_gesture', \
    #                 '27_gesture', '28_gesture', '29_gesture', '30_gesture', '31_gesture',\
    #                 '32_gesture', '33_gesture', '34_gesture', '35_gesture', '36_gesture',\
    #                 '37_gesture',  '01',  '02',  '03', '04', '06', '11', '13', '16', '18', \
    #                 '19', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', \
    #                 '36', '37', '01_kitchen', '04_kitchen']) #'20', '21', '22', '23', 
    root_dir = '/mnt/data/MultimodalEgoHands/' 
    root_dir_2 = '/mnt/data/MultimodalEgoHands/gestures/'  
    save_dir = '/mnt/data/fangqiang/TherHandsPro/'
    vis_dir = '/mnt/data/fangqiang/TherHands-mesh/'
    ego_calib_file = "/mnt/data/MultimodalEgoHands/calibration/ego_calib.json"
    for sub in sub_ls:
        root_path = os.path.join(root_dir, 'subject_' + sub)
        if not os.path.exists(root_path):
            root_path = os.path.join(root_dir_2, 'subject_' + sub)
        save_path = os.path.join(save_dir, 'subject_' + sub)
        vis_path = os.path.join(vis_dir, 'subject_' + sub)
        clips =  sorted(os.listdir(root_path))
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        for clip in clips:
            clip_path = os.path.join(save_path, clip)
            gt_path = os.path.join(clip_path, 'gt_info')
            vis_clip_path = os.path.join(vis_path, clip, 'gt_ego_mesh')
            gt_files = sorted(glob(gt_path + '/' + '*.json'))
            if not os.path.exists(vis_clip_path):
                os.makedirs(vis_clip_path)
            cam_param = get_camera_param(ego_calib)
            left_hand, right_hand = load_hand_meshes(gt_files)
            viz_hand_meshed(left_hand, right_hand, vis_clip_path, cam_param)
            vis_files = sorted(glob(vis_clip_path + '/'+ '*.png'))
            if (len(vis_files)>0):
                Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(vis_files, fps=8.5)
                Imgclip.write_videofile(os.path.join(vis_path, clip, 'mv_gt_mesh.mp4'))
                



if __name__ == "__main__":
    main()
