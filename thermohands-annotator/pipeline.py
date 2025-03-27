import os
import yaml
import subprocess
from tools.vis_ego_exo_data import vis_ego_exo_data
from tools.infer_ego_depth_point_cloud import infer_ego_point_cloud
from tools.viz_kiss_icp_results import viz_kiss_icp_res
from tools.infer_ego_exo_ir_marker import infer_ego_exo_markers
from tools.infer_ego_exo_transform import infer_transform_matrix
from tools.infer_ego_exo_pose_mask import infer_pose_mask
from tools.infer_kps_3d_triangulation import infer_3D_kps
from tools.infer_hand_point_cloud import infer_hand_pcd
from tools.optimize_hand_pose import hand_optimiztion
from tools.vis_ego_exo_anno import movie_annotation
from tools.infer_gt_hand_mask import infer_hand_mask_gt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Thermal Hand Annotation Pipeline")
    parser.add_argument('--subs', type=str, nargs='+', default=[
        "01", "02", "03", "04", "01_gesture", "02_gesture", "03_gesture", "04_gesture"
    ], help="List of subject folders to process")
    parser.add_argument('--root_dir', type=str, default='/mnt/data/MultimodalEgoHands/', help="Root directory of raw data")
    parser.add_argument('--root_dir_2', type=str, default='/mnt/data/MultimodalEgoHands/gestures/', help="Secondary root directory for gesture data")
    parser.add_argument('--save_dir', type=str, default='/mnt/data/fangqiang/TherHandsPro/', help="Directory to save results")
    return parser.parse_args()

def main():
    args = parse_args()

    sub_ls = sorted(args.subs)
    root_dir = args.root_dir
    root_dir_2 = args.root_dir_2
    save_dir = args.save_dir
    kissicp_sh = args.kissicp_sh

    print("Subjects to process:", sub_ls)
    print("Root dir:", root_dir)
    print("Root2 dir:", root_dir_2)
    print("Save dir:", save_dir)

    kissicp_sh = '../tools/run_kissicp.sh'
    for sub in sub_ls:
        kissicp_cfg = '../tools/kiss_icp_config.yaml'
        root_path = os.path.join(root_dir, 'subject_' + sub)
        if not os.path.exists(root_path):
            root_path = os.path.join(root_dir_2, 'subject_' + sub)
        save_path = os.path.join(save_dir, 'subject_' + sub)
        odom_path = os.path.join(save_dir, 'subject_' + sub + '_odom')
        cfg_path = '../tools/kiss_icp_cfgs/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(cfg_path):
            os.makedirs(cfg_path)
        if not os.path.exists(odom_path):
            os.makedirs(odom_path)
        with open(kissicp_cfg) as file:
            data = yaml.full_load(file)
        data['out_dir'] = odom_path
        kissicp_cfg = os.path.join(cfg_path, kissicp_cfg.split('/')[-1].split('.')[-2] + '_' + sub + '.yaml')
        with open(kissicp_cfg, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
        ################################################
        ##### Step 1 - visualize all capture data ######
        ################################################
        print('***** Start to visualize ego and exo data *****')
        vis_ego_exo_data(root_path, save_path)


        ###################################################################
        ##### Step 2 - infer point cloud from egocentric depth image ######
        ##### and run KISS-ICP to obtian the odometry and visualize it ####
        ##################
            
        print('***** Start to infer point cloud from ego depth *****')
        infer_ego_point_cloud(root_path, save_path)
        print('***** Start to run KISS-ICP for ego odometry *****')
        subprocess.run([kissicp_sh, save_path, odom_path, kissicp_cfg], text=True)
        print('***** Start to visualize KISS-ICP results *****')
        viz_kiss_icp_res(root_path, save_path, odom_path)
        
        ####################################################################
        ###### Step 3 - annotate markers for the first frames ##############
        ###### and calculate transformation between two views ##############
        ####################################################################
        # please comment the following lines when run remotely or you have annotated the markers
        print('***** Start to annotate markers *****')
        infer_ego_exo_markers(root_path, save_path)
        print('***** Start to infer the ego-exo transformation matrix *****')
        infer_transform_matrix(root_path, save_path, odom_path)

        ####################################################################
        ###### Step 4 - infer 2D hand pose and mask, infer 3D hand pose ####
        ###### via triangulation and hand point cloud via cropping #########
        ####################################################################
    
        print('***** Start to infer 2D hand pose and mask *****')
        infer_pose_mask(root_path, save_path, 'cuda:1')
        print('***** Start to crop hand point cloud *****')
        infer_hand_pcd(root_path, save_path)

        print('***** Start to triangulate 3D hand pose *****')
        infer_3D_kps(root_path, save_path)
        ###################################################################
        ##### Step 5 - optimize the 3D hand pose by fitting MANO model ####
        ###################################################################
        print('***** Start the optimization for 3D hand pose *****')
        hand_optimiztion(root_path, save_path, 0)

        ####################################################################
        ###### Step 6 - generate 2D mask ground truth based on GT ##########
        ####################################################################
        print('***** Start to generate 2D mask ground truth *****')
        infer_hand_mask_gt(root_path, save_path)
        
        ###################################################################
        ##### Step 7 - make annotations, e.g., pose, mask, movies #########
        ###################################################################
        print('***** Start to make annotation movies *****')
        movie_annotation(root_path, save_path)


if __name__ == "__main__":
    main()
