import os
import numpy as np
import open3d as o3d
import json
import cv2
from tqdm import tqdm
import pandas as pd
import scipy.io as scio
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d
from glob import glob


    
def main():

    root_dir = '/mnt/data/lawrence/ThermalHands/'
    save_dir = '/mnt/data/TherHandsPro/'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        marker_path = os.path.join(save_dir, clip, 'ego_marker')
        depth_path = os.path.join(root_dir, clip, 'egocentric', 'depth')
        marker_files = sorted(os.listdir(marker_path))
        depth_files = sorted(os.listdir(depth_path))
        #  read the egocentric calibration file
        ego_calib_file = os.path.join(path, 'egocentric', 'egocentric_calib.json')
        # Open the file in read mode ('r')
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        rgb2ir_R= ego_calib['rgb2ir_rmatrix']
        rgb2ir_t = ego_calib['rgb2ir_tvecs']
        rgb2ir = np.eye(4)
        rgb2ir[:3, :3] = rgb2ir_R
        rgb2ir[:3,  3] = rgb2ir_t
        for marker_file in marker_files:
            marker_file = os.path.join(marker_path, marker_file)
            with open(marker_file, 'r') as json_file:
                marker_rgb = json.load(json_file)['markers']
            depth_file = os.path.join(depth_path, depth_file)
            depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            
        


if __name__ == '__main__':
    main()