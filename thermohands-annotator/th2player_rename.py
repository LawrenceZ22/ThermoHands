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
import cv2
import shutil


def main():

   
    src_dir = '/mnt/data/fangqiang/TherHands-mesh/'
    tgt_dir = '/mnt/data/fangqiang/TherHands-mesh-new/'
    ref_dir = '/mnt/data/fangqiang/TherHandsPro/'
    sub_ls = sorted(os.listdir(src_dir))
    for sub in sub_ls[8:9]:
        root_path = os.path.join(src_dir, sub)
        save_path = os.path.join(tgt_dir, sub)
        ref_path = os.path.join(ref_dir, sub)
        clips =  sorted(os.listdir(root_path))
        for clip in clips:
            root_clip_path = os.path.join(root_path, clip, 'gt_ego_mesh')
            save_clip_path = os.path.join(save_path, clip, 'gt_ego_mesh')
            ref_clip_path = os.path.join(ref_path, clip, 'gt_info')
            if not os.path.exists(save_clip_path):
                os.makedirs(save_clip_path)
            mesh_files = sorted(glob(root_clip_path + '/' + '*.png'))
            info_files = sorted(glob(ref_clip_path + '/' + '*.json'))
            assert len(mesh_files) == len(info_files)
            for mesh_file, info_file in tqdm(zip(mesh_files, info_files), total = len(mesh_files)):
                new_file = save_clip_path + '/' + info_file.split('/')[-1].split('.')[0] + '.png'
                shutil.copyfile(mesh_file, new_file)
                



if __name__ == "__main__":
    main()
