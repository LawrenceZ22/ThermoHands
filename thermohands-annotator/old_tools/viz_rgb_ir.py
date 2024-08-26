import os
import numpy as np
from glob import glob
from tqdm import tqdm
import tifffile as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import cv2
import time

def main():

    root_dir = '/mnt/data/lawrence/ThermalHands/'
    save_dir = '/mnt/data/TherHandsPro/'
    clips = sorted(os.listdir(root_dir))
    for clip in ['read_book_1225']: 
        path = os.path.join(root_dir, clip)
        vis_path = os.path.join(save_dir, clip, 'vis_rgb')
        vis_path_ir = os.path.join(save_dir, clip, 'vis_ir')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        if not os.path.exists(vis_path_ir):
            os.makedirs(vis_path_ir)
        img_path = os.path.join(path, 'egocentric', 'rgb')
        ir_path = os.path.join(path, 'egocentric', 'ir')
        img_files = sorted(glob(img_path + '/'+ '*.png'))
        ir_files = sorted(glob(ir_path + '/'+ '*.png'))
        for img_file, ir_file in tqdm(zip(img_files,ir_files), total = len(img_files)):
            # Open the depth image 
            img = cv2.imread(img_file)
            ir = cv2.imread(ir_file)
            cv2.imwrite(os.path.join(vis_path, img_file.split('/')[-1].split('.')[-2] + '.png'), img)
            cv2.imwrite(os.path.join(vis_path_ir, ir_file.split('/')[-1].split('.')[-2] + '.png'), ir)
            # plt.imshow(img)
            # plt.title('RGB Camera Image [m]')
            # plt.axis('off')  # Turn off axis labels
            # plt.savefig(os.path.join(vis_path, img_file.split('/')[-1].split('.')[-2] + '.png'))
            # plt.close()
            # plt.clf()





if __name__ == "__main__":
    main()

