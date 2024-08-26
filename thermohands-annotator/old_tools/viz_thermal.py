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
        vis_path = os.path.join(save_dir, clip, 'vis_thermal')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        img_path = os.path.join(path, 'egocentric', 'thermal')
        img_files = sorted(glob(img_path + '/'+ '*.tiff'))
        for img_file in tqdm(img_files):
            # Open the thermal image in TIFF format
            img = cv2.imread(img_file)
            cv2.imwrite(os.path.join(vis_path, img_file.split('/')[-1].split('.')[-2] + '.png'), img)
            # plt.imshow(img, cmap='jet')
            # plt.colorbar()
            # plt.title('Thermal Camera Image')
            # plt.axis('off')  # Turn off axis labels
            # plt.savefig(os.path.join(vis_path, img_file.split('/')[-1].split('.')[-2] + '.png'))
            # plt.close()
            # plt.clf()





if __name__ == "__main__":
    main()

