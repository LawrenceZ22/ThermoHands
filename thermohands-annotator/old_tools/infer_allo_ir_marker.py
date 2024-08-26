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


def process_ir(image_path, vis_path, save_path, threshold = 210, area_threshold = [4, 36]):

    # Load the IR image
    ir_image = cv2.imread(image_path)
    cam_id = image_path.split('.')[-2].split('_')[0]
    # Convert the image to grayscale
    ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    # Create a dictionary to store the hand pose results
    marker_info = {"image_path": image_path, "markers": []}
    # Create a black rectangle image
    win_x, win_y, win_h, win_w = 260, 180, 240, 240
    black_rectangle = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    # Paste the black rectangle onto the original image
    ir_gray[win_y:win_y+win_h, win_x:win_x+win_w] = black_rectangle[:,:,0]
    # ir_image[win_y:win_y+win_h, win_x:win_x+win_w] = black_rectangle
     # Apply binary thresholding
    _, binary_image = cv2.threshold(ir_gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area> area_threshold[0]) & (area < area_threshold[1]):
            # Do something with the marker contour, e.g., draw a bounding box
            x, y, w, h = cv2.boundingRect(contour)
            center = [x+w/2, y+h/2]
            cv2.rectangle(ir_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            marker_info["markers"].append({"x": center[0] , "y": center[1]})

    # Save the annotated image to the output directory - marker
    output_path = os.path.join(vis_path, cam_id, os.path.basename(image_path))
    cv2.imwrite(output_path, ir_image)

    # Save the marker 2d data to a JSON file
    output_path = os.path.join(save_path, cam_id, os.path.splitext(os.path.basename(image_path))[0] + ".json")
    with open(output_path, "w") as json_file:
        json.dump(marker_info, json_file, indent=4)

    
    
def main():

    ##################### Do not run this code, too much false postives. Instead mannually annotate. #####################
    num_cams = 12
    root_dir = '/mnt/data/lawrence/ThermalHands/'
    save_dir = '/mnt/data/TherHandsPro/'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        vis_marker_path = os.path.join(save_dir, clip, 'vis_allo_marker')
        save_marker_path = os.path.join(save_dir, clip, 'allo_marker')
        if not os.path.exists(vis_marker_path):
            os.makedirs(vis_marker_path)
        if not os.path.exists(save_marker_path):
            os.makedirs(save_marker_path)
        for j in range(1,num_cams+1):
            j = str(j)
            if not os.path.exists(os.path.join(vis_marker_path,j.zfill(2))):
                os.makedirs(os.path.join(vis_marker_path,j.zfill(2)))
            if not os.path.exists(os.path.join(save_marker_path,j.zfill(2))):
                os.makedirs(os.path.join(save_marker_path,j.zfill(2)))
        # read the ir file and detect the reflective markers
        ir_path = os.path.join(path, 'allocentric')
        frames = sorted(os.listdir(ir_path))
        for frame in tqdm(frames[::6], desc="detecting allocentric markers for " + clip):
            frame_path = os.path.join(ir_path, frame)
            img_files = sorted(glob(frame_path + '/'+ '*_A.png'))
            for img_file in tqdm(img_files):
                process_ir(img_file,vis_marker_path, save_marker_path)
        
        
    

if __name__ == '__main__':
    main()