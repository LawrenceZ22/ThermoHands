import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import time
import json
import torch
import mediapipe as mp
from segment_anything import SamPredictor, sam_model_registry
from pathos.multiprocessing import ProcessingPool as Pool


# Function to process an image and estimate hand pose, mask
def process_image_id(img_file, predictor, vis_pose_path, vis_mask_path, save_pose_path, save_mask_path, gc_iteration):

    cam_id = img_file.split('.')[-2].split('_')[0]
    # Read the image
    image = cv2.imread(img_file)
    image2 = image.copy()
    
    # Define background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image using MediaPipe Hands
    results = HANDS.process(image_rgb)
    
    # Create a dictionary to store the hand pose results
    hand_pose_data = {"image_path": img_file, "hand_landmarks": []}

    mask_ls = []

    if results.multi_hand_landmarks:
        # If multiple hands are detected, you can iterate through them
        for landmarks in results.multi_hand_landmarks:
            
            # Define a mask to initialize GrabCut
            mask = np.zeros(image2.shape[:2], np.uint8)
            # Extract landmark coordinates
            landmark_points = []
            # landmarks contain the hand pose information
            # You can access specific landmarks by index
            for idx, landmark in enumerate(landmarks.landmark):
                # Access the x, y coordinates of the landmark
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmark_points.append((x, y))
                landmark_data = [{"x": landmark.x, "y": landmark.y, "x_coordinate": x, "y_coordinate": y}]
                hand_pose_data["hand_landmarks"].append(landmark_data)
                # Draw a circle at the landmark location
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            if not predictor == None:
                # Prompt the mask with keypoints around the hand
                predictor.set_image(image)
                mask, _, _ = predictor.predict(point_coords=np.array(landmark_points), point_labels = np.ones(len(landmark_points)),multimask_output=False,)
                mask2 = mask[0].astype('uint8')
            else:
                # Initialize the mask with a rectangle around the hand
                scale = 0.05
                # Convert keypoints to a rect (region of interest)
                x, y, w, h = cv2.boundingRect(np.array(landmark_points))  # Assuming keypoints are in (x, y) format    
                x = int(x - scale * w)
                y = int(y - scale * h)
                w = int(w * (1 + 2 * scale))
                h = int(h * (1 + 2 * scale))    
                rect = (x, y, w, h)
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, gc_iteration, cv2.GC_INIT_WITH_RECT)
                # Modify the mask to make it binary
                mask2 = np.where((mask == 2) | (mask == 0) | (mask == 1), 0, 1).astype('uint8')

            mask_ls.append(mask2)

            # Draw lines connecting hand landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (5, 6), (6, 7), (7, 8),
                (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16),
                (17, 18), (18, 19), (19, 20)
            ]
            for connection in connections:
                point1 = landmark_points[connection[0]]
                point2 = landmark_points[connection[1]]
                cv2.line(image, point1, point2, (0, 255, 0), 2)

    # Save the annotated image to the output directory - pose
    output_path = os.path.join(vis_pose_path, cam_id, os.path.basename(img_file))
    cv2.imwrite(output_path, image)

    # aggregate the mask from multi-hands
    mask_agg = np.zeros(image2.shape[:2], np.uint8)
    for mask in mask_ls:
        mask_agg = np.where((mask_agg ==1)| (mask==1), 1, 0).astype('uint8')
    # remove the green background misclassfied
    # Define the HSV range for green color (adjust the range as needed)
    lower_green = np.array([35, 50, 50])  # Lower bound for green
    upper_green = np.array([85, 255, 255])  # Upper bound for green
    # Create a mask for the green background
    # Convert the image to HSV
    image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(image2_hsv, lower_green, upper_green)
    mask_agg = np.where((mask_agg ==1) & (green_mask==0), 1, 0).astype('uint8')
    image2 *= mask_agg[:, :, np.newaxis]
    # Save the annotated image to the output directory - mask
    output_path = os.path.join(vis_mask_path, cam_id, os.path.basename(img_file))
    cv2.imwrite(output_path, image2)
        
    # Save the hand pose data to a JSON file
    output_path = os.path.join(save_pose_path, cam_id, os.path.splitext(os.path.basename(img_file))[0] + ".json")
    with open(output_path, "w") as json_file:
        json.dump(hand_pose_data, json_file, indent=4)

    # Save the hand mask data to csv file
    output_path =  os.path.join(save_mask_path, cam_id, os.path.splitext(os.path.basename(img_file))[0] + ".csv")
    np.savetxt(output_path, mask_agg, fmt='%d', delimiter=',')


def main():

    ##################### Estimate and save the hand pose and mask from the allo-centric RGB images #####################
    multi_process = False
    if multi_process:
        pool = Pool(4) 

    num_cams = 12
    gc_iteration = 2
    # Initialize the SAM module
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_b"](checkpoint="/home/fangqiang/thermal-hand/sam/sam_vit_b_01ec64.pth").to(device=DEVICE)
    predictor = SamPredictor(sam)

    root_dir = '/mnt/data/lawrence/ThermalHands/'
    save_dir = '/mnt/data/TherHandsPro/'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        vis_pose_path = os.path.join(save_dir, clip, 'vis_allo_pose')
        vis_mask_path = os.path.join(save_dir, clip, 'vis_allo_mask')
        save_pose_path = os.path.join(save_dir, clip, 'allo_pose')
        save_mask_path = os.path.join(save_dir, clip, 'allo_mask')
        if not os.path.exists(vis_pose_path):
            os.makedirs(vis_pose_path)
        if not os.path.exists(vis_mask_path):
            os.makedirs(vis_mask_path)
        if not os.path.exists(save_pose_path):
            os.makedirs(save_pose_path)
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)
        for j in range(1,num_cams+1):
            j = str(j)
            if not os.path.exists(os.path.join(vis_pose_path,j.zfill(2))):
                os.makedirs(os.path.join(vis_pose_path,j.zfill(2)))
            if not os.path.exists(os.path.join(vis_mask_path,j.zfill(2))):
                os.makedirs(os.path.join(vis_mask_path,j.zfill(2)))
            if not os.path.exists(os.path.join(save_pose_path,j.zfill(2))):
                os.makedirs(os.path.join(save_pose_path,j.zfill(2)))
            if not os.path.exists(os.path.join(save_mask_path,j.zfill(2))):
                os.makedirs(os.path.join(save_mask_path,j.zfill(2)))
            
        img_path = os.path.join(path, 'allocentric')
        frames = sorted(os.listdir(img_path))
        # Prepare for the parellel processing for multiple views
        if multi_process:
            zip_predictor = [predictor for i in range((num_cams))]
            zip_vis_pose_path = [vis_pose_path for i in range((num_cams))]
            zip_vis_mask_path = [vis_mask_path for i in range((num_cams))]
            zip_save_pose_path = [save_pose_path for i in range((num_cams))]
            zip_save_mask_path = [save_mask_path for i in range((num_cams))]
            zip_gc_iteration = [gc_iteration for i in range((num_cams))]
            # zip_hands = [mp.solutions.hands.Hands() for i in range(num_cams)]
        for frame in tqdm(frames[::6], desc="estimating allocentric hand pose and mask for " + clip):
            frame_path = os.path.join(img_path, frame)
            img_files = sorted(glob(frame_path + '/'+ '*_C.jpg'))
            if multi_process:
                # pool.restart()
                pool.imap(process_image_id, img_files, zip_predictor, zip_vis_pose_path, zip_vis_mask_path, zip_save_pose_path, zip_save_mask_path, zip_gc_iteration)
                pool.close()
                pool.join()
            else:
                for img_file in tqdm(img_files):
                    process_image_id(img_file, predictor, vis_pose_path, vis_mask_path, save_pose_path, save_mask_path, gc_iteration)


    # Release resources
    HANDS.close()
                

if __name__ == "__main__":
    mp_hands = mp.solutions.hands
    HANDS = mp_hands.Hands()
    main()
