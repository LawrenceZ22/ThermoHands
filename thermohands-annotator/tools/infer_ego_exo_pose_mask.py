import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import json
import mediapipe as mp
from segment_anything import SamPredictor, sam_model_registry


# Function to process an image and estimate hand pose, mask
def process_image(image_path, hands, predictor, vis_pose_path, vis_mask_path, save_pose_path, save_mask_path, gc_iteration, ego_calib, last_pose_data, last_mask_ls, kitchen=False):
    
    ir_camera_matrix = np.array(ego_calib['ir_mtx'])  # Intrinsic matrix for IR camera
    ir_distortion_coeffs = np.array(ego_calib['ir_dist'])  # Distortion coefficients for IR camera
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.undistort(image, ir_camera_matrix, ir_distortion_coeffs)
    image2 = image.copy()
    
    # Define background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image using MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Create a dictionary to store the hand pose results
    hand_pose_data = {"hand_landmarks": [], "left_hand_index": []}

    mask_ls = []
    center_hand = []
    if results.multi_hand_landmarks:
        # If multiple hands are detected, you can iterate through them
        # if len(results.multi_hand_landmarks)>2:
        #     print('stop here')
        for landmarks in results.multi_hand_landmarks[:2]:
            # Define a mask to initialize GrabCut
            # mask = np.zeros(image2.shape[:2], np.uint8)
            # Extract landmark coordinates
            landmark_points = []
            # landmarks contain the hand pose information
            # You can access specific landmarks by index
            for _, landmark in enumerate(landmarks.landmark):
                # Access the x, y coordinates of the landmark
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmark_points.append((x, y))

            hand_pose_data["hand_landmarks"].append(landmark_points)
            # compute the center point of the hand
            center_hand.append(np.array(landmark_points).mean(axis=0))
            # Convert keypoints to a rect (region of interest)
            scale = 0.05
            x, y, w, h = cv2.boundingRect(np.array(landmark_points))  # Assuming keypoints are in (x, y) format    
            x = int(x - scale * w)
            y = int(y - scale * h)
            w = int(w * (1 + 2 * scale))
            h = int(h * (1 + 2 * scale))    
            rect = (x, y, w, h)
            input_box = np.array([x, y, x+w, y+h])
    
            if not predictor == None:
                # Prompt the mask with keypoints around the hand
                predictor.set_image(image)
                # mask, _, _ = predictor.predict(point_coords=np.array(landmark_points), point_labels = np.ones(len(landmark_points)),multimask_output=False)
                # mask, _, _ = predictor.predict(
                #                 point_coords=None,
                #                 point_labels=None,
                #                 box=input_box[None, :],
                #                 multimask_output=False,
                #             )

                mask, _, _ = predictor.predict(
                                point_coords=np.array(landmark_points), 
                                point_labels = np.ones(len(landmark_points)),
                                box=input_box[None, :],
                                multimask_output=False,
                            )
                mask2 = mask[0].astype('uint8')
            else:
                # Initialize the mask with a rectangle around the hand
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, gc_iteration, cv2.GC_INIT_WITH_RECT)
                # Modify the mask to make it binary
                mask2 = np.where((mask == 2) | (mask == 0) | (mask == 1), 0, 1).astype('uint8')
            mask_ls.append(mask2)

    
    # aggregate the mask from multi-hands, 1 represents left, 2 represents right
    mask_agg = np.zeros(image2.shape[:2])
    if len(center_hand) == 2:
        hand_dist = np.linalg.norm(center_hand[0] - center_hand[1])
    else:
        hand_dist = 0
    if len(center_hand) == 2:
        if kitchen:
            dist_to_left_bottom0 = np.sqrt((center_hand[0][1] - image.shape[0]) ** 2 + center_hand[0][0] ** 2)
            dist_to_left_bottom1 = np.sqrt((center_hand[1][1] - image.shape[0]) ** 2 + center_hand[1][0] ** 2)
            left_index = 0 if dist_to_left_bottom0 < dist_to_left_bottom1 else 1
        else:
            left_index = 0 if center_hand[0][0] < center_hand[1][0] else 1
        mask_agg[mask_ls[left_index] == 1] = 1
        mask_agg[mask_ls[1-left_index] == 1] = 2
        image2[mask_agg==0] = 0
        image2[mask_agg==2] = image2[mask_agg==2]/2
        hand_pose_data["left_hand_index"].append(left_index)
    # less than two hands detected, follow the result from the last frame
    if len(center_hand) < 2 or hand_dist < 20:
        hand_pose_data = last_pose_data
        mask_ls = last_mask_ls
        mask_agg[mask_ls[hand_pose_data["left_hand_index"][0]] == 1] = 1
        mask_agg[mask_ls[1-hand_pose_data["left_hand_index"][0]] == 1] = 2
        image2[mask_agg==0] = 0
        image2[mask_agg==2] = image2[mask_agg==2]/2

    # draw keypoints and connections
    for landmarks in hand_pose_data["hand_landmarks"]:
        for x, y in landmarks:
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        # Draw lines connecting hand landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20)
        ]
        for connection in connections:
            point1 = landmarks[connection[0]]
            point2 = landmarks[connection[1]]
            cv2.line(image, point1, point2, (0, 255, 0), 1)


    # remove the green background misclassfied
    # Define the HSV range for green color (adjust the range as needed)
    # lower_green = np.array([35, 50, 50])  # Lower bound for green
    # upper_green = np.array([85, 255, 255])  # Upper bound for green
    # Create a mask for the green background
    # Convert the image to HSV
    # image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    # green_mask = cv2.inRange(image2_hsv, lower_green, upper_green)
    # mask_agg = np.where((mask_agg ==1) & (green_mask==0), 1, 0).astype('uint8')
    
    # Save the annotated image to the output directory - pose
    output_path = os.path.join(vis_pose_path, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

    # Save the annotated image to the output directory - mask
    output_path = os.path.join(vis_mask_path, os.path.basename(image_path))
    cv2.imwrite(output_path, image2)
    
    # Save the hand pose data to a JSON file
    output_path = os.path.join(save_pose_path, os.path.splitext(os.path.basename(image_path))[0] + ".json")
    with open(output_path, "w") as json_file:
        json.dump(hand_pose_data, json_file, indent=4)

    # Save the hand mask data to csv file
    output_path =  os.path.join(save_mask_path, os.path.splitext(os.path.basename(image_path))[0] + ".csv")
    np.savetxt(output_path, mask_agg, fmt='%d', delimiter=',')

    return hand_pose_data, mask_ls

def infer_pose_mask(root_dir, save_dir, device = 'cuda:1', kitchen = 'False'):

    ##################### Estimate and save the hand pose and mask from the ego-centric and exocentric RGB images #####################

    gc_iteration = 5
    # Initialize the MediaPipe Hands module
    mp_hands = mp.solutions.hands  
    # Initialize the SAM module
    DEVICE = torch.device(device if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["vit_l"](checkpoint="../sam/sam_vit_l_0b3195.pth").to(device=DEVICE)
    predictor = SamPredictor(sam)
    # predictor = None

    ego_calib_file = root_dir.split('/')[:-1] + '/calibration/ego_calib.json'
    exo_calib_file = root_dir.split('/')[:-1] + '/calibration/exo_calib.json'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        hands = mp_hands.Hands(min_detection_confidence=0.1)  
        hands_exo = mp_hands.Hands(min_detection_confidence=0.1)
        path = os.path.join(root_dir, clip)
        vis_pose_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_2d')
        vis_mask_path = os.path.join(save_dir, clip, 'vis_ego', 'mask_2d')
        vis_exo_pose_path = os.path.join(save_dir, clip, 'vis_exo', 'pose_2d')
        vis_exo_mask_path = os.path.join(save_dir, clip, 'vis_exo', 'mask_2d')
        save_pose_path = os.path.join(save_dir, clip, 'ego', 'pose_2d')
        save_mask_path = os.path.join(save_dir, clip, 'ego', 'mask_2d')
        save_exo_pose_path = os.path.join(save_dir, clip, 'exo', 'pose_2d')
        save_exo_mask_path = os.path.join(save_dir, clip, 'exo', 'mask_2d')
        if not os.path.exists(vis_pose_path):
            os.makedirs(vis_pose_path)
        if not os.path.exists(vis_mask_path):
            os.makedirs(vis_mask_path)
        if not os.path.exists(save_pose_path):
            os.makedirs(save_pose_path)
        if not os.path.exists(save_mask_path):
            os.makedirs(save_mask_path)
        if not os.path.exists(vis_exo_pose_path):
            os.makedirs(vis_exo_pose_path)
        if not os.path.exists(vis_exo_mask_path):
            os.makedirs(vis_exo_mask_path)
        if not os.path.exists(save_exo_pose_path):
            os.makedirs(save_exo_pose_path)
        if not os.path.exists(save_exo_mask_path):
            os.makedirs(save_exo_mask_path)
        img_path = os.path.join(path, 'egocentric', 'rgb')
        img_exo_path = os.path.join(path, 'exocentric', 'rgb')
        img_files = sorted(glob(img_path + '/'+ '*.png'))
        img_exo_files = sorted(glob(img_exo_path + '/'+ '*.png'))
        with open(ego_calib_file, 'r') as file:
            ego_calib = json.load(file)
        with open(exo_calib_file, 'r') as file:
            exo_calib = json.load(file)
        last_pose_data = {}
        last_mask_ls = []
        last_exo_pose_data = {}
        last_exo_mask_ls =[]
        # assert len(img_files) == len(img_exo_files)
        for img_file, img_exo_file in tqdm(zip(img_files, img_exo_files), total = len(img_files),  desc = clip):
            last_pose_data, last_mask_ls = process_image(img_file, hands, predictor, vis_pose_path, vis_mask_path, save_pose_path, save_mask_path, gc_iteration, ego_calib, last_pose_data, last_mask_ls)
            last_exo_pose_data, last_exo_mask_ls = process_image(img_exo_file, hands_exo, predictor, vis_exo_pose_path, vis_exo_mask_path, save_exo_pose_path, save_exo_mask_path, gc_iteration, exo_calib, last_exo_pose_data, last_exo_mask_ls, kitchen)
        torch.cuda.empty_cache()
    # Release resources
    hands.close()
    hands_exo.close()
    



if __name__ == "__main__":
    root_dir = '/mnt/data/MultimodalEgoHands/subject_01/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_01/'
    infer_pose_mask(root_dir, save_dir)
