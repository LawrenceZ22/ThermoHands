import os
import numpy as np
import json
import cv2
from tqdm import tqdm
from filterpy.kalman import KalmanFilter
from glob import glob

def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix (assuming constant velocity model)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Measurement function
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # Covariance matrix
    kf.P *= 1000.
    
    # Measurement noise
    position_measurement_noise = 10
    kf.R = np.array([[position_measurement_noise, 0],
                     [0, position_measurement_noise]])
    
    # Process noise
    kf.Q = np.eye(kf.dim_x) 
    velocity_process_noise = 0.1  
    kf.Q[2][2] = velocity_process_noise  # Velocity X noise
    kf.Q[3][3] = velocity_process_noise  # Velocity Y noise
    
    return kf

def process_ir(image_path, vis_path, save_path, threshold = 190, area_threshold = [10, 100]):

    # Load the IR image
    ir_image = cv2.imread(image_path)
    # Convert the image to grayscale
    ir_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    # Create a dictionary to store the hand pose results
    marker_info = {"image_path": image_path, "markers": []}
    # Create a black rectangle image
    # win_x, win_y, win_w, win_h = 260, 180, 260, 280
    # black_rectangle = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    # # Paste the black rectangle onto the original image
    # ir_gray[win_y:win_y+win_h, win_x:win_x+win_w] = black_rectangle[:,:,0]
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
    output_path = os.path.join(vis_path, os.path.basename(image_path))
    cv2.imwrite(output_path, ir_image)

    # Save the marker 2d data to a JSON file
    output_path = os.path.join(save_path, os.path.splitext(os.path.basename(image_path))[0] + ".json")
    with open(output_path, "w") as json_file:
        json.dump(marker_info, json_file, indent=4)

def process_marker_exo(path, save_marker_exo_path, vis_marker_exo_path, exo_calib):

    ir_camera_matrix = np.array(exo_calib['ir_mtx'])  # Intrinsic matrix for IR camera
    ir_distortion_coeffs = np.array(exo_calib['ir_dist'])  # Distortion coefficients for IR camera
    exo_rgb_path = os.path.join(path, 'exocentric', 'rgb')
    exo_rgb_files = sorted(glob(exo_rgb_path + '/'+ '*.png'))
    viz_image = cv2.imread(exo_rgb_files[0])
    viz_image = cv2.undistort(viz_image, ir_camera_matrix, ir_distortion_coeffs)
    # Create a window to display the image
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', viz_image)
    anno_markers = []
    def mouse_callback(event, x, y, flags, param):
        # Capture the mouse click event and store the position
        if event == cv2.EVENT_LBUTTONDOWN:
            marker_id = len(anno_markers) 
            anno_markers.append((x, y))
            cv2.circle(viz_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the marker position
            cv2.imshow('Image', viz_image)
    # Set the mouse callback function
    cv2.setMouseCallback('Image', mouse_callback)
    print("Click on the image to annotate marker positions. Press 'q' to quit.")

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

    # save the annotated exo markers and viz them
    exo_markers = np.array(anno_markers)
    for exo_rgb_file in tqdm(exo_rgb_files[0:1]):
        marker_info = {"image_path": exo_rgb_file, "markers": exo_markers.tolist()}
        # Save the new marker 2d data to a JSON file
        output_path = os.path.join(save_marker_exo_path, os.path.splitext(os.path.basename(exo_rgb_file))[0] + ".json")
        with open(output_path, "w") as json_file:
            json.dump(marker_info, json_file, indent=4)
        # Load the exo rgb image
        exo_rgb_image = cv2.imread(exo_rgb_file)
        for marker, i in zip(marker_info["markers"], range(0, len(marker_info['markers']))):
            x = int(marker[0])
            y = int(marker[1])
            cv2.rectangle(exo_rgb_image, (x-3, y-3), (x + 3, y + 3), (0, 255, 0), 2)
            cv2.putText(exo_rgb_image, str(i), (x-5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        # Save the annotated image to the output directory - marker
        output_path = os.path.join(vis_marker_exo_path, os.path.basename(exo_rgb_file))
        cv2.imwrite(output_path, exo_rgb_image)
    
    return output_path


def infer_ego_exo_markers(root_dir, save_dir):

    exo_calib_file = '/mnt/data/MultimodalEgoHands/calibration/exo_calib.json'
    with open(exo_calib_file, 'r') as file:
        exo_calib = json.load(file)
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        vis_marker_path = os.path.join(save_dir, clip, 'vis_ego', 'marker_2d')
        save_marker_path = os.path.join(save_dir, clip, 'ego', 'marker_2d')
        vis_marker_n_path = os.path.join(save_dir, clip, 'vis_ego', 'marker_2d_kf')
        save_marker_n_path = os.path.join(save_dir, clip, 'ego', 'marker_2d_kf')
        vis_marker_exo_path = os.path.join(save_dir, clip, 'vis_exo', 'marker_2d_anno')
        save_marker_exo_path = os.path.join(save_dir, clip, 'exo', 'marker_2d_anno')
        if not os.path.exists(vis_marker_path):
            os.makedirs(vis_marker_path)
        if not os.path.exists(save_marker_path):
            os.makedirs(save_marker_path)
        if not os.path.exists(vis_marker_n_path):
            os.makedirs(vis_marker_n_path)
        if not os.path.exists(save_marker_n_path):
            os.makedirs(save_marker_n_path)
        if not os.path.exists(vis_marker_exo_path):
            os.makedirs(vis_marker_exo_path)
        if not os.path.exists(save_marker_exo_path):
            os.makedirs(save_marker_exo_path)
        # read the ir file and detect the reflective markers for the egocentric
        ir_path = os.path.join(path, 'egocentric', 'ir')
        ir_files = sorted(glob(ir_path + '/'+ '*.png'))
        # for ir_file in tqdm(ir_files):
        #     process_ir(ir_file, vis_marker_path, save_marker_path)
        

        # the exocentric camera is static, therefore we only annotate the first frame 
        output_path = process_marker_exo(path, save_marker_exo_path, vis_marker_exo_path, exo_calib)


        # some markers are missed in certain frame, need to fill the gap using the Kalman Filter
        marker_files = sorted(os.listdir(save_marker_path))
        prev_markers = []
        # for marker_file, ir_file in tqdm(zip(marker_files, ir_files), total=len(marker_files)):

        #     with open(os.path.join(save_marker_path, marker_file), "r") as json_file:
        #         marker_info = json.load(json_file)['markers']
        #     # sort the makers from the left from egocentric view
        #     markers = np.array([[marker['x'], marker['y']] for marker in marker_info])
        #     sort_indices = np.argsort(markers[:, 0])
        #     markers = markers[sort_indices]
        for ir_file in tqdm(ir_files[0:1]):
            if len(prev_markers) >0:
                break
                # # Calculate the Euclidean distance between markers from consecutive frames
                # distances = np.linalg.norm(prev_markers[:, None, :] - markers[None, :, :], axis=2)
                # # Initialize a array to hold the matched markers
                # matched_markers = np.zeros(prev_markers.shape)
                # # Iterate over each previous marker and find the closest current marker
                # for previous_index, distance_row in enumerate(distances):
                #     # Find the index of the closest current marker
                #     current_index = np.argmin(distance_row)
                #     # Check if the closest marker is within the threshold distance
                #     if distance_row[current_index] < 20:
                #         matched_markers[previous_index] = markers[current_index]
                #         kalman_filters[previous_index].update(markers[current_index])
                #     else:
                #         kalman_filters[previous_index].predict()
                #         matched_markers[previous_index] = kalman_filters[previous_index].x[:2].flatten()
                #         # kalman_filters[previous_index].update(matched_markers[previous_index])
                # prev_markers = matched_markers
            # Initialize for the first frame
            else:
                viz_image = cv2.imread(ir_file)
                viz_anno_exo = cv2.imread(output_path)
                # Create a window to display the image
                cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
                cv2.imshow('Image', viz_image)
                cv2.imshow('Exo image', viz_anno_exo)
                anno_markers = []
                def mouse_callback(event, x, y, flags, param):
                    # Capture the mouse click event and store the position
                    if event == cv2.EVENT_LBUTTONDOWN:
                        marker_id = len(anno_markers) 
                        anno_markers.append((x, y))
                        cv2.circle(viz_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the marker position
                        cv2.imshow('Image', viz_image)
                # Set the mouse callback function
                cv2.setMouseCallback('Image', mouse_callback)
                print("Click on the image to annotate marker positions. Press 'q' to quit.")

                while True:
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                cv2.destroyAllWindows()

                # only keep the top markers for initilization
                prev_markers = np.array(anno_markers)
                # Initialize Kalman filters for all markers
                # kalman_filters = [initialize_kalman_filter() for _ in range(markers.shape[0])]
                # for k in range(markers.shape[0]):
                #      kalman_filters[k].update(markers[k])

            # Create a dictionary to store the marker results
            marker_info = {"image_path": ir_file, "markers": prev_markers.tolist()}
            # Save the new marker 2d data to a JSON file
            output_path = os.path.join(save_marker_n_path, os.path.splitext(os.path.basename(ir_file))[0] + ".json")
            with open(output_path, "w") as json_file:
                json.dump(marker_info, json_file, indent=4)
            # Load the IR image
            ir_image = cv2.imread(ir_file)
            for marker in marker_info["markers"]:
                x = int(marker[0])
                y = int(marker[1])
                cv2.rectangle(ir_image, (x-3, y-3), (x + 3, y + 3), (0, 255, 0), 2)
            # Save the annotated image to the output directory - marker
            output_path = os.path.join(vis_marker_n_path, os.path.basename(ir_file))
            cv2.imwrite(output_path, ir_image)
            
                


    

if __name__ == '__main__':

    root_dir = '/mnt/data/MultimodalEgoHands/subject_03/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_03/'
    infer_ego_exo_markers(root_dir, save_dir)