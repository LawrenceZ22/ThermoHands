import cv2
import os
from glob import glob
import json

def main():

    root_dir = '/Users/s2207427/Documents/thermal-hand/ThermalHands/'
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        # read the ir file and detect the reflective markers
        ir_files = sorted(glob(path + '/'+ '*_C.jpg'))
        for ir_file in ir_files:
            # Load the image
            image = cv2.imread(ir_file)
            # Create a window to display the image
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.imshow('Image', image)
            # Initialize variables to store marker positions
            marker_info = {"markers": []}
        
            def mouse_callback(event, x, y, flags, param):
                # Capture the mouse click event and store the position
                if event == cv2.EVENT_LBUTTONDOWN:
                    marker_id = len(marker_info["markers"]) 
                    marker_info["markers"].append((x, y, marker_id))
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at the marker position
                    cv2.imshow('Image', image)
              

            # Set the mouse callback function
            cv2.setMouseCallback('Image', mouse_callback)

            print("Click on the image to annotate marker positions. Press 'q' to quit.")

            while True:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            
            # Save the annotated image to the output directory - marker
            output_path = os.path.join(path, 'vis_allo_marker.png')
            cv2.imwrite(output_path, image)

             # Save the marker 2d data to a JSON file
            output_path = os.path.join(path, "allo_marker.json")
            with open(output_path, "w") as json_file:
                json.dump(marker_info, json_file, indent=4)

            cv2.destroyAllWindows()

            # You can now access the marker positions in the 'marker_positions' list
            print("Marker Positions:", marker_info["markers"])

if __name__ == "__main__":
    main()
