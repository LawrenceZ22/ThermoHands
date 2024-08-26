
import os
import matplotlib.pyplot as plt
import json

# Directories
sequence_folder = '/mnt/data/MultimodalEgoHands/subject_21/staple_paper/egocentric/rgb/'
auto_annotations_folder = '/mnt/data/fangqiang/TherHandsPro/subject_21/staple_paper/ego/pose_2d/'  # Automatic annotations
annotations_folder = '/mnt/data/fangqiang/mannual-anno-hand/subject_21/staple_paper/ego_anno/'  # Manual adjustments
figures_folder = '/mnt/data/fangqiang/mannual-anno-hand/subject_21/staple_paper/ego_anno_vis/'

# Ensure directories exist
os.makedirs(annotations_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

# List all image files
sequence_files = sorted([f for f in os.listdir(sequence_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Placeholder for manual annotations
annotations = {}
plotted_points = {}

def load_auto_annotations(image_name):
    # Assuming each image has a corresponding JSON file with the same name
    annotation_file = os.path.join(auto_annotations_folder, f"{image_name.split('.')[-2]}.json")
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as f:
            anno = json.load(f)
            auto_anno = {}
            points = []
            points.append(anno['hand_landmarks'][anno['left_hand_index'][0]])
            points.append(anno['hand_landmarks'][1-anno['left_hand_index'][0]])
            auto_anno['points'] = points[0] + points[1]
    return auto_anno

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    annotations[current_image]['points'].append((ix, iy))
    point, = plt.plot(ix, iy, 'ro', markersize=3)
    plotted_points[current_image].append(point)
    plt.draw()

def press(event):
    if event.key == 'b' and annotations[current_image]['points']:
        annotations[current_image]['points'].pop()
        last_point = plotted_points[current_image].pop()
        last_point.remove()
        plt.draw()
    elif event.key == 'n':
        plt.close()

for current_image in sequence_files[134:]:
    auto_annotations = load_auto_annotations(current_image)
    
    img_path = os.path.join(sequence_folder, current_image)
    img = plt.imread(img_path)
    fig = plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.title(f'Annotate or Review: {current_image}')
    
    annotations[current_image] = {'points': auto_annotations.get('points', [])}
    plotted_points[current_image] = []

    # Plot automatic annotations
    for point in annotations[current_image]['points']:
        plotted_point, = plt.plot(*point, 'ro', markersize=3)
        plotted_points[current_image].append(plotted_point)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', press)
    
    plt.show()

    # Save the figure with annotations
    figure_path = os.path.join(figures_folder, f'annotated_{current_image}.png')
    fig.savefig(figure_path)
    plt.close(fig)
    
    # Save (or overwrite with manual adjustments) the annotations for the current image in a JSON file
    annotation_path = os.path.join(annotations_folder, f'{current_image}.json')
    with open(annotation_path, 'w') as f:
        json.dump(annotations[current_image], f)

print("Annotation process completed.")
