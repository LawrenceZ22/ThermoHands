import os
from glob import glob
import moviepy.video.io.ImageSequenceClip

fps = 30  # 8.5 for thermal image;  10 for 3dmd mesh;  30 for RGB-D camera
v_name = 'vis_depth'
root_dir = '/mnt/data/TherHandsPro/'
save_dir = '/mnt/data/TherHandsPro/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
source_dirs = sorted(os.listdir(root_dir))
# source_dirs = [root_dir]
for source in source_dirs:
    image_dir = os.path.join(root_dir, source, v_name)
    if os.path.exists(image_dir):
        image_files = sorted(glob(image_dir + '/'+ '*.png'))
        save_path = os.path.join(save_dir, source, '{}.mp4'.format(v_name))
        if (len(image_files)>0):
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(save_path)
    break


