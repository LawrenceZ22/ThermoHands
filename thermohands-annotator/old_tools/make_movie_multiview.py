import os
from glob import glob
import moviepy.video.io.ImageSequenceClip

fps = 10  # 8.5 for thermal image;  10 for 3dmd mesh;  30 for RGB-D camera
v_name = 'vis_allo_pose'
root_dir = '/mnt/data/TherHandsPro/'
save_dir = '/mnt/data/TherHandsPro/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
source_dirs = sorted(os.listdir(root_dir))
# source_dirs = [root_dir]
for source in source_dirs:
    image_dir = os.path.join(root_dir, source, v_name)
    if os.path.exists(image_dir):
        cam_dirs = sorted(os.listdir(image_dir))
        for cam in cam_dirs:
            cam_dir = os.path.join(image_dir, cam)
            image_files = sorted(glob(cam_dir + '/'+ '*.jpg'))
            save_path = os.path.join(save_dir, source, v_name + '_video', '{}.mp4'.format(cam))
            if (len(image_files)>0):
                clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                clip.write_videofile(save_path)


