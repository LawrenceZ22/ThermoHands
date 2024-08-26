import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import shutil
import moviepy.video.io.ImageSequenceClip

def vis_ego_exo_data(root_dir, save_dir):

    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        vis_path_rgb_ego = os.path.join(save_dir, clip, 'vis_ego', 'rgb')
        vis_path_ir_ego = os.path.join(save_dir, clip, 'vis_ego', 'ir')
        vis_path_depth_ego = os.path.join(save_dir, clip, 'vis_ego', 'depth')
        vis_path_thermal_ego = os.path.join(save_dir, clip, 'vis_ego', 'thermal')
        vis_path_rgb_exo = os.path.join(save_dir, clip, 'vis_exo', 'rgb')
        vis_path_depth_exo = os.path.join(save_dir, clip, 'vis_exo', 'depth')
        if not os.path.exists(vis_path_rgb_ego):
            os.makedirs(vis_path_rgb_ego)
        if not os.path.exists(vis_path_ir_ego):
            os.makedirs(vis_path_ir_ego)
        if not os.path.exists(vis_path_depth_ego):
            os.makedirs(vis_path_depth_ego) 
        if not os.path.exists(vis_path_thermal_ego):
            os.makedirs(vis_path_thermal_ego)
        if not os.path.exists(vis_path_rgb_exo):
            os.makedirs(vis_path_rgb_exo)
        if not os.path.exists(vis_path_depth_exo):
            os.makedirs(vis_path_depth_exo) 
        rgb_path = os.path.join(path, 'egocentric', 'rgb')
        ir_path = os.path.join(path, 'egocentric', 'ir')
        depth_path = os.path.join(path, 'egocentric', 'depth')
        thermal_path = os.path.join(path, 'egocentric', 'thermal')
        exo_rgb_path = os.path.join(path, 'exocentric', 'rgb')
        exo_depth_path = os.path.join(path, 'exocentric', 'depth')
        rgb_files = sorted(glob(rgb_path + '/'+ '*.png'))
        ir_files = sorted(glob(ir_path + '/'+ '*.png'))
        depth_files = sorted(glob(depth_path + '/'+ '*.png'))
        thermal_files = sorted(glob(thermal_path + '/'+ '*.tiff'))
        exo_rgb_files = sorted(glob(exo_rgb_path + '/'+ '*.png'))
        exo_depth_files = sorted(glob(exo_depth_path + '/'+ '*.png'))

        # visualize rgb, ir and depth images
        assert len(rgb_files) == len(ir_files)
        assert len(ir_files) == len(depth_files)
        assert len(ir_files) == len(thermal_files)
        assert len(exo_rgb_files) == len(exo_depth_files)
        for rgb_file, ir_file, depth_file, exo_rgb_file, exo_depth_file in tqdm(zip(rgb_files,ir_files,depth_files,exo_rgb_files,exo_depth_files), total = len(rgb_files), desc = "visualizing rgb, ir and depth images"):
            # Open the images 
            rgb = cv2.imread(rgb_file)
            ir = cv2.imread(ir_file)
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
            depth = depth/1000
            exo_rgb = cv2.imread(exo_rgb_file)
            exo_depth = cv2.imread(exo_depth_file, cv2.IMREAD_UNCHANGED)
            exo_depth = exo_depth/1000
            cv2.imwrite(os.path.join(vis_path_rgb_ego, rgb_file.split('/')[-1].split('.')[-2] + '.png'), rgb)
            cv2.imwrite(os.path.join(vis_path_ir_ego, ir_file.split('/')[-1].split('.')[-2] + '.png'), ir)
            cv2.imwrite(os.path.join(vis_path_rgb_exo, exo_rgb_file.split('/')[-1].split('.')[-2] + '.png'), exo_rgb)
            plt.imshow(depth, cmap='jet', vmin=0, vmax=2)
            plt.axis('off')  # Turn off axis labels
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path_depth_ego, depth_file.split('/')[-1].split('.')[-2] + '.png'))
            plt.close()
            plt.clf()
            plt.imshow(exo_depth, cmap='jet', vmin=0, vmax=2)
            plt.axis('off')  # Turn off axis labels
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path_depth_exo, exo_depth_file.split('/')[-1].split('.')[-2] + '.png'))
            plt.close()
            plt.clf()

        # visualize thermal images
        for thermal_file in tqdm(thermal_files, desc = "visualizing thermal images"):
            thermal = cv2.imread(thermal_file)
            plt.imshow(thermal, cmap='jet')
            plt.axis('off')  # Turn off axis labels
            plt.tight_layout()
            plt.savefig(os.path.join(vis_path_thermal_ego, thermal_file.split('/')[-1].split('.')[-2] + '.png'))
            plt.close()
            plt.clf()
        
        # make videos for saved images
        rgb_files = sorted(glob(vis_path_rgb_ego + '/'+ '*.png'))
        ir_files = sorted(glob(vis_path_ir_ego + '/'+ '*.png'))
        depth_files = sorted(glob(vis_path_depth_ego + '/'+ '*.png'))
        thermal_files = sorted(glob(vis_path_thermal_ego + '/'+ '*.png'))
        exo_rgb_files = sorted(glob(vis_path_rgb_exo + '/'+ '*.png'))
        exo_depth_files = sorted(glob(vis_path_depth_exo + '/'+ '*.png'))
        if (len(rgb_files)>0):
            Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(rgb_files, fps=8.5)
            Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_rgb.mp4'))
            shutil.rmtree(vis_path_rgb_ego)
        if (len(ir_files)>0):
            Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(ir_files, fps=8.5)
            Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_ir.mp4'))
            shutil.rmtree(vis_path_ir_ego)
        if (len(depth_files)>0):
            Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(depth_files, fps=8.5)
            Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_depth.mp4'))
            shutil.rmtree(vis_path_depth_ego)
        if (len(thermal_files)>0):
            Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(thermal_files, fps=8.5)
            Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_thermal.mp4'))
            shutil.rmtree(vis_path_thermal_ego)
        if (len(exo_rgb_files)>0):
            Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(exo_rgb_files, fps=8.5)
            Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_exo', 'mv_rgb.mp4'))
            shutil.rmtree(vis_path_rgb_exo)
        if (len(exo_depth_files)>0):
            Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(exo_depth_files, fps=8.5)
            Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_exo', 'mv_depth.mp4'))
            shutil.rmtree(vis_path_depth_exo)
        print('###finish for {}###'.format(clip))


if __name__ == "__main__":
    root_dir = '/mnt/data/MultimodalEgoHands/subject_01/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_01/'
    vis_ego_exo_data(root_dir, save_dir)

