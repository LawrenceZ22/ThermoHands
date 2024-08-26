import os
from glob import glob
import moviepy.video.io.ImageSequenceClip


def movie_annotation(root_dir, save_dir):

    
    clips = sorted(os.listdir(root_dir))
    for clip in clips: 
        path = os.path.join(root_dir, clip)
        vis_pose_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_2d')
        vis_mask_path = os.path.join(save_dir, clip, 'vis_ego', 'mask_2d')
        vis_pose_tri_path = os.path.join(save_dir, clip, 'vis_ego', 'pose_3d_tri')
        vis_odometry = os.path.join(save_dir, clip, 'vis_ego', 'kiss_icp_odom')
        vis_pose_exo_path = os.path.join(save_dir, clip, 'vis_exo', 'pose_2d')
        vis_mask_exo_path = os.path.join(save_dir, clip, 'vis_exo', 'mask_2d')
        vis_marker_exo_path = os.path.join(save_dir, clip, 'vis_exo', 'marker_2d_proj')
        gt_ego_path = os.path.join(save_dir, clip, 'gt_pose_ego')
        gt_exo_path = os.path.join(save_dir, clip, 'gt_pose_exo')
        gt_left_path = os.path.join(save_dir, clip, 'gt_pose_left3D')
        gt_right_path = os.path.join(save_dir, clip, 'gt_pose_right3D')
        pose_files = sorted(glob(vis_pose_path + '/'+ '*.png'))
        mask_files = sorted(glob(vis_mask_path + '/'+ '*.png'))
        odometry_files = sorted(glob(vis_odometry + '/'+ '*.png'))
        pose_exo_files = sorted(glob(vis_pose_exo_path + '/'+ '*.png'))
        mask_exo_files = sorted(glob(vis_mask_exo_path + '/'+ '*.png'))
        marker_exo_files = sorted(glob(vis_marker_exo_path + '/'+ '*.png'))
        pose_tri_files = sorted(glob(vis_pose_tri_path + '/'+ '*.png'))
        gt_ego_files = sorted(glob(gt_ego_path + '/'+ '*.png'))
        gt_exo_files = sorted(glob(gt_exo_path + '/'+ '*.png'))
        gt_left_files = sorted(glob(gt_left_path + '/'+ '*.png'))
        gt_right_files = sorted(glob(gt_right_path + '/'+ '*.png'))
        # if (len(pose_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(pose_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_pose_2d.mp4'))
        # if (len(mask_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(mask_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_mask_2d.mp4'))
        # if (len(odometry_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(odometry_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_odometry.mp4'))
        # if (len(pose_tri_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(pose_tri_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_pose_3d_tri.mp4'))
        # if (len(pose_exo_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(pose_exo_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_exo', 'mv_pose_2d.mp4'))
        # if (len(mask_exo_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(mask_exo_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_exo', 'mv_mask_2d.mp4'))
        # if (len(marker_exo_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(marker_exo_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_exo', 'mv_marker_2d_proj.mp4'))  
        # if (len(gt_ego_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(gt_ego_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_gt_ego.mp4'))  
        if (len(gt_exo_files)>0):
            Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(gt_exo_files, fps=8.5)
            Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_exo', 'mv_gt_exo.mp4'))  
        # if (len(gt_right_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(gt_right_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_gt_right.mp4'))  
        # if (len(gt_left_files)>0):
        #     Imgclip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(gt_left_files, fps=8.5)
        #     Imgclip.write_videofile(os.path.join(save_dir, clip, 'vis_ego', 'mv_gt_left.mp4'))  
        print('finish for {}'.format(clip))





if __name__ == "__main__":
    root_dir = '/mnt/data/MultimodalEgoHands/subject_01_kitchen/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_01_kitchen/'
    movie_annotation(root_dir, save_dir)

