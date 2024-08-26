import os
from glob import glob
import shutil

def delete_some_footprint(directory):
    dirs =  sorted(os.listdir(directory))
    for dir in dirs:
        if not dir[-4:] == 'odom':
            dir_path = os.path.join(directory, dir)
            actions = os.listdir(dir_path)
            for action in actions:
                action_path = os.path.join(dir_path, action)
                loss_path = os.path.join(action_path, 'gt_loss')
                exo_gt_path = os.path.join(action_path, 'gt_pose_exo')
                vis_ego_mask_path = os.path.join(action_path, 'vis_ego', 'mask_2d')
                vis_exo_mask_path = os.path.join(action_path, 'vis_exo', 'mask_2d')
                vis_ego_odom_path = os.path.join(action_path, 'vis_ego', 'kiss_icp_odom')
                vis_ego_movies = glob(os.path.join(action_path, 'vis_ego') + '/' + '*.mp4')
                vis_exo_movies = glob(os.path.join(action_path, 'vis_exo') + '/' + '*.mp4')
                # if os.path.exists(loss_path):
                #     print(f"Deleting: {loss_path}")
                #     shutil.rmtree(loss_path)
                if os.path.exists(exo_gt_path):
                    print(f"Deleting: {exo_gt_path}")
                    shutil.rmtree(exo_gt_path)
                if os.path.exists(vis_exo_mask_path):
                    print(f"Deleting: {vis_exo_mask_path}")
                    shutil.rmtree(vis_exo_mask_path)
                if os.path.exists(vis_ego_mask_path):
                    print(f"Deleting: {vis_ego_mask_path}")
                    shutil.rmtree(vis_ego_mask_path)
                if os.path.exists(vis_ego_odom_path):
                    print(f"Deleting: {vis_ego_odom_path}")
                    shutil.rmtree(vis_ego_odom_path)
                if os.path.exists(os.path.join(action_path, 'vis_ego')):
                    for ego_movie in vis_ego_movies:
                        print(f"Deleting: {ego_movie}")
                        os.remove(ego_movie)
                if os.path.exists(os.path.join(action_path, 'vis_exo')):
                    for exo_movie in vis_exo_movies:
                        print(f"Deleting: {exo_movie}")
                        os.remove(exo_movie)
                    # Uncomment the line above after you're sure the script targets the correct files.

# Example usage:
directory = '/mnt/data/fangqiang/TherHandsPro/'
# actions = ['cut_paper', 'fold_paper', 'pour_water', 'read_book', 'staple_paper', 'write_with_pen', 'write_with_pencil', 'pinch_and_drag', 'pinch_and_hold', \
#     'swipe', 'tap', 'touch']
delete_some_footprint(directory)
 