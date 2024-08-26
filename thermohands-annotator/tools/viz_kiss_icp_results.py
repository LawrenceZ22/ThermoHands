import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.pyplot import MultipleLocator


def viz_kiss_icp_res(root_dir, save_dir, odom_path):

    clips = sorted(os.listdir(root_dir))
    clips_odom = sorted(os.listdir(odom_path))[:-1]
    for clip, clip_odom in zip(clips, clips_odom):
        vis_odom_path = os.path.join(save_dir, clip, 'vis_ego', 'kiss_icp_odom')
        if not os.path.exists(vis_odom_path):
            os.makedirs(vis_odom_path)
        poses_file = os.path.join(odom_path, clip_odom, 'depth_pcd_poses.npy')
        odometry = np.load(poses_file)
        num_frames = odometry.shape[0]
        transforms = np.zeros((num_frames, 4, 4))
        ego_loc = np.zeros((num_frames, 3))
        ego_angle = np.zeros((num_frames, 3))
        transforms[0] = odometry[0]
        for i in range(1, num_frames):
            transforms[i] = transforms[i-1] @ odometry[i]
        for i in range(1, num_frames):
            ego_loc[i] = (odometry[i] @ np.hstack((ego_loc[i-1], np.ones(1))))[:3]
            rotation_matrix = transforms[i][:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            ego_angle[i] = euler_angles
        for i in tqdm(range(0, num_frames)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(ego_loc[:i,0], ego_loc[:i,1], ego_loc[:i,2], color = (254/255, 129/255, 125/255), linewidth = 1)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.set_xlim(-0.01,0.01)
            ax.set_ylim(-0.01,0.01)
            ax.set_zlim(-0.01,0.01)
            x_locator = MultipleLocator(0.005)
            y_locator = MultipleLocator(0.005)
            z_locator = MultipleLocator(0.005)
            ax.xaxis.set_major_locator(x_locator)
            ax.yaxis.set_major_locator(y_locator)
            ax.zaxis.set_major_locator(z_locator)
            ax.set_title('Egocentric Camera Trajectory - Frame {}'.format(i))
            output_path = os.path.join(vis_odom_path, '{}.png'.format(str(i).zfill(5)))
            plt.savefig(output_path, dpi = 300)
            plt.close()
            plt.clf()

        


if __name__ == '__main__':
    root_dir = '/mnt/data/MultimodalEgoHands/subject_03/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/subject_03/'
    odom_path = '/mnt/data/fangqiang/TherHandsPro/subject_03_odom/'
    viz_kiss_icp_res(root_dir, save_dir, odom_path)