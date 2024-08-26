import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from glob import glob
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    root_dir =  '/mnt/data/MultimodalEgoHands/'
    root_dir2 = '/mnt/data/MultimodalEgoHands/gestures/'
    save_dir = '/mnt/data/fangqiang/TherHandsPro/'
    save_subs = sorted(os.listdir(save_dir))
    subs =  sorted(os.listdir(root_dir))
    subs2 = sorted(os.listdir(root_dir2))
    subs_path = []
    subs_main = []
    for i, sub in enumerate(subs):
        if sub[:7] == 'subject' and sub[-3] == '_':
            subs_path.append(os.path.join(save_dir, sub))
            if not sub[:10] in subs_main:
                subs_main.append(sub[:10])
    for i, sub in enumerate(subs2):
        if sub[:7] == 'subject' and len(sub) == 18:
            subs_path.append(os.path.join(save_dir, sub))
            if not sub[:10] in subs_main:
                subs_main.append(sub[:10])
    actions_main = ['cut_paper', 'fold_paper', 'pour_water', 'read_book', 'staple_paper', 'write_with_pen', \
                'write_with_pencil','pinch_and_drag', 'pinch_and_hold', 'swipe', 'tap', 'touch']
    pose_dict_l = {name: [] for name in actions_main}
    pose_dict_r = {name: [] for name in actions_main}
    shape_dict_l = dict.fromkeys(subs_main)
    shape_dict_r = dict.fromkeys(subs_main)
    for sub_path in tqdm(subs_path):
        actions = sorted(os.listdir(sub_path))
        shapes_l = []
        shapes_r = []
        for action in actions:
            action_path = os.path.join(sub_path, action)
            gt_path = os.path.join(action_path, 'gt_info')
            gt_files = glob(gt_path + '/*.json')
            for k, gt_file in enumerate(gt_files[::5]):
                with open(gt_file, 'r') as json_file:
                    load_hand = json.load(json_file)
                pose_dict_l[action].append(load_hand['poseCoeff_L'][0][3:])
                pose_dict_r[action].append(load_hand['poseCoeff_R'][0][3:])
                if k==0:
                    shapes_l.append(load_hand['beta_L'][0])
                    shapes_r.append(load_hand['beta_R'][0])
        shape_dict_l[sub_path.split('/')[-1][:10]] = np.array(shapes_l)
        shape_dict_r[sub_path.split('/')[-1][:10]] = np.array(shapes_r)
    for action in actions_main:
        pose_dict_l[action] = np.array(pose_dict_l[action])
        pose_dict_r[action] = np.array(pose_dict_r[action])
    for sub in subs_main:
        shape_dict_l[sub] = np.array(shape_dict_l[sub]).mean(axis=0)
        shape_dict_r[sub] = np.array(shape_dict_r[sub]).mean(axis=0)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(pose_dict_l[sub])
        # # Optionally reduce dimensionality if X has a lot of features
        # # pca = PCA(n_components=20)
        # # X_pca = pca.fit_transform(X_scaled)

        # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200)
        # data_2d = tsne.fit_transform(X_scaled)

        # plt.figure(figsize=(12, 10))
        # plt.scatter(data_2d[:, 0], data_2d[:, 1], s=1, alpha=0.5)

        # plt.title('2D t-SNE Embedding of Hand Poses')
        # plt.xlabel('Component 1')
        # plt.ylabel('Component 2')
        # plt.legend(markerscale=2)
        # plt.savefig('tsne_pose_l.png', dpi=200)
    # start the visualization
    all_shape_l = [shape_dict_l[sub] for sub in subs_main]  # Example data
    all_shape_r = [shape_dict_r[sub] for sub in subs_main]  # Example data

    # Concatenate all arrays and keep track of labels (subject IDs)
    data_l = np.vstack(all_shape_l)
    data_r = np.vstack(all_shape_r)
    labels = np.array([i for i, arr in enumerate(all_shape_l) for _ in range(len(arr))])

    scaler = StandardScaler()
    data_scaled_l = scaler.fit_transform(data_l) 
    data_scaled_r = scaler.fit_transform(data_r) 
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200)
    data_2d_l = tsne.fit_transform(data_scaled_l)
    data_2d_r = tsne.fit_transform(data_scaled_r)

    # Visualize
    plt.figure(figsize=(12, 10))
    for i in range(len(subs_main)):
        plt.scatter(data_2d_l[i, 0], data_2d_l[i, 1], label=subs_main[i], s = 1, alpha=0.5)
    plt.title('2D t-SNE Embedding of Hand shapes - Left', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left',fontsize=12)
    plt.savefig('tsne_shape_l.png', dpi=200)
    plt.close()

    plt.figure(figsize=(12, 10))
    for i in range(len(subs_main)):
        plt.scatter(data_2d_r[i, 0], data_2d_r[i, 1], label=subs_main[i], s = 1, alpha=0.5)
    plt.title('2D t-SNE Embedding of Hand shapes - Right', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left',fontsize=12)
    plt.savefig('tsne_shape_r.png', dpi=200)
    plt.close()


    all_pose_l = [pose_dict_l[action] for action in actions_main]  # Example data
    all_pose_r = [pose_dict_r[action] for action in actions_main]  # Example data


    # Concatenate all arrays and keep track of labels (subject IDs)
    data_l = np.vstack(all_pose_l)
    data_r = np.vstack(all_pose_r)
    labels = np.array([i for i, arr in enumerate(all_pose_l) for _ in range(len(arr))])

    scaler = StandardScaler()
    data_scaled_l = scaler.fit_transform(data_l) 
    data_scaled_r = scaler.fit_transform(data_r) 
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200)
    data_2d_l = tsne.fit_transform(data_scaled_l)
    data_2d_r = tsne.fit_transform(data_scaled_r)

    # Visualize
    plt.figure(figsize=(12, 10))
    for i in range(len(actions_main)):
        plt.scatter(data_2d_l[labels == i, 0], data_2d_l[labels == i, 1], label=actions_main[i], s = 1, alpha=0.5)
    plt.title('2D t-SNE Embedding of Hand Poses - Left', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.savefig('tsne_pose_l.png', dpi=200)
    plt.close()

    plt.figure(figsize=(12, 10))
    for i in range(len(actions_main)):
        plt.scatter(data_2d_r[labels == i, 0], data_2d_r[labels == i, 1], label=actions_main[i], s = 1, alpha=0.5)
    plt.title('2D t-SNE Embedding of Hand Poses - Right', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.savefig('tsne_pose_r.png', dpi=200)
    plt.close()




if __name__ == "__main__":
    main()
