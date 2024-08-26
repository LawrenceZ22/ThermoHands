import os
from glob import glob
import shutil
from tqdm import tqdm
import json


root_dir =  '/mnt/data/MultimodalEgoHands/'
root_dir2 = '/mnt/data/MultimodalEgoHands/gestures/'
save_dir = '/mnt/data/fangqiang/TherHandsPro/'
split_file = '/home/fangqiang/thermal-hand/split.json'
save_subs = sorted(os.listdir(save_dir))

with open(split_file, 'r') as file:
    data_split = json.load(file)

train_split = data_split["train"]
val_split = data_split["val"]
test_split = data_split["test"]
subs =  sorted(os.listdir(root_dir))
subs2 = sorted(os.listdir(root_dir2))
subs_path = []
for i, sub in enumerate(subs):
    if sub[:7] == 'subject':
        subs_path.append(os.path.join(root_dir, sub))
for i, sub in enumerate(subs2):
    if sub[:7] == 'subject':
        subs_path.append(os.path.join(root_dir2, sub))
print('This dataset include {} session in total'.format(len(subs_path)))
train_frame_action = {'cut_paper': 0, 'fold_paper': 0, 'pour_water':0, 'read_book':0, 'staple_paper':0, 'write_with_pen':0, 'write_with_pencil':0,\
    'pinch_and_drag':0, 'pinch_and_hold':0, 'swipe':0, 'tap':0, 'touch': 0}
val_frame_action = {'cut_paper': 0, 'fold_paper': 0, 'pour_water':0, 'read_book':0, 'staple_paper':0, 'write_with_pen':0, 'write_with_pencil':0,\
    'pinch_and_drag':0, 'pinch_and_hold':0, 'swipe':0, 'tap':0, 'touch': 0}
test_frame_action = {'cut_paper': 0, 'fold_paper': 0, 'pour_water':0, 'read_book':0, 'staple_paper':0, 'write_with_pen':0, 'write_with_pencil':0,\
    'pinch_and_drag':0, 'pinch_and_hold':0, 'swipe':0, 'tap':0, 'touch': 0}
total_frames = 0
train_frames = 0
val_frames = 0
test_frames = 0
total_seqs = 0
train_seqs = 0
val_seqs = 0
test_seqs = 0
total_frames_dark = 0
total_frames_light = 0
total_frames_kitchen = 0
total_frames_glove = 0
seqs_dark = 0
seqs_light = 0
seqs_kitchen = 0
seqs_glove = 0
unique_subject = []

for sub_path in tqdm(subs_path):
    sub_frames = 0
    if not sub_path.split('/')[-1][:10] in unique_subject:
        unique_subject.append(sub_path.split('/')[-1][:10])
    actions = sorted(os.listdir(sub_path))
    total_seqs += len(actions)
    for action in actions:
        action_path = os.path.join(sub_path, action)
        rgb_path = os.path.join(action_path, 'egocentric', 'ir')
        png_files = glob(os.path.join(rgb_path, '*.png'))
        if not action=='wash_hands_real' :
            total_frames += len(png_files)
            if sub_path.split('/')[-1] in train_split:
                train_frames += len(png_files)
                train_seqs += 1
                train_frame_action[action] += len(png_files) 
            if sub_path.split('/')[-1] in val_split:
                val_frames += len(png_files)
                val_seqs += 1
                val_frame_action[action] += len(png_files) 
            if sub_path.split('/')[-1] in test_split:
                test_frames += len(png_files)
                test_seqs +=1
                test_frame_action[action] += len(png_files) 
            sub_frames += len(png_files)
        if sub_path.split('/')[-1][11:15] == 'dark':
            total_frames_dark +=len(png_files)
            seqs_dark+=1
        if sub_path.split('/')[-1][11:17] == 'strong':
            total_frames_light +=len(png_files)
            seqs_light+=1
        if sub_path.split('/')[-1][11:14] == 'kit' and not action=='wash_hands_real' :
            total_frames_kitchen +=len(png_files)
            seqs_kitchen+=1
        if sub_path.split('/')[-1][11:16] == 'glove':
            total_frames_glove +=len(png_files)
            seqs_glove +=1
    print('Number of frames {} in the session {}'.format(sub_frames, sub_path.split('/')[-1]))
print('Number of frames {} in the dataset'.format(total_frames))
print('Number of training frames {} in the dataset'.format(train_frames))
print('Number of validation frames {} in the dataset'.format(val_frames))
print('Number of testing frames {} in the dataset'.format(test_frames))
print('Number of office frames {} in the dataset'.format(total_frames - total_frames_dark - total_frames_glove - total_frames_kitchen - total_frames_light))
print('Number of dark frames {} in the dataset'.format(total_frames_dark))
print('Number of light frames {} in the dataset'.format(total_frames_light))
print('Number of kitchen frames {} in the dataset'.format(total_frames_kitchen))
print('Number of glove frames {} in the dataset'.format(total_frames_glove))
print('Number of seqs {} in the dataset'.format(total_seqs))
print('Number of training seqs {} in the dataset'.format(train_seqs))
print('Number of validation seqs {} in the dataset'.format(val_seqs))
print('Number of testing seqs {} in the dataset'.format(test_seqs))
print('Number of office seqs {} in the dataset'.format(total_seqs - seqs_dark - seqs_glove - seqs_kitchen - seqs_light))
print('Number of dark seqs {} in the dataset'.format(seqs_dark))
print('Number of light seqs {} in the dataset'.format(seqs_light))
print('Number of kitchen seqs {} in the dataset'.format(seqs_kitchen))
print('Number of glove seqs {} in the dataset'.format(seqs_glove))
print('Number of unique subjects {} in the dataset'.format(len(unique_subject)))

for action in train_frame_action:
    print('Number of frames {} for action {} in train'.format(train_frame_action[action], action))
for action in val_frame_action:
    print('Number of frames {} for action {} in val'.format(val_frame_action[action], action))
for action in test_frame_action:
    print('Number of frames {} for action {} in test'.format(test_frame_action[action], action))

total_anno = 0
for save_sub in save_subs:
    if not save_sub[-4:] == 'odom':
        save_path = os.path.join(save_dir, save_sub)
        actions = os.listdir(save_path)
        for action in actions:
            action_path = os.path.join(save_path, action)
            gts = os.listdir(os.path.join(action_path, 'gt_info'))
            total_anno += len(gts)
print('Number of annotated frames {} in the dataset'.format(total_anno))

