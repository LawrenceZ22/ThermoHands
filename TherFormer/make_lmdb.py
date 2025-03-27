import glob
import os,cv2
import numpy as np
 
from joblib import Parallel, delayed
from PIL import Image
import shutil
import lmdb

from datasets import h2ohands,thermal,thermal_cam, thermal_ir, thermal_depth
#Use lmdb to save image, and facilitate training
def convert_dataset_split_to_lmdb(dataset_name,dataset_folder,split):
    if dataset_name == 'ir':
        pose_dataset = thermal_ir.H2OHands(dataset_folder=dataset_folder,
                                    split=split,
                                    ntokens_pose=1,
                                    ntokens_action=1,
                                    spacing=1,
                                    is_shifting_window=False,
                                    split_type='actions')
    
    if dataset_name == 'thermal':
        pose_dataset = thermal.H2OHands(dataset_folder=dataset_folder,
                                    split=split,
                                    ntokens_pose=1,
                                    ntokens_action=1,
                                    spacing=1,
                                    is_shifting_window=False,
                                    split_type='actions')
    elif dataset_name == 'thermal_cam':
        pose_dataset = thermal_cam.H2OHands(dataset_folder=dataset_folder,
                                    split=split,
                                    ntokens_pose=1,
                                    ntokens_action=1,
                                    spacing=1,
                                    is_shifting_window=False,
                                    split_type='actions')




    image_names = pose_dataset.image_names
    mask_names = pose_dataset.mask_names
    sample_infos= pose_dataset.sample_infos


    rgb_root = pose_dataset.rgb_root
    image_path = os.path.join(rgb_root,image_names[0])
    mask_path = os.path.join(rgb_root, mask_names[0])
    data_size_per_img= np.array(Image.open(image_path).convert("RGB")).nbytes 
    data_size=data_size_per_img*len(image_names)

    dir_lmdb=os.path.join(dataset_folder,'lmdb_imgs',split)
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*10,max_dbs=1000)
    pre_seq_tag=''
    commit_interval=100
    for idx in range(0,len(image_names)):
        if dataset_name=='fhbhands':
            cur_seq_tag='_'.join(image_names[idx].split('/')[:-1])
        else:
            cur_seq_tag='{:04d}'.format(sample_infos[idx]["seq_idx"])
        if cur_seq_tag!=pre_seq_tag:
            pre_seq_tag=cur_seq_tag
            print(cur_seq_tag)

            if idx>0:
                txn.commit()
        
            subdb=env.open_db(cur_seq_tag.encode('ascii'))
            txn=env.begin(db=subdb,write=True)

        key_byte = image_names[idx].encode('ascii')
        image_path = os.path.join(rgb_root,image_names[idx])
        data = np.array(Image.open(image_path).convert("RGB"))
        print(idx,image_names[idx])
        txn.put(key_byte,data)

        if (idx+1)%commit_interval==0:
            txn.commit()
            txn=env.begin(db=subdb,write=True)

    txn.commit()
    env.close()
    
    data_size_per_img= np.array(Image.open(mask_path).convert("RGB")).nbytes 
    data_size=data_size_per_img*len(mask_names)
    dir_lmdb=os.path.join(dataset_folder,'lmdb_imgs',split)
    if not os.path.exists(dir_lmdb):
        os.makedirs(dir_lmdb)

    env = lmdb.open(dir_lmdb,map_size=data_size*10,max_dbs=1000)
    pre_seq_tag=''
    commit_interval=100
    for idx in range(0,len(mask_names)):
        if dataset_name=='fhbhands':
            cur_seq_tag='_'.join(image_names[idx].split('/')[:-1])
        else:
            cur_seq_tag='{:04d}'.format(sample_infos[idx]["seq_idx"])
        if cur_seq_tag!=pre_seq_tag:
            pre_seq_tag=cur_seq_tag
            print(cur_seq_tag)

            if idx>0:
                txn.commit()
        
            subdb=env.open_db(cur_seq_tag.encode('ascii'))
            txn=env.begin(db=subdb,write=True)

        key_byte = mask_names[idx].encode('ascii')
        mask_path = os.path.join(rgb_root,mask_names[idx])
        data = np.array(Image.open(mask_path).convert("RGB"))
        print(idx,mask_names[idx])
        txn.put(key_byte,data)

    txn.commit()
    env.close()
    
convert_dataset_split_to_lmdb('thermal',"/mnt/data/sunglare/",'train')