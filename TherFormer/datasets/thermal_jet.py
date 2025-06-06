
import os
import lmdb

import numpy as np
from PIL import Image, ImageFile
import sys
sys.path.append('../../')
from datasets import h2outils
from datasets.queries import BaseQueries,TransQueries, get_trans_queries
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True
def read_json_gt(gt):
    f = open(gt)
    gt = json.load(f)
    joint3d_l = gt['kps3D_L']
    joint3d_r = gt['kps3D_R']
    joint3d=np.array(joint3d_l)
    joint3d=np.concatenate((joint3d,joint3d_r),axis=0)
    return joint3d

class H2OHands(object):
    def __init__(
        self,
        dataset_folder,
        split, 
        ntokens_pose, 
        ntokens_action, 
        spacing, 
        is_shifting_window,
        split_type="actions",
    ):
        super().__init__() 

        self.ntokens_pose=ntokens_pose
        self.ntokens_action=ntokens_action
        self.spacing=spacing
        self.is_shifting_window=is_shifting_window 

        self.all_queries = [
            BaseQueries.IMAGE,           
            BaseQueries.CAMINTR,
            
            TransQueries.JOINTS2D, 
            TransQueries.JOINTSABS25D,

            BaseQueries.JOINTS3D,
            BaseQueries.ACTIONIDX,
            BaseQueries.OBJIDX,
        ]
             
        self.counter = 0

        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries) 

        self.name = "thermal"
        split_opts = ["actions"]
        if split_type not in split_opts:
            raise ValueError("Split for dataset {} should be in {}, got {}".format(self.name, split_opts, split_type))

        self.split_type = split_type 
        self.root = dataset_folder
        self.rgb_root=dataset_folder
        
        self.reduce_res = True
        if self.reduce_res: 
            self.reduce_factor = 1 / 2
        else:
            self.reduce_factor = 1
            assert False, 'Warning-reduce factor is 1'

        self.split = split
 
        frame_segments,action_to_idx,object_to_idx=h2outils.get_segments_actions_objects_info_from_files(self.root)  

        self.frame_segments=frame_segments[self.split]
        self.object_to_idx=object_to_idx
        self.action_to_idx=action_to_idx
        
        self.num_actions = len(self.action_to_idx.keys())
        self.num_objects = len(self.object_to_idx.keys())
         
        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
 
        # Infor for rendering
        self.cam_intr=np.array([[409.5380 ,   0        , 323.1047],
       [  0       , 409.9164 , 253.5884],
       [  0      ,   0     ,   1     ]])
        self.image_size = [320,256] 
        self.load_dataset()
        self.cam_intr[:2] = self.cam_intr[:2] * self.reduce_factor
        if self.split=='train':
            try:
                self.env_r=lmdb.open(os.path.join(self.root,'lmdb_imgs',self.split),readonly=True,lock=False,readahead=False,meminit=False,\
                                map_size=(1024)**3,max_spare_txns=32,max_dbs=1000)
            except:
                self.env_r=None


    def load_dataset(self): 
        image_names = []
        mask_names = []
        joints2d = []
        joints3d = []
        
        hand_sides = []
        sample_infos = []
 
        action_idxs, obj_idxs = [], [] 
        
        for info_segments in self.frame_segments: 
            for iid in range(info_segments["start_idx"],info_segments["end_idx"]+1):
                csample_info={"subject":info_segments["subject"],
                        "scene":info_segments["scene"],
                        "sequence":info_segments["sequence"],
                        "frame_idx":iid,
                        "action_name":info_segments["action_name"],
                        "object_name":info_segments["object_name"],
                        "seq_idx":info_segments["segment_idx"]}
                sample_infos.append(csample_info)
                hand_sides.append('both')
                action_idxs.append(info_segments["action_idx"])
                obj_idxs.append(info_segments["object_idx"]) 

                relative_img_path=os.path.join(info_segments["subject"],info_segments["scene"],"jet_320_256","{:06d}.png".format(iid))
                relative_mask_path = os.path.join(info_segments["subject"],info_segments["scene"],"mask320_256","{:06d}.png".format(iid))
                image_names.append(relative_img_path)
                mask_names.append(relative_mask_path)
                
                # skel_camcoords=h2outils.get_skeleton(os.path.join(self.root,info_segments["subject"],info_segments["scene"],info_segments["sequence"],"cam4/hand_pose/","{:06d}.txt".format(iid)))
                skel_camcoords = read_json_gt(os.path.join(self.root,info_segments["subject"],info_segments["scene"],"gt_info","{:06d}.json".format(iid)))
                
                # skel_camcoords = np.zeros([42,3])

                depth2thermal = np.array([ [ 9.99969378e-01, -7.80614645e-03,  5.54376539e-04],
                            [ 7.81514128e-03,  9.99794958e-01, -1.86806296e-02],
                            [-4.08439138e-04,  1.86843901e-02,  9.99825348e-01]])
                depth2thermal_t = np.array([-0.0584895784077234566, 0.0137535585090518, -0.008942316725850105])

                print(self.counter)
                self.counter += 1

                skel_camcoords = np.dot(depth2thermal,skel_camcoords.T).T + depth2thermal_t
                skel_camcoords = skel_camcoords*1000# convert m to mm
                skel_camcoords = np.dot(depth2thermal.T,skel_camcoords.T).T + depth2thermal_t

                joints3d.append(skel_camcoords)
                hom_2d = np.array(self.cam_intr).dot(skel_camcoords.transpose()).transpose()
                skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]
                joints2d.append(skel2d)
                 
        annotations = {
            "image_names": image_names,
            "joints2d": joints2d,
            "joints3d": joints3d,
            "hand_sides": hand_sides,
            "sample_infos": sample_infos,
            "mask_names": mask_names,
            "action_idxs":action_idxs,
            "obj_idxs":obj_idxs, 
        }

          
        self.image_names = annotations["image_names"]        

        self.joints2d = annotations["joints2d"]
        self.joints3d = annotations["joints3d"] 
        self.hand_sides = annotations["hand_sides"]
        self.sample_infos = annotations["sample_infos"]
        self.mask_names = annotations["mask_names"]

        self.action_idxs = annotations["action_idxs"]
        self.obj_idxs=annotations["obj_idxs"] 
        window_starts,fulls=h2outils.get_seq_map(self.frame_segments,self.sample_infos,
                                                    ntokens_action=self.ntokens_action,
                                                    ntokens_pose=self.ntokens_pose,
                                                    spacing=self.spacing,
                                                    is_shifting_window=self.is_shifting_window)
         
        
        self.window_starts = window_starts
        self.fulls=fulls 
            
    
    def get_start_frame_idx(self, idx):
        idx=min(idx,len(self.window_starts)-1)
        return self.window_starts[idx]
             

    def get_dataidx(self, idx):
        idx=min(idx,len(self.fulls)-1)
        return self.fulls[idx]
    
    def open_seq_lmdb(self,idx):
        if self.split!='train':
            return 0,0
        idx=self.get_dataidx(idx)
        cur_seq_tag='{:04d}'.format(self.sample_infos[idx]["seq_idx"])
        subdb=self.env_r.open_db(cur_seq_tag.encode('ascii'),create=False)
        #subdbm = self.env_m.open_db(cur_seq_tag.encode('ascii'),create=False)
        txn=self.env_r.begin(write=False,db=subdb)
        # txnm = self.env_m.begin(write=False,db=subdbm)
        return txn#,txnm

    
    def get_image(self, idx, txn):
        idx = self.get_dataidx(idx)
        img_path = self.image_names[idx]
        
        if self.split=='train':
            buf=txn.get(img_path.encode('ascii'))
            img_flat=np.frombuffer(buf,dtype=np.uint8)
            img = img_flat.reshape(self.image_size[1],self.image_size[0],3).copy()
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
 
        else:
            img_path = os.path.join(self.root, img_path)
            img = Image.open(img_path).convert("RGB")
        return img 
    
    def get_mask(self, idx, txn):
        idx = self.get_dataidx(idx)
        img_path = self.mask_names[idx]
        if self.split=='train':
            buf=txn.get(img_path.encode('ascii'))
            img_flat=np.frombuffer(buf,dtype=np.uint8)
            img = img_flat.reshape(self.image_size[1],self.image_size[0],3).copy()
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB")
 
        else:
            img_path = os.path.join(self.root, img_path)
            img = Image.open(img_path).convert("RGB")
        return img 


    def get_joints3d(self, idx):
        idx = self.get_dataidx(idx)
        joints = self.joints3d[idx]
        return joints / 1000

    def get_joints2d(self, idx):
        idx = self.get_dataidx(idx)
        joints = self.joints2d[idx] * self.reduce_factor
        return joints

 
    def get_abs_joints25d(self,idx):
        idx=self.get_dataidx(idx)
        joints=self.joints3d[idx]/1000
        joints[:,:2]=self.joints2d[idx]*self.reduce_factor
        return joints
   
    

    def get_camintr(self, idx):
        idx = self.get_dataidx(idx)
        camintr = self.cam_intr
        return camintr.astype(np.float32)
 

    def get_action_idxs(self, idx):
        idx=self.get_dataidx(idx)
        action_idx=self.action_idxs[idx]
        return action_idx
    def get_obj_idxs(self,idx):
        idx=self.get_dataidx(idx)
        return self.obj_idxs[idx]
    

    def get_sample_info(self,idx):
        idx=self.get_dataidx(idx)
        sample_info=self.sample_infos[idx]
        return sample_info
   
    
    def get_future_frame_idx(self, cur_idx, fut_idx, spacing, verbose=False):
        cur_idx=self.get_dataidx(cur_idx)
        fut_idx=self.get_dataidx(fut_idx)
        cur_sample_info=self.sample_infos[cur_idx]
        fut_sample_info=self.sample_infos[fut_idx]

        if fut_sample_info["seq_idx"]!=cur_sample_info["seq_idx"] or \
                fut_sample_info["frame_idx"]-cur_sample_info["frame_idx"]!=spacing:
            fut_idx=cur_idx
            not_padding=0
        else:
            not_padding=1 
        return fut_idx,not_padding


    def __len__(self):
        return len(self.window_starts)

