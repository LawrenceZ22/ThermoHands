import torch
import numpy as np
import torch.nn.functional as torch_f
from einops import rearrange
from torchvision.transforms import Resize
import time
import torch.nn.functional as F
import torch.nn as nn
from models import resnet
from models.transformer import Transformer_Encoder, PositionalEncoding
from models.utils import  To25DBranchThermal,compute_hand_loss,loss_str2func,To25DBranch
from models.mlp import MultiLayerPerceptron
from datasets.queries import BaseQueries, TransQueries
import matplotlib.pyplot as plt
import sys
from models.segmentation import ResCNN, ResNetFeatureExtractor_new,MaskLoss,AdaptiveFusionModule,poseheadnosoftmax,SimpleCNN, DoubleConv,DownsampleModule,ResNetFeatureExtractor, SimpleFPN,MLP,HandHeatmapLayer,PoseEstimationHead
from models.deformable_transformer import DeformableTransformerEncoderLayer,DeformableTransformerEncoder 
class ResNet_(torch.nn.Module):
    def __init__(self,resnet_version=18):
        super().__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            self.base_net = resnet.resnet18(pretrained=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            self.base_net = resnet.resnet50(pretrained=True)
        else:
            self.base_net=None
    
    
    def forward(self, image):
        features, res_layer5 = self.base_net(image)
        return features, res_layer5
def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

class TemporalNet(torch.nn.Module):
    def __init__(self,  is_single_hand,
                        transformer_d_model,
                        transformer_dropout,
                        transformer_nhead,
                        transformer_dim_feedforward,
                        transformer_num_encoder_layers_action,
                        transformer_num_encoder_layers_pose,
                        transformer_normalize_before=True,

                        lambda_action_loss=None,
                        lambda_hand_2d=None,
                        lambda_hand_z=None,
                        ntokens_pose=1,
                        ntokens_action=1,
                        
                        dataset_info=None,
                        trans_factor=100,
                        scale_factor=0.0001,
                        pose_loss='l2',
                        dim_grasping_feature=128,):

        super().__init__()
        
        self.ntokens_pose= ntokens_pose
        self.ntokens_action=ntokens_action

        self.pose_loss=loss_str2func()[pose_loss]
        
        self.lambda_hand_z=lambda_hand_z
        self.lambda_hand_2d=lambda_hand_2d        
        self.lambda_action_loss=lambda_action_loss


        self.is_single_hand=is_single_hand
        self.num_joints=21 if self.is_single_hand else 42

        num_feature_levels = 1
        transformer_d_model = 1024
        #Image Feature
        t_transformer_dim = 512
        self.meshregnet = ResNetFeatureExtractor_new(transformer_d_model=transformer_d_model,temporal_d_model=t_transformer_dim,resnet_version='resnet18')
        
        self.transformer_pe=PositionalEncoding(d_model=transformer_d_model) 
        # self.ad_fusion = AdaptiveFusionModule(t_transformer_dim,t_transformer_dim*2)

        encoder_layer = DeformableTransformerEncoderLayer(d_model=transformer_d_model,n_levels=num_feature_levels,n_points=8)
        self.encoder = DeformableTransformerEncoder(encoder_layer, 2)
        self.transformer_temporal=Transformer_Encoder(d_model=t_transformer_dim, 
                                nhead=transformer_nhead, 
                                num_encoder_layers=transformer_num_encoder_layers_pose,
                                dim_feedforward=transformer_dim_feedforward,
                                dropout=0.0, 
                                activation="relu", 
                                normalize_before=transformer_normalize_before)
                
        self.t_transformer_pe=PositionalEncoding(d_model=t_transformer_dim) 
        self.image_to_hand_pose=MultiLayerPerceptron(base_neurons=[t_transformer_dim, t_transformer_dim,t_transformer_dim], out_dim=self.num_joints*3,
                                act_hidden='leakyrelu',act_final='none')    
        self.simplecnn = ResCNN(transformer_d_model,t_transformer_dim)
       
        self.scale_factor = scale_factor 
        self.trans_factor = trans_factor   
        self.postprocess_hand_pose=To25DBranch(trans_factor=self.trans_factor,scale_factor=self.scale_factor)

        self.loss_fn = MaskLoss()

        self.loss_norm = False
        self.mask_loss = True

    
    def forward(self, batch_flatten,  verbose=False):       
        flatten_images = batch_flatten[TransQueries.IMAGE].cuda()
        #Loss
        total_loss = torch.Tensor([0]).cuda()
        losses = {}
        results = {}
        gt_mask = batch_flatten[TransQueries.MASK].cuda()
        gt_mask = (gt_mask > 0.5).float() #contain other values because of previous resizing

        h_feature, pred_mask, l_feature =self.meshregnet(flatten_images) 
        b_t, c, h, w = h_feature.shape
        pred_mask = torch.sigmoid(pred_mask) ##1: hand; 0: background 
        gt_mask = gt_mask[:,:1,:,:]
        gt_mask = F.interpolate(gt_mask, size=(pred_mask.shape[-2:]),mode='nearest')
        mask_loss = self.loss_fn(pred_mask,gt_mask)

        # Spatial attention for high level feature
        feature = h_feature 

        src_flatten = feature.view(-1,feature.shape[-3],feature.shape[-2]*feature.shape[-1]) # B*T, W*H, C
        src_flatten =  torch.transpose(src_flatten, 1, 2)
        mask_flatten = pred_mask.view(pred_mask.shape[0],pred_mask.shape[-2]*feature.shape[-1])
        masks = pred_mask.squeeze(1).unsqueeze(0)
        lvl_pos_embed_flatten = self.transformer_pe(src_flatten)
        spatial_shapes = [(feature.shape[-2],feature.shape[-1])]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        masks = masks > 0.9 #for reference point selection
        mask_flatten = mask_flatten < 0.1 #padding mask for attention
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, masks, lvl_pos_embed_flatten, mask_flatten)
        # b*t,HW,C 
        # Fuse spatial inforamtion to channel to reduce feature map
        memory = torch.transpose(memory, 1, 2)
        memory = memory.view(-1,c,h,w) 
     

        spatial_low_dim_feature = self.simplecnn(memory)
        spatial_low_dim_feature = spatial_low_dim_feature.view(-1,self.ntokens_pose,spatial_low_dim_feature.shape[-1])#B,T,C
        

        # # temporal attention for low level feature
        pos_embed_t = self.t_transformer_pe(spatial_low_dim_feature)
        batch_seq_pweights=batch_flatten['not_padding'].cuda().float().view(-1,self.ntokens_pose)
        batch_seq_pweights[:,0]=1.
        batch_seq_pmasks=(1-batch_seq_pweights).bool()
        batch_seq_pout_feature,_=self.transformer_temporal(src=spatial_low_dim_feature, src_pos=pos_embed_t,
                            key_padding_mask=batch_seq_pmasks, verbose=False)
        

        flatten_pout_feature=torch.flatten(batch_seq_pout_feature,start_dim=0,end_dim=1)
        
        #hand pose
        flatten_hpose=self.image_to_hand_pose(flatten_pout_feature)
        

        flatten_hpose=flatten_hpose.view(-1,self.num_joints,3)
        flatten_hpose_25d_3d=self.postprocess_hand_pose(sample=batch_flatten,scaletrans=flatten_hpose,verbose=verbose) 

        weights_hand_loss=batch_flatten['not_padding'].cuda().float()
        hand_results,total_loss,hand_losses=self.recover_hand(flatten_sample=batch_flatten,flatten_hpose_25d_3d=flatten_hpose_25d_3d,weights=weights_hand_loss,
                        total_loss=total_loss,verbose=verbose)    
  
        results.update(hand_results)
        losses.update(hand_losses)

        losses.update({"mask_loss": mask_loss})
        if self.mask_loss:
            if self.loss_norm:
                total_loss += mask_loss/ 4 / (1e-9 + mask_loss.detach())
            else:
                mask_loss = torch.clamp(mask_loss, max=total_loss/4)
                total_loss += mask_loss * 1
        losses.update({"total_loss": total_loss})

        
        
        
        return total_loss, results, losses
    
    def recover_hand(self, flatten_sample, flatten_hpose_25d_3d, weights, total_loss,verbose=False):
        hand_results, hand_losses={},{}
        
        joints3d_gt = flatten_sample[BaseQueries.JOINTS3D].cuda()
        hand_results["gt_joints3d"]=joints3d_gt         
        hand_results["pred_joints3d"]=flatten_hpose_25d_3d["rep3d"].detach().clone()
        hand_results["pred_joints2d"]=flatten_hpose_25d_3d["rep2d"]
        hand_results["pred_jointsz"]=flatten_hpose_25d_3d["rep_absz"]
 
        hpose_loss=0.
        
        joints25d_gt = flatten_sample[TransQueries.JOINTSABS25D].cuda()
        hand_losses=compute_hand_loss(est2d=flatten_hpose_25d_3d["rep2d"],
                                    gt2d=joints25d_gt[:,:,:2],
                                    estz=flatten_hpose_25d_3d["rep_absz"],
                                    gtz=joints25d_gt[:,:,2:3],
                                    est3d=flatten_hpose_25d_3d["rep3d"],
                                    gt3d= joints3d_gt,
                                    weights=weights,
                                    is_single_hand=self.is_single_hand,
                                    pose_loss=self.pose_loss,
                                    verbose=verbose)

        if self.loss_norm:
            hpose_loss+=  hand_losses["recov_joints2d"]/(hand_losses["recov_joints2d"].detach() + 1e-9)+ hand_losses["recov_joints_absz"]/((hand_losses["recov_joints_absz"].detach() + 1e-9))
        else:
            hpose_loss+=hand_losses["recov_joints2d"]*self.lambda_hand_2d+ hand_losses["recov_joints_absz"]*self.lambda_hand_z + hand_losses["recov_joint3d"] *self.lambda_hand_z
        

        if total_loss is None:
            total_loss= hpose_loss
        else:
            total_loss += hpose_loss

        return hand_results, total_loss, hand_losses

    def predict_object(self,sample,features, weights, total_loss,verbose=False):
        olabel_feature=features
        out=self.obj_classification(olabel_feature)
        
        olabel_results, olabel_losses={},{}
        olabel_gts=sample[BaseQueries.OBJIDX].cuda()
        olabel_results["obj_gt_labels"]=olabel_gts
        olabel_results["obj_pred_labels"]=out["pred_labels"]
        olabel_results["obj_reg_possibilities"]=out["reg_possibilities"]

        
        olabel_loss = torch_f.cross_entropy(out["reg_outs"],olabel_gts,reduction='none')
        olabel_loss = torch.mul(torch.flatten(olabel_loss),torch.flatten(weights))

            
        olabel_loss=torch.sum(olabel_loss)/torch.sum(weights)
        

        if total_loss is None:
            total_loss=self.lambda_action_loss*olabel_loss
        else:
            total_loss+=self.lambda_action_loss*olabel_loss
            olabel_losses["olabel_loss"]=olabel_loss
        return olabel_results, total_loss, olabel_losses


    def predict_action(self,sample,features,weights,total_loss=None,verbose=False):
        action_feature=features
        out=self.action_classification(action_feature)
        
        action_results, action_losses={},{}
        action_gt_labels=sample[BaseQueries.ACTIONIDX].cuda()[0::self.ntokens_action].clone()
        action_results["action_gt_labels"]=action_gt_labels
        action_results["action_pred_labels"]=out["pred_labels"]
 
        action_results["action_reg_possibilities"]=out["reg_possibilities"]
        action_loss = torch_f.cross_entropy(out["reg_outs"],action_gt_labels,reduction='none')  
        action_loss = torch.mul(torch.flatten(action_loss),torch.flatten(weights)) 
        action_loss=torch.sum(action_loss)/torch.sum(weights) 

        if total_loss is None:
            total_loss=self.lambda_action_loss*action_loss
        else:
            total_loss+=self.lambda_action_loss*action_loss
        action_losses["action_loss"]=action_loss
        return action_results, total_loss, action_losses