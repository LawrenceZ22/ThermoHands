import torch
import torch.nn.functional as torch_f
import torch.nn as nn
torch.set_printoptions(precision=4,sci_mode=False)
from datasets.queries import BaseQueries, TransQueries  

def resize_and_pad(tensor, new_height=224, new_width=224):
            # Calculate aspect ratio
            original_height = tensor.shape[2]
            original_width = tensor.shape[3]
            aspect_ratio = original_height / original_width

            # Determine new size
            if aspect_ratio > 1:  # Height is greater than width
                resize_width = new_width
                resize_height = round(aspect_ratio * new_width)
            else:  # Width is greater than height
                resize_height = new_height
                resize_width = round(new_height / aspect_ratio)

            # Resize step
            resize = transforms.Resize((resize_height, resize_width))
            resized_tensor = resize(tensor)

            # Padding step
            # Calculate padding
            pad_height = (new_height - resized_tensor.shape[2]) / 2
            pad_width = (new_width - resized_tensor.shape[3]) / 2

            # Ensure padding is integer
            pad_height_top = int(pad_height)
            pad_height_bottom = int(pad_height)
            pad_width_left = int(pad_width)
            pad_width_right = int(pad_width)

            # Apply padding
            padded_tensor = pad(resized_tensor, (pad_width_left, pad_height_top, pad_width_right, pad_height_bottom), fill=0)

            return padded_tensor

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def loss_str2func():
    return {'l1': torch_f.l1_loss, 'l2':torch_f.mse_loss}

def act_str2func():
    return {'softmax': nn.Softmax(),'elu':nn.ELU(),'leakyrelu':nn.LeakyReLU(),'relu':nn.ReLU()}


def torch2numpy(input):
    if input is None:
        return None
    if torch.is_tensor(input):
        input=input.detach().cpu().numpy()
    return input


def print_dict_torch(dict_):    
    for k,v in dict_.items():
        if torch.is_tensor(v):
            pass
            # print(k,v.size())
        # else:
        #     pass
            # print(k,v)

def recover_3d_proj_pinhole_thermal(camintr, est_scale, est_trans,off_z=0.4, input_res=(128, 128), verbose=False):
    # Estimate scale and trans between 3D and 2D
    focal = camintr[:, :1, :1]
    batch_size = est_trans.shape[0]
    num_joints = est_trans.shape[1]
    focal = focal.view(batch_size, 1, 1)
    est_scale = est_scale.view(batch_size, -1, 1)# z factor
    est_trans = est_trans.view(batch_size, -1, 2)# 2D x,y, img_center as 0,0
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2].view(batch_size,1,2).repeat(1,num_joints,1)
    # img_centers = (cam_centers.new(input_res) / 2).view(1, 1, 2).repeat(batch_size,num_joints, 1)

    est_xy0= est_trans
    est_XY0=(est_xy0-cam_centers) * est_Z0 / focal
    
    est_c3d = torch.cat([est_XY0, est_Z0], -1)
    return est_xy0,est_Z0, est_c3d


def recover_3d_proj_pinhole(camintr, est_scale, est_trans,off_z=0.7, input_res=(128, 128), verbose=False):
    # Estimate scale and trans between 3D and 2D
    # thermal .7
    #ir .5
    focal = camintr[:, :1, :1]
    batch_size = est_trans.shape[0]
    num_joints = est_trans.shape[1]
    focal = focal.view(batch_size, 1, 1)
    est_scale = est_scale.view(batch_size, -1, 1)# z factor
    est_trans = est_trans.view(batch_size, -1, 2)# 2D x,y, img_center as 0,0
    # est_scale is homogeneous to object scale change in pixels
    est_Z0 = focal * est_scale + off_z
    cam_centers = camintr[:, :2, 2].view(batch_size,1,2).repeat(1,num_joints,1)
    img_centers = (cam_centers.new(input_res) / 2).view(1, 1, 2).repeat(batch_size,num_joints, 1)

    est_xy0= est_trans+img_centers
    est_XY0=(est_xy0-cam_centers) * est_Z0 / focal
    
    est_c3d = torch.cat([est_XY0, est_Z0], -1)
    return est_xy0,est_Z0, est_c3d


class To25DBranch(nn.Module):
    def __init__(self, trans_factor=1, scale_factor=1):
        """
        Args:
            trans_factor: Scaling parameter to insure translation and scale
                are updated similarly during training (if one is updated 
                much more than the other, training is slowed down, because
                for instance only the variation of translation or scale
                significantly influences the final loss variation)
            scale_factor: Scaling parameter to insure translation and scale
                are updated similarly during training
        """
        super(To25DBranch, self).__init__()
        self.trans_factor = trans_factor
        self.scale_factor = scale_factor
        self.inp_res = [256, 256]

    def forward(self, sample, scaletrans, verbose=False):        
        batch_size = scaletrans.shape[0]
        trans = scaletrans[:, :, :2]
        scale = scaletrans[:, :, 2]
        final_trans = trans.view(batch_size,-1, 2)* self.trans_factor
        final_scale = scale.view(batch_size,-1, 1)* self.scale_factor
        height, width = tuple(sample[TransQueries.IMAGE].shape[2:])
        camintr = sample[TransQueries.CAMINTR].cuda() 
        
        est_xy0,est_Z0, est_c3d=recover_3d_proj_pinhole(camintr=camintr,est_scale=final_scale,est_trans=final_trans,input_res=(width,height), verbose=verbose)
        return {
            "rep2d": est_xy0, 
            "rep_absz": est_Z0,
            "rep3d": est_c3d,
        }


class To25DBranchThermal(nn.Module):
    def __init__(self, trans_factor=1, scale_factor=1):
        """
        Args:
            trans_factor: Scaling parameter to insure translation and scale
                are updated similarly during training (if one is updated 
                much more than the other, training is slowed down, because
                for instance only the variation of translation or scale
                significantly influences the final loss variation)
            scale_factor: Scaling parameter to insure translation and scale
                are updated similarly during training
        """
        super(To25DBranchThermal, self).__init__()
        self.trans_factor = 1
        self.scale_factor = scale_factor
        self.inp_res = [256, 256]

    def forward(self, sample, scaletrans, verbose=False):     
        height, width = tuple(sample[TransQueries.IMAGE].shape[2:])
        batch_size = scaletrans.shape[0]
        trans = scaletrans[:, :, :2]
        trans[:,:,0] = trans[:,:,0] * height
        trans[:,:,1] = trans[:,:,1] * width
        scale = scaletrans[:, :, 2]
        final_trans = trans.view(batch_size,-1, 2)* self.trans_factor
        final_scale = scale.view(batch_size,-1, 1)* self.scale_factor
        camintr = sample[TransQueries.CAMINTR].cuda() 
        
        est_xy0,est_Z0, est_c3d=recover_3d_proj_pinhole_thermal(camintr=camintr,est_scale=final_scale,est_trans=final_trans,input_res=(width,height), verbose=verbose)
        return {
            "rep2d": est_xy0, 
            "rep_absz": est_Z0,
            "rep3d": est_c3d,
        }
def compute_hand_loss(est2d,gt2d,estz,gtz,est3d,gt3d,weights,is_single_hand,pose_loss,verbose):
    hand_losses={}
    sum_weights=torch.where(torch.sum(weights)>0,torch.sum(weights),torch.Tensor([1]).cuda())[0]
    if not (est2d is None):
        loss2d=pose_loss(est2d,gt2d,reduction='none')
        loss2d=torch.bmm(loss2d.view(loss2d.shape[0],-1,1),weights.view(-1,1,1)) 

        hand_losses["recov_joints2d"]=torch.sum(loss2d)/(loss2d.shape[1]*sum_weights)
    if not(estz is None):        
        lossz=pose_loss(estz,gtz,reduction='none')
        lossz=torch.bmm(lossz.view(lossz.shape[0],-1,1),weights.view(-1,1,1))
        hand_losses["recov_joints_absz"]=torch.sum(lossz)/(lossz.shape[1]*sum_weights)
    if not (est3d is None):
        loss3d= pose_loss(est3d,gt3d,reduction='none')
        loss3d=torch.bmm(loss3d.view(loss3d.shape[0],-1,1),weights.view(-1,1,1))
        hand_losses["recov_joint3d"] = torch.sum(loss3d)/(loss3d.shape[1]*sum_weights)

    return hand_losses

