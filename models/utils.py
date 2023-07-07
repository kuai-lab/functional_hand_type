import torch
import torch.nn.functional as torch_f
import torch.nn as nn
torch.set_printoptions(precision=4,sci_mode=False)
from datasets.queries import BaseQueries, TransQueries  



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
            print(k,v.size())
        else:
            print(k,v)

def recover_3d_proj_pinhole(camintr, est_scale, est_trans,off_z=0.4, input_res=(128, 128), verbose=False):
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

    
import math
def ae(y_true, y_pred, fir, sec, thd): 
    true_ba = y_true[:,fir] - y_true[:,sec]
    true_bc = y_true[:,thd] - y_true[:,sec]

    true_cosine_angle = torch.sum(true_ba * true_bc, dim=-1) / (torch.linalg.norm(true_ba, dim=-1) * torch.linalg.norm(true_bc, dim=-1) + 1e-9)
    true_angle = torch.acos(true_cosine_angle) / math.pi

    pred_ba = y_pred[:,fir] - y_pred[:,sec]
    pred_bc = y_pred[:,thd] - y_pred[:,sec]

    pred_cosine_angle = torch.sum(pred_ba * pred_bc, dim=-1) / (torch.linalg.norm(pred_ba, dim=-1) * torch.linalg.norm(pred_bc, dim=-1) + 1e-9)
    pred_angle = torch.acos(pred_cosine_angle) / math.pi

    return torch.abs(true_angle - pred_angle)

def compute_hand_loss(est2d,gt2d,estz,gtz,est3d,gt3d,weights,is_single_hand,pose_loss,verbose,data_type="fphab"):
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
        loss3d = pose_loss(est3d,gt3d,reduction='none')
        loss3d = torch.bmm(loss3d.view(loss3d.shape[0],-1,1),weights.view(-1,1,1))
        hand_losses["recov_joint3d"] = torch.sum(loss3d)/(loss3d.shape[1]*sum_weights)

        # lambda_3d = 0.5

        NCJ_gt = gt3d[:,0].unsqueeze(-2) - gt3d[:,1:]
        NCJ_pred = est3d[:,0].unsqueeze(-2) - est3d[:,1:]
        loss3d_NCJ = pose_loss(NCJ_gt, NCJ_pred, reduction='none')
        loss3d_NCJ = torch.bmm(loss3d_NCJ.view(loss3d_NCJ.shape[0],-1,1), weights.view(-1,1,1))
        hand_losses["recov_joint_NCJ"] = torch.sum(loss3d_NCJ) / (loss3d_NCJ.shape[1]*sum_weights)# * (1-lambda_3d)

        if data_type=='fphab':
            for finger_idx in range(5):
                if finger_idx == 0: loss3d_angle = ae(gt3d, est3d, 0, 1+finger_idx, 6+finger_idx) / 15
                else: loss3d_angle += ae(gt3d, est3d, 0, 1+finger_idx, 6+finger_idx) / 15
                loss3d_angle += ae(gt3d, est3d, 1+finger_idx, 6+finger_idx, 11+finger_idx) / 15
                loss3d_angle += ae(gt3d, est3d, 6+finger_idx, 11+finger_idx, 16+finger_idx) / 15
        
        loss3d_angle = torch.bmm(loss3d_angle.view(loss3d_angle.shape[0],-1,1), weights.view(-1,1,1))
        hand_losses["recov_joint_angle"] = torch.sum(loss3d_angle) / (loss3d_angle.shape[1]*sum_weights)

    return hand_losses