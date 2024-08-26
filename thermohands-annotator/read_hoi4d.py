import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch
from manopth.manolayer import ManoLayer

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True, save_path = None):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    plt.clf()

theta_src = [
            -2.0602667331695557,
            0.7926902174949646,
            -1.3996726274490356,
            0.48853379487991333,
            0.1733083873987198,
            -0.21793615818023682,
            0.26516902446746826,
            -0.18705017864704132,
            0.09728702157735825,
            0.0045403665862977505,
            0.13168810307979584,
            -0.3220592141151428,
            0.3330320715904236,
            0.07424019277095795,
            -0.2946649491786957,
            0.1792285293340683,
            -0.03710220381617546,
            0.12757404148578644,
            0.0003909860097337514,
            -0.011972298845648766,
            -0.3487284481525421,
            -0.7118625044822693,
            -0.09315446764230728,
            0.7091787457466125,
            -0.3005821704864502,
            0.12225812673568726,
            0.13498243689537048,
            -0.31813371181488037,
            0.28170472383499146,
            0.04884650558233261,
            0.5453094840049744,
            0.09638706594705582,
            -0.2411850392818451,
            -0.10873077809810638,
            -0.17021068930625916,
            0.0007620035903528333,
            -0.006946500390768051,
            -0.0335218720138073,
            -0.3794999420642853,
            0.9491039514541626,
            0.1508944034576416,
            0.3700357675552368,
            -0.20840688049793243,
            -0.07369515299797058,
            0.08233315497636795,
            -0.6786583065986633,
            0.11744888871908188,
            -0.7844653129577637
        ]
mano_layer = ManoLayer(mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', use_pca=True, flat_hand_mean=True, side = 'left')
pkl_path = "/home/fangqiang/handpose/refinehandpose_right/ZY20210800003/H3/C20/N11/S282/s02/T2/99.pickle"
f = open(pkl_path, 'rb')
hand_info = pickle.load(f, encoding='latin1')
theta = torch.FloatTensor(hand_info['poseCoeff']).unsqueeze(0)
beta = torch.FloatTensor(hand_info['beta']).unsqueeze(0)
trans = torch.FloatTensor(hand_info['trans']).unsqueeze(0)
# theta = torch.FloatTensor(theta_src).unsqueeze(0)
# theta[0, 2:] = 0
# theta[0, 1] +=1
# theta[0, 26:29] += 2
verts, joints = mano_layer(theta, beta, trans)
display_hand({
            'verts': verts.cpu().detach(),
            'joints': joints.cpu().detach()
    },
        mano_faces=mano_layer.th_faces.cpu().detach(), alpha = 0.9, save_path='hoi4d_left.png')
#  1.8652408 ,  0.23306954, -3.9743845 ,  1.8470424 , -2.7024195 ,
#  -1.6979216 , -2.3543622 , -1.6505556 , -3.6711211 , -4.9681177 
# 0.1453711 , -2.8722289 , -2.5535254 ,  2.5632253 ,  0.55347234,
# -3.3307893 , -0.42183232, -0.8576401 , -3.2785273 , -1.6602439 
#   2.3225, -1.3118, -5.0235, -5.0194, -0.2945,  3.8039,  
# 6.9123, -8.1884, -3.9032,  0.6501
      
   