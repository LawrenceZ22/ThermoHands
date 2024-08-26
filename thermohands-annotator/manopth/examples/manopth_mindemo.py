import torch
from manopth.manolayer import ManoLayer
from manopth import demo

batch_size = 1
# Select number of principal components for pose space
ncomps = 45

# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='/home/fangqiang/thermal-hand/manopth/mano/models', side = 'left', use_pca=True, ncomps=ncomps, flat_hand_mean=True)

# Generate random shape parameters
random_shape = torch.zeros(batch_size, 10) 
# Generate random pose parameters, including 3 values for global axis-angle rotation, 3 for translation
random_pose = torch.zeros(batch_size, ncomps + 6)

# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
hand_verts = hand_verts 
hand_joints = hand_joints 
demo.display_hand({
    'verts': hand_verts,
    'joints': hand_joints
},
                  mano_faces=mano_layer.th_faces, show=False, save=True)
