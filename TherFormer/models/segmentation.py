import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
class EfficientNetFeatureExtractor(torch.nn.Module):
    def __init__(self, effnet_version='efficientnet_v2_s',transformer_d_model=256):
        super().__init__()

        # self.fuse1 = DoubleConv(256,160)
        # self.fuse2 = DoubleConv(320,64)
        # self.fuse3 = DoubleConv(128,48)
        # self.fuse4 = DoubleConv(96,24)
        # self.mask_head = nn.Conv2d(48, 1, kernel_size=1)
        # self.transformer_d_model = transformer_d_model
        # self.head = SingleConv(256,512)
        self.latlayer1 = nn.Conv2d(160, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(64, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(48, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(24, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.mask_head = DoubleConv(256, 1, 64)
        self.smooth3 = nn.Conv2d(
            transformer_d_model, transformer_d_model, kernel_size=3, stride=1, padding=1
        )
        self.toplayer = nn.Conv2d(
            256, transformer_d_model, kernel_size=1, stride=1, padding=0
        )  # Reduce channels

        # Select the appropriate EfficientNetV2 variant
        if effnet_version == 'efficientnet_v2_s':
            self.base_net = create_model('tf_efficientnetv2_s', pretrained=True,features_only=True)
    
        else:
            raise ValueError("only efficientnet_v2_s is implemented")

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self,x): #FPN implementation modified from Deformer

        outputs = self.base_net(x)

        c1 = outputs[0]
        c2 = outputs[1]
        c3 = outputs[2]
        c4 = outputs[3]
        c5 = outputs[4]
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p1 = self._upsample_add(p2, self.latlayer4(c1))
        p1 = self.smooth3(p1) #b,256,128,160
        mask = self.mask_head(p1)
        
        return p1, mask


class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, resnet_version='resnet18',transformer_d_model=128):
        super().__init__()


        self.latlayer1 = nn.Conv2d(256, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.mask_head = nn.Sequential( 
            nn.Conv2d(transformer_d_model, transformer_d_model//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(transformer_d_model//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_d_model//2, 1, kernel_size=1, padding=0, bias=False))
        self.smooth3 = nn.Conv2d(
            transformer_d_model, transformer_d_model, kernel_size=3, stride=1, padding=1
        )
        self.toplayer = nn.Conv2d(
            512, transformer_d_model, kernel_size=1, stride=1, padding=0
        )  #

        # Select the appropriate EfficientNetV2 variant
        if resnet_version == 'resnet18':
            self.base_net  = create_model('resnet18', pretrained=True,features_only=True)
    
        else:
            raise ValueError("only resnet18 is implemented")

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y

    def forward(self,x): #FPN implementation modified from Deformer

        outputs = self.base_net(x)


        # c1 = outputs[0]
        c2 = outputs[1]
        c3 = outputs[2]
        c4 = outputs[3]
        c5 = outputs[4]
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # p1 = self._upsample_add(p2, self.latlayer4(c1))
        p2 = self.smooth3(p2) #b,256,128,160
        mask = self.mask_head(p2)
        
        return p2, mask


class ResNetFeatureExtractor_ml(torch.nn.Module):
    def __init__(self, resnet_version='resnet18',transformer_d_model=128,temporal_d_model=1024):
        super().__init__()
        
        self.latlayer1 = nn.Conv2d(256, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(
            transformer_d_model, transformer_d_model, kernel_size=3, stride=1, padding=1
        )
        self.toplayer = nn.Conv2d(
            512, transformer_d_model, kernel_size=1, stride=1, padding=0
        )  #
        self.mask_head = nn.Sequential( 
            nn.Conv2d(transformer_d_model, transformer_d_model//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(transformer_d_model//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_d_model//2, 1, kernel_size=1, padding=0, bias=False))
        self.layer1 = nn.Conv2d(128, transformer_d_model, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512, temporal_d_model, kernel_size=1, stride=1, padding=0)

        # Select the appropriate EfficientNetV2 variant
        if resnet_version == 'resnet50':
            self.base_net  = create_model('resnet50', pretrained=True,features_only=True)
        elif resnet_version == 'resnet18':
            self.base_net  = create_model('resnet18', pretrained=True,features_only=True)
        else:
            raise ValueError("only resnet50 is implemented")

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y
    def forward(self,x):
        

        outputs = self.base_net(x)
        high_level_feature = self.layer1(outputs[2])
        low_level_feature = self.layer2(outputs[4])
        
        # c2 = outputs[1]
        # c3 = outputs[2]
        # c4 = outputs[3]
        # c5 = outputs[4]
        # p5 = self.toplayer(c5)
        # p4 = self._upsample_add(p5, self.latlayer1(c4))
        # p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p2 = self._upsample_add(p3, self.latlayer3(c2))
        # p2 = self.smooth3(p2) #b,256,128,160

        
        mask = self.mask_head(high_level_feature)
        
        return high_level_feature, mask, low_level_feature



class ResNetFeatureExtractor_new(torch.nn.Module):
    def __init__(self, resnet_version='resnet34',transformer_d_model=128,temporal_d_model=1024):
        super().__init__()
        
       
        self.mask_head = nn.Sequential( 
            nn.Conv2d(transformer_d_model, transformer_d_model//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(transformer_d_model//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_d_model//2, 1, kernel_size=1, padding=0, bias=False))

        if resnet_version == 'resnet50':
            self.base_net  = create_model('resnet50', pretrained=True,features_only=True,out_indices=(0, 1, 2))
            self.layer1 = nn.Conv2d(512, transformer_d_model, kernel_size=1, stride=1, padding=0)
        elif resnet_version == 'resnet18':
            self.base_net  = create_model('resnet18', pretrained=True,features_only=True,out_indices=(0, 1, 2))
            self.layer1 = nn.Conv2d(128, transformer_d_model, kernel_size=1, stride=1, padding=0)
        elif resnet_version == 'resnet34':
            self.base_net  = create_model('resnet34', pretrained=True,features_only=True,out_indices=(0, 1, 2))
            self.layer1 = nn.Conv2d(128, transformer_d_model, kernel_size=1, stride=1, padding=0)
        else:
            raise ValueError("only resnet50 is implemented")

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y
    def forward(self,x):
        
        outputs =self.base_net(x)
        high_level_feature = self.layer1(outputs[2])
        mask = self.mask_head(high_level_feature)
        return high_level_feature, mask, None

import torch
import torch.nn as nn
class AdaptiveFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveFusionModule, self).__init__()
        self.linear = nn.Linear(out_channels,in_channels)

    def forward(self, F_L, F_C):
        # Concatenate features along the channel dimension
        F_concat = torch.cat((F_L, F_C), dim=-1)
        
        # Apply 3D convolution
        W = self.linear(F_concat)
        
        # Apply sigmoid activation
        sigma_W = torch.sigmoid(W)
        
        # Compute fused features
        F_F = sigma_W * F_L + (1 - sigma_W) * F_C
        
        
        return F_F

class ResCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1024):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, out_channels, kernel_size=3, padding=1, stride=2, bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(128*4*5,out_channels)

        # Match the dimensions for the residual connection
        # self.match_dimensions = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2**3, bias=True), # Adjust stride according to the downscaling in double_conv
        #     nn.BatchNorm2d(out_channels)
        # )
        
    def forward(self, x):
        # Prepare the residual (shortcut) connection
        # Get the main convolutional output
        x = self.double_conv(x)
        x = torch.flatten(x,1)
        # Add the shortcut connection to the main path
        x = self.linear(x)
        # Apply ReLU after combining the paths
        return x
    
class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1024):
        super().__init__()
        

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, kernel_size=3, padding=1, stride=2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
        
class FeatureFuser(torch.nn.Module):
     def __init__(self, t_dim, s_dim):
        super().__init__()
        self.layer_t = nn.Linear(t_dim,s_dim)
        self.final_layer = nn.Linear(s_dim,42)
    
     def forward(self,t_feature,s_feature):
         fused_feature = F.relu(self.layer_t(t_feature) + s_feature)
         return F.relu(self.final_layer(fused_feature))
         


class SimpleFPN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.toplayer = nn.Conv2d(
            in_channels, in_channels//2, kernel_size=1, stride=1, padding=0
        ) 
        self.latlayer1 = nn.Conv2d(out_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(in_channels//2, out_channels, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False) + y
    def forward(self, x_spatial, x_temporal):
        x_temporal = self.toplayer(x_temporal)
        fuse = self._upsample_add(x_temporal, self.latlayer1(x_spatial))
        smooth = self.smooth2(fuse)
        return smooth
class HandHeatmapLayer(nn.Module):
    def __init__(self, roi_height=60, roi_width=60, joint_nb=42):
        super(HandHeatmapLayer, self).__init__()

        self.out_height = roi_height
        self.out_width = roi_width
        self.joint_nb = joint_nb

        self.betas = nn.Parameter(torch.ones((self.joint_nb, 1), dtype=torch.float32))

        center_offset = 0.5
        vv, uu = torch.meshgrid(
            torch.arange(self.out_height, dtype=torch.float32), 
            torch.arange(self.out_width, dtype=torch.float32)
        )
        uu, vv = uu + center_offset, vv + center_offset
        self.register_buffer("uu", uu / self.out_width)
        self.register_buffer("vv", vv / self.out_height)

        self.softmax = nn.Softmax(dim=2)

    def spatial_softmax(self, latents):
        latents = latents.view((-1, self.joint_nb, self.out_height * self.out_width))
        latents = latents * self.betas
        heatmaps = self.softmax(latents)
        heatmaps = heatmaps.view(-1, self.joint_nb, self.out_height, self.out_width)
        return heatmaps

    def generate_output(self, heatmaps):
        predictions = torch.stack(
            (
                torch.sum(torch.sum(heatmaps * self.uu, dim=3), dim=2),
                torch.sum(torch.sum(heatmaps * self.vv, dim=3), dim=2),
            ),
            dim=2,
        )
        return predictions

    def forward(self, latent):
        heatmap = self.spatial_softmax(latent)
        prediction = self.generate_output(heatmap)
        return prediction

class ExtendedHandHeatmapLayer(HandHeatmapLayer):
    def __init__(self, roi_height=64, roi_width=80, joint_nb=42, depth_channels=1):
        super().__init__(roi_height, roi_width, joint_nb)
        self.depth_predictor = nn.Sequential(
            nn.Conv2d(joint_nb, depth_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(depth_channels, joint_nb),
            nn.ELU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(joint_nb * 2, joint_nb * 4), 
            nn.ReLU(),
            nn.Linear(joint_nb * 4, joint_nb * 2),  
            nn.Sigmoid()  # Ensure the coordinates are normalized (0-1)
        )

    def generate_output(self, heatmaps):
        xy_predictions = super().generate_output(heatmaps)
        joint_positions_flat = xy_predictions.view(xy_predictions.size(0), -1)
        refined_positions = self.fc_layers(joint_positions_flat)
        refined_positions = refined_positions.view_as(xy_predictions)
        
        # Flatten heatmaps for depth prediction to maintain spatial information
        depth_predictions = self.depth_predictor(heatmaps).unsqueeze(-1)  # Adding an extra dimension for consistency
        
        predictions = torch.cat((refined_positions, depth_predictions), dim=2)
        return predictions

    def forward(self, latent):
        heatmap = self.spatial_softmax(latent)
        prediction = self.generate_output(heatmap)
        return prediction

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class poseheadnosoftmax(nn.Module):
    def __init__(self,C):
        super(poseheadnosoftmax, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        self.flattened_size = 16 * 20 * 128
        # MLP layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 42*3) # Output size to match the target

    def forward(self, x):
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten
        x = torch.flatten(x,1)
        # MLP
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Reshape to the desired output shape (B, 42, 3)
        x = x.view(-1, 42, 3)
        return x  
    
class PoseheadNoSoftmaxNoFPN(nn.Module): #X
    def __init__(self,C,h,w):
        super(PoseheadNoSoftmaxNoFPN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        self.flattened_size = h * w * 128
        # MLP layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 42*3) # Output size to match the target

    def forward(self, x):
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten
        x = torch.flatten(x,1)
        # MLP
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Reshape to the desired output shape (B, 42, 3)
        x = x.view(-1, 42, 3)
        return x  
class PoseEstimationHead(nn.Module):
    def __init__(self, num_channels, num_joints, height, width,batch_size):
        super(PoseEstimationHead, self).__init__()
        self.num_joints = num_joints

        # Convolution layers for x, y coordinates and depth estimation
        # self.conv_xy = nn.Conv2d(num_channels, num_joints, kernel_size=1, bias=False)
        self.conv_depth = nn.Conv2d(num_channels, num_joints, kernel_size=1)
        self.linear = nn.Linear(height*width*num_joints, num_joints*3)

        # # Initialize the coordinate grid for x, y computation
        # x_map = torch.linspace(0, width - 1, steps=width).repeat(batch_size, num_joints, height, 1) / (width - 1)
        # y_map = torch.linspace(0, height - 1, steps=height).repeat(batch_size, num_joints, width, 1).transpose(2, 3) / (height - 1)
        
        # # Flatten and repeat the grid for each joint
        # self.register_buffer('x_map', x_map)
        # self.register_buffer('y_map', y_map)

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Apply convolutions to get feature maps for xy coordinates and depth
        # xy_map = self.conv_xy(x)  # Output shape: [batch_size, num_joints, height, width]
        depth_map = self.conv_depth(x)  # Output shape: [batch_size, num_joints, height, width]
        depth_map = torch.flatten(depth_map,1)
        depth_map = self.linear(depth_map)  # Output shape: [batch_size, num_joints*3]
        depth_map = F.relu(depth_map)
        output = depth_map.view(batch_size,-1,3)

        # # Spatial softmax for xy coordinates
        # xy_map = xy_map.view(batch_size, self.num_joints, -1)
        # xy_map = F.softmax(xy_map, dim=2)  # Apply softmax over the flattened spatial dimensions

        # # Ensure the grid is expanded to match the batch size
        # probabilities = xy_map.view(batch_size, self.num_joints, height, width)
    
    # # Compute the expected positions (normalized)
    #     x_coords = torch.sum(probabilities * self.x_map, dim=(2, 3))
    #     y_coords = torch.sum(probabilities * self.y_map, dim=(2, 3))
    
    #     xy_coords = torch.stack([x_coords, y_coords], dim=-1)
    #     # print(xy_coords)

        # Simplified global average pooling for depth estimation
        # depth = torch.mean(depth_map, dim=(2, 3))  # Output shape: [batch_size, num_joints]

        # Concatenate xy coordinates and depth values
        # output = torch.cat([xy_coords, depth.unsqueeze(-1)], dim=-1)  # Final output shape: [batch_size, num_joints, 3]

        return output

class DownsampleModule(nn.Module):
    def __init__(self):
        super(DownsampleModule, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # (B, 256, 64, 80)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # (B, 256, 32, 40)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # (B, 256, 16, 20)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # (B, 256, 8, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # x = self.conv4(x)
        return x
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.dice_loss = DiceLoss()
        pos_weight = torch.tensor([30])
        self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

    def forward(self, inputs, targets):
        # dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        return ce_loss#*1.5 + dice_loss*0.5 