import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
import torchvision
import torch
import torchvision.models as models
from torchvision.models import VGG16_BN_Weights
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models import vgg16_bn
from typing import Any, Dict, List, Optional, Tuple
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
import torchvision.ops as ops


# Helper function to add SSD's loc and conf layers
def add_ssd_prediction_layers(extras, bboxes, num_classes):
    loc_layers = []
    conf_layers = []


    # VGG - 512, 1024 channels for loc and conf layers
    loc_layers.extend([nn.Conv2d(512, bboxes[0] * 4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(512, bboxes[0] * num_classes, kernel_size=3, padding=1)])
    loc_layers.extend([nn.Conv2d(1024, bboxes[1] * 4, kernel_size=3, padding=1)])
    conf_layers.extend([nn.Conv2d(1024, bboxes[1] * num_classes, kernel_size=3, padding=1)])

    # Extras
    for k, v in enumerate(extras[1::2], 2):
        loc_layers.extend([nn.Conv2d(v.out_channels, bboxes[k] * 4, kernel_size=3, padding=1)])
        conf_layers.extend([nn.Conv2d(v.out_channels, bboxes[k] * num_classes, kernel_size=3, padding=1)])


    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
    in_channels, out_channels, kernel_size=2, stride=2
    )


def Extra():
    layers = []
    layers.extend([nn.Conv2d(1024, 256, kernel_size=1, stride=1)])
    layers.extend([nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)])
    layers.extend([nn.Conv2d(512, 128, kernel_size=1, stride=1)])
    layers.extend([nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)])
    layers.extend([nn.Conv2d(256, 128, kernel_size=1, stride=1)])
    layers.extend([nn.Conv2d(128, 256, kernel_size=3, stride=1)])
    layers.extend([nn.Conv2d(256, 128, kernel_size=1)])
    layers.extend([nn.Conv2d(128, 256, kernel_size=3, stride=1)])
    return nn.ModuleList(layers)


# L2Norm layer as defined in SSD
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x


# Modified Y_net to include SSD components and loc/conf predictions
class Y_Net(nn.Module):
    def __init__(self, num_classes, bboxes):  # Default to VOC dataset classes + background
        super(Y_Net, self).__init__()
        self.num_classes = num_classes
        self.bboxes = bboxes
        
        # Encoder (VGG16_bn) and UNet blocks
        self.encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = double_conv(512, 1024) # UNet bottleneck
        
        # UNet Decoding layers
        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = double_conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = double_conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # SSD components
        self.extras = Extra()
        self.l2norm = L2Norm(512, 20)
        
        # Adding SSD loc and conf layers
        self.loc, self.conf = add_ssd_prediction_layers(self.extras, self.bboxes, self.num_classes)

        self.conv_block5_to_1024 = nn.Conv2d(512, 1024, kernel_size=1)

        
    def forward(self, x):
        sources = []
        loc = []
        conf = []

        # UNet path
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        # Segmentation output
        x = self.up_conv6(x)
        x1_res = F.interpolate(x, size=(block5.size(2), block5.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x1_res, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x2_res = F.interpolate(x, size=(block4.size(2), block4.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x2_res, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x3_res = F.interpolate(x, size=(block3.size(2), block3.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x3_res, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x4_res = F.interpolate(x, size=(block2.size(2), block2.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x4_res, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x5_res = F.interpolate(x, size=(block1.size(2), block1.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x5_res, block1], dim=1)
        x = self.conv10(x)

        segmentation_output = self.conv11(x)

        # SSD path
        # Apply L2Norm on the 4th block output
        block4 = F.interpolate(block4, size=(38, 38), mode='bilinear', align_corners=False)
        x4 = self.l2norm(block4)
        sources.append(x4)

        
        x5 = self.conv_block5_to_1024(block5)
        x5 = F.interpolate(x5, size=(19, 19), mode='bilinear', align_corners=False)
        sources.append(x5)

        # Extra layers for SSD
        for k, v in enumerate(self.extras):
            x5 = F.relu(v(x5))
            if k % 2 == 1:
                sources.append(x5)

        #print(len(sources))
        # Apply loc and conf layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # Reshape for SSD output
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
 
        return segmentation_output, (loc, conf)
