import numpy as np
import transformers
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd





        
##############################################################################################################
##############################################################################################################
class VGGBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        return x


##############################################################################################################
##############################################################################################################

class LightWeight_CNN(nn.Module):
    def __init__(self, input_shape, output_shape, num_vcg):
        super().__init__()
        self.num_vcg = num_vcg
        self.vgg_block1 = VGGBlock2(input_shape[0], 32)
        self.vgg_block2 = VGGBlock2(32, 64)
        self.vgg_block3 = VGGBlock2(64, 64)
        if self.num_vcg==1: self.fc1 = nn.Linear(int(input_shape[1]*input_shape[2]*32/4), 512)
        elif self.num_vcg==2: self.fc1 = nn.Linear(int(input_shape[1]*input_shape[2]*64/16), 512)
        elif self.num_vcg==3: self.fc1 = nn.Linear(int(input_shape[1]*input_shape[2]*64/64), 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        if self.num_vcg>=1:
            x = self.vgg_block1(x)
            if self.num_vcg>=2:
                x = self.vgg_block2(x)
                if self.num_vcg>=3:
                    x = self.vgg_block3(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##############################################################################################################
##############################################################################################################
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels) )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

##############################################################################################################
##############################################################################################################
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        block = BasicBlock

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
##############################################################################################################
##############################################################################################################

from torchvision.models import mobilenet_v2

def MobileNetV2(input_shape, output_shape):
    # Load the base MobileNetV2 model
    model = mobilenet_v2(pretrained=False)

    # Adjust the first convolution layer if input channels differ
    in_channels = input_shape[0]
    if in_channels != 3:
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # Adjust the classifier for the desired output shape
    model.classifier[1] = nn.Linear(model.last_channel, output_shape)

    return model
##############################################################################################################
##############################################################################################################
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights

def ResNet18(input_shape=(3, 224, 224), num_classes=10, pretrained=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Adjust for small input sizes
    if input_shape[1] < 64 or input_shape[2] < 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    return model


##############################################################################################################
##############################################################################################################
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

class ResNet10(ResNet):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10):
        # ResNet10 has [1, 1, 1, 1] blocks in each layer
        super(ResNet10, self).__init__(block=BasicBlock, layers=[1, 1, 1, 1])
        
        # Adjust for small input sizes
        if input_shape[1] < 64 or input_shape[2] < 64:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
##############################################################################################################
##############################################################################################################
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

class ResNet20(ResNet):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(ResNet20, self).__init__(block=BasicBlock, layers=[3, 3, 3, 0])

        if input_shape[1] < 64 or input_shape[2] < 64:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


##############################################################################################################
##############################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super(EfficientNet, self).__init__()

        # Load pretrained EfficientNet-B0 backbone
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()  # Remove original classifier

        # Determine the number of features output by the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_input_resized = nn.functional.interpolate(dummy_input, size=(224, 224), mode='bilinear')
            features = self.backbone(dummy_input_resized)
            feature_dim = features.shape[1]

        # Custom classifier
        self.fc1 = nn.Linear(feature_dim, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Resize input to match EfficientNet expected input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

##############################################################################################################
##############################################################################################################