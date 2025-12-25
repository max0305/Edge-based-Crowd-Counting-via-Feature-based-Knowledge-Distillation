import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CrowdResNet18(nn.Module):
    def __init__(self):
        super(CrowdResNet18, self).__init__()
        # load pre-trained ResNet18
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Extract layers up to layer3 and modify strides for higher resolution
        # we modify the strides of layer3 to keep feature map at 1/8 size
        
        self.frontend = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, # 1/4
            base.layer2, # 1/8
            base.layer3,  # 1/16
        )
        
        # Make layer3 stride 1 to keep 1/8 resolution
        self.frontend[6][0].conv1.stride = (1, 1)
        self.frontend[6][0].downsample[0].stride = (1, 1)

        # Backend to generate density map
        # Layer3 output has 256 channels
        self.backend = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1) # density map output
        )
        
        self._init_weights()

    def forward(self, x):
        features = self.frontend(x)
        out = self.backend(features)
        return out, features

    def _init_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
