import torch
import torch.nn as nn
from torchvision import models

class LiteDilatedHead(nn.Module):
    """
    Custom LiteDilatedHead: Replaces standard dilated convolutions with 
    Depthwise Separable Convolutions + Dilation to reduce parameters.
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[2, 2, 2, 2, 2, 2]):
        super(LiteDilatedHead, self).__init__()
        self.layers = nn.ModuleList()
        
        current_channels = in_channels
        for rate in dilation_rates:
            # Depthwise Separable Convolution with Dilation
            # 1. Depthwise: groups=in_channels
            dw = nn.Conv2d(current_channels, current_channels, kernel_size=3, 
                           padding=rate, dilation=rate, groups=current_channels, bias=False)
            # 2. Pointwise: kernel_size=1
            pw = nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=False)
            
            block = nn.Sequential(
                dw,
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True),
                pw,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.layers.append(block)
            current_channels = out_channels # Update for next layer if needed
            
        # Final 1x1 conv to generate density map (1 channel)
        self.output_layer = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

class MobileCSRNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileCSRNet, self).__init__()
        # Backbone: MobileNetV2
        # We take features until a certain stage. 
        # MobileNetV2 features usually have 32, 16, 24, 32, 64, 96, 160, 320 channels at different stages.
        # Let's assume we take the output with 96 channels (approx 1/16 or 1/8 resolution).
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.frontend = mobilenet.features[:14] # Adjust index based on desired feature map size/channels
        
        # Backend: LiteDilatedHead
        # Assuming input channels from frontend is 96
        self.backend = LiteDilatedHead(in_channels=96, out_channels=96)
        
        self._initialize_weights()

    def forward(self, x):
        features = self.frontend(x)
        density_map = self.backend(features)
        return density_map, features # Return features for distillation

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
