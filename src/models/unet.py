import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super(Up3D, self).__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding if necessary
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    """Final output convolution"""

    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """3D U-Net architecture"""

    def __init__(self, n_channels, n_classes, trilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        factor = 2 if trilinear else 1

        channels = [64, 128, 256, 512, 1024]

        self.inc = DoubleConv3D(n_channels, channels[0])
        self.down1 = Down3D(channels[0], channels[1])
        self.down2 = Down3D(channels[1], channels[2])
        self.down3 = Down3D(channels[2], channels[3])
        self.down4 = Down3D(channels[3], channels[4] // factor)

        self.up1 = Up3D(channels[4], channels[3] // factor, trilinear)
        self.up2 = Up3D(channels[3], channels[2] // factor, trilinear)
        self.up3 = Up3D(channels[2], channels[1] // factor, trilinear)
        self.up4 = Up3D(channels[1], channels[0], trilinear)
        self.outc = OutConv3D(channels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)     # [B, 64, D, H, W]
        x2 = self.down1(x1)  # [B, 128, D/2, H/2, W/2]
        x3 = self.down2(x2)  # [B, 256, D/4, H/4, W/4]
        x4 = self.down3(x3)  # [B, 512, D/8, H/8, W/8]
        x5 = self.down4(x4)  # [B, 1024, D/16, H/16, W/16]

        x = self.up1(x5, x4)  # [B, 512, D/8, H/8, W/8]
        x = self.up2(x, x3)   # [B, 256, D/4, H/4, W/4]
        x = self.up3(x, x2)   # [B, 128, D/2, H/2, W/2]
        x = self.up4(x, x1)   # [B, 64, D, H, W]
        logits = self.outc(x) # [B, n_classes, D, H, W]
        return logits