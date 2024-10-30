import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # Check if bilinear upsampling is used
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Use transposed convolution for upsampling
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust shape if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2's size
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate along the channels dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        channels = [64, 128, 256, 512, 1024]

        # Encoder path
        self.inc = DoubleConv(n_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])   # Downsample to 1/2
        self.down2 = Down(channels[1], channels[2])  # Downsample to 1/4
        self.down3 = Down(channels[2], channels[3])  # Downsample to 1/8
        factor = 2 if bilinear else 1
        self.down4 = Down(channels[3], channels[4] // factor)  # Downsample to 1/16

        # Decoder path
        self.up1 = Up(channels[4], channels[3] // factor, bilinear)  # Upsample to 1/8
        self.up2 = Up(channels[3], channels[2] // factor, bilinear)   # Upsample to 1/4
        self.up3 = Up(channels[2], channels[1] // factor, bilinear)   # Upsample to 1/2
        self.up4 = Up(channels[1], channels[0], bilinear)              # Upsample to original size
        self.outc = OutConv(channels[0], n_classes)            # Final output layer

    def forward(self, x):
        x1 = self.inc(x)       # Initial conv
        x2 = self.down1(x1)    # Down 1
        x3 = self.down2(x2)    # Down 2
        x4 = self.down3(x3)    # Down 3
        x5 = self.down4(x4)    # Bottleneck

        x = self.up1(x5, x4)   # Up 1
        x = self.up2(x, x3)    # Up 2
        x = self.up3(x, x2)    # Up 3
        x = self.up4(x, x1)    # Up 4
        logits = self.outc(x)  # Output
        return logits