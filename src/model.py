import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        #Encoder 
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.enc2 = DSConv(32, 64, stride=2)
        self.enc3 = DSConv(64, 128, stride=2)
        self.enc4 = DSConv(128, 256, stride=2)

        # Bottleneck
        self.bottleneck = DSConv(256, 256)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DSConv(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DSConv(64, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = DSConv(32, 32)

        self.up4 = nn.ConvTranspose2d(32, 32, 2, stride=2)

        # Output Layer
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):

        input_size = x.shape[2:]   # save original size

        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x)
        x = self.dec1(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up3(x)
        x = self.dec3(x)

        x = self.up4(x)

        x = self.classifier(x)

        # Resize to input resolution
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x