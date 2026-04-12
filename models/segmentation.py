"""Segmentation model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        # building conv block
        self.block=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        # forward through block
        return self.block(x)


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self,num_classes:int=3,in_channels:int=3,dropout_p:float=0.5):
        super().__init__()

        # encoder init
        self.encoder=VGG11Encoder(in_channels=in_channels)

        # decoder setup
        self.up5=nn.ConvTranspose2d(512,512,kernel_size=2,stride=2)
        self.dec5=ConvBlock(512+512,512)

        self.up4=nn.ConvTranspose2d(512,512,kernel_size=2,stride=2)
        self.dec4=ConvBlock(512+512,512)

        self.up3=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.dec3=ConvBlock(256+256,256)

        self.up2=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.dec2=ConvBlock(128+128,128)

        self.up1=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.dec1=ConvBlock(64+64,64)

        # final conv
        self.final_conv=nn.Conv2d(64,num_classes,kernel_size=1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # encoder pass
        bottleneck,features=self.encoder(x,return_features=True)

        # unpack features
        f1=features["block1"]
        f2=features["block2"]
        f3=features["block3"]
        f4=features["block4"]
        f5=features["block5"]

        # decoder stage 5
        x=self.up5(bottleneck)
        x=torch.cat([x,f5],dim=1)
        x=self.dec5(x)

        # decoder stage 4
        x=self.up4(x)
        x=torch.cat([x,f4],dim=1)
        x=self.dec4(x)

        # decoder stage 3
        x=self.up3(x)
        x=torch.cat([x,f3],dim=1)
        x=self.dec3(x)

        # decoder stage 2
        x=self.up2(x)
        x=torch.cat([x,f2],dim=1)
        x=self.dec2(x)

        # decoder stage 1
        x=self.up1(x)
        x=torch.cat([x,f1],dim=1)
        x=self.dec1(x)

        # final output
        x=self.final_conv(x)

        return x