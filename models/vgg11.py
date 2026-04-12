"""VGG11 encoder
"""

from typing import Dict,Tuple,Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns."""

    def __init__(self,in_channels:int=3,use_batchnorm:bool=True):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        self.use_batchnorm=use_batchnorm  #flag for bn

        #block1 init
        self.block1=self._make_block(in_channels,64,num_convs=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)

        #block2 init
        self.block2=self._make_block(64,128,num_convs=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)

        #block3 init
        self.block3=self._make_block(128,256,num_convs=2)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)

        #block4 init
        self.block4=self._make_block(256,512,num_convs=2)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2)

        #block5 init
        self.block5=self._make_block(512,512,num_convs=2)
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2)

    def _make_block(self,in_channels:int,out_channels:int,num_convs:int)->nn.Sequential:
        layers=[]  # storing layers

        for i in range(num_convs):
            #choosing input channels
            conv_in=in_channels if i==0 else out_channels
            layers.append(nn.Conv2d(conv_in,out_channels,kernel_size=3,padding=1))
            
            #optional bn
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            #activation layer
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(
        self,x:torch.Tensor,return_features:bool=False
    )->Union[torch.Tensor,Tuple[torch.Tensor,Dict[str,torch.Tensor]]]:
        """Forward pass."""

        features={}  #storing skips

        #block1 forward
        x=self.block1(x)
        features["block1"]=x
        x=self.pool1(x)

        #block2 forward
        x=self.block2(x)
        features["block2"]=x
        x=self.pool2(x)

        #block3 forward
        x=self.block3(x)
        features["block3"]=x
        x=self.pool3(x)

        #block4 forward
        x=self.block4(x)
        features["block4"]=x
        x=self.pool4(x)

        #block5 forward
        x=self.block5(x)
        features["block5"]=x
        x=self.pool5(x)

        #returning features if needed
        if return_features:
            return x,features
        return x
    

class VGG11(nn.Module):
    """
    Autograder-compatible VGG11 model.
    Wraps VGG11Encoder.
    """

    def __init__(self,in_channels:int=3):
        super().__init__()

        #default encoder setup
        self.encoder=VGG11Encoder(in_channels=in_channels,use_batchnorm=True)

    def forward(self,x:torch.Tensor):
        #simple forward
        return self.encoder(x)