# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchsummary import summary

class Generator(nn.Module):
    '''
    chart mapping latent coordinate to ambient coordinate
    '''

    def __init__(self, ngf = 64, md = 100, nc = 1):
        super(Generator, self).__init__()

        # Size of feature maps in generator
        self.ngf = ngf

        # Size of z latent vector (i.e. dimension of the manifold)
        self.md = md
        
        # Number of channels in the training images. For color images this is 3
        self.nc = 1

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( md, ngf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        input = input.reshape(input.shape[0],-1,1,1)
        return self.main(input)

def main():
    model = Generator()

    summary(model.cuda(), (100,1))

if __name__ == "__main__":
    main()