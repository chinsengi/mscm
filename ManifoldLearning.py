import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet, ResNetBasicBlock
from generator import Generator
from torchsummary import summary
import utils

class ManifoldLearning(nn.Module):
    '''
    chart mapping from manifold to latent space
    '''

    def __init__(self, nchannels, imsize, manifold_dim):
        super(ManifoldLearning, self).__init__()
        self.encoder = ResNet(nchannels, manifold_dim, block=ResNetBasicBlock, deepths=[2, 2, 2])
        self.decoder = Generator(md = manifold_dim, nc = nchannels)
        self.md = manifold_dim
        self.nchannels = nchannels
        self.imsize = imsize

    def forward(self, x):

        # x.reshape(x.shape[0], self.nchannels, self.imsize, self.imsize)
        latent_x = self.encoder(x)

        #project to embedded manifold 
        # latent_x = ambient_x[:, 0:self.md]
        # loss_embedding = torch.norm(ambient_x[:,self.md:])/(ambient_x.shape[1]-self.md)
        # loss_embedding = loss_embedding-torch.mean(utils.standard_normal_logprob(latent_x))/100
        # breakpoint()
        # map the latent representation back to the manifold
        resx = self.decoder(latent_x)

        # reconstruction loss
        loss = n.MSELoss()(x, resx)
        return loss, resx, latent_x


def main():
    batch_size, channels, height, width = 100, 1, 32, 32
    inputs = torch.rand(batch_size, channels, height, width)
    model = ManifoldLearning(channels, height, 100)
    output = model(inputs)
    print(output)
    

if __name__ == '__main__':
    main()
    