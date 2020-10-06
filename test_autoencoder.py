import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
import torch.nn.functional as F
from torchvision.utils import save_image

import autoencoder
import toy_data as toy_data
import utils as utils
from visualize_flow import visualize_transform

from train_misc import standard_normal_logprob
from train_misc import build_model_augment
from bad_grad_viz import register_hooks
from autoencoder import encoder

parser = argparse.ArgumentParser('Koopman Flow')
parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--toy", type=eval, default=False, choices=[True, False])
parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church'], type=str, default="mnist")
parser.add_argument("--imagesize", type=int, default=28)
parser.add_argument(
    "--max_grad_norm", type=float, default=1e10,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
)
parser.add_argument("--depth", type=int, default=5, help='Number of Koopman layers.')
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=utils.NONLINEARITIES)

parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--coeff', type=float, default=10)
parser.add_argument("--num_epochs", type=int, default=100000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=200)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--pretrain', type=eval, default=False, choices=[True, False])
parser.add_argument('--resume', type=int, default=800, help='the epoch to resume')
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#send data to gpu
cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#end

def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x
#end

def get_train_loader(train_set, epoch):
    current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    logger.info("===> Using batch size {}. Total {} iterations/epoch.".format(current_batch_size, len(train_loader)))
    return train_loader
#end

def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root="./data", split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root="./data", split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ]), download=True
        )
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == 'celeba':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            train=True, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.CelebA(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            'data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.LSUN(
            'data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    data_shape = (im_dim, im_size, im_size)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape
#end get_data_set

if __name__ == '__main__':

    #build model
    model = encoder(args).to(device)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    #load data 
    if args.toy is False:
        train_set, test_loader, data_shape = get_dataset(args)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    
    if args.pretrain:
        checkpt = torch.load(os.path.join(args.save,'epoch-{}.pth'.format(args.resume)))
        model.load_state_dict(checkpt['model_state_dict'])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2], gamma=0.2)
        
    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')

    model.train()
    for epoch in range(1, args.num_epochs + 1):
        train_loader = get_train_loader(train_set, epoch)
        model.train()
        for itr, (x,y) in enumerate(train_loader):

            # cast data and move to device
            x = cvt(x)

            optimizer.zero_grad()
            _, loss = model(x)
            loss_meter.update(loss.item())

            loss.backward()                

            optimizer.step()

            time_meter.update(time.time() - end)

            log_message = (
                    'Iter {} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
                        str(itr), time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg
                    )
                )
            
            logger.info(log_message)
            end = time.time()
            
       
    #end

    