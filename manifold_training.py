import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import math
import numpy as np
import os
import time
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from ManifoldLearning import ManifoldLearning
from mlp import MLP
import utils

from utils import jacobian

parser = argparse.ArgumentParser('Manifold Score Matching')
parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument('--alp', type=float, default=0.001)
parser.add_argument("--toy", type=eval, default=False, choices=[True, False])
parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church'], type=str, default="mnist")
parser.add_argument("--imagesize", type=int, default=32)
parser.add_argument(
    "--max_grad_norm", type=float, default=1e5,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
)
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--batch_size_sm', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--coeff', type=float, default=100)
parser.add_argument('--manifold_dim', type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--save', type=str, default='experiments/')
parser.add_argument('--viz_freq', type=int, default=500)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--pretrain1', type=eval, default=False, choices=[True, False])
parser.add_argument('--pretrain2', type=eval, default=False, choices=[True, False])
parser.add_argument('--resume1', type=str, default='experiments/epoch-1-1.pth', help='the model data for manifold training resume')
parser.add_argument('--resume2', type=str, default='experiments/epoch-1-2.pth', help='the model data for score matching to resume')
parser.add_argument(
    '--skip_manifold_training', type=eval, default=True, choices=[True, False]
)
parser.add_argument('--skip_score_matching', type=eval, default = True, choices = [True, False])
parser.add_argument('--pre_cal', type=eval, default=False, choices=[True, False])
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--autopretrain', type=eval, default=True, choices=[True, False])
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

def get_train_loader(train_set, shuffle=True, batch_size=None, ):
    current_batch_size = batch_size if batch_size != None else args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=shuffle, drop_last=True, pin_memory=True
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

def hessian(x, mtinv, mt, md):
    '''
    This is a helper function 
    '''
    ret = torch.zeros(1,md)
    # breakpoint()
    for i in range(0,md):
        vjp = torch.autograd.grad(mt[:,i], x, mtinv[i,:], create_graph=True)[0]
        ret = ret+vjp.detach().cpu()
    
    return ret
#endhessian

def compute_loss(score, latent_x, divergence):
    firstitr = True
    batch_size = score.shape[0]
    md = score.shape[1]
    for i in range(0, batch_size):
        jac = jacobian(score[i,:], latent_x, create_graph=True).squeeze()[0:md, i, 0:md].squeeze()
        if firstitr:
            loss = torch.trace(jac)
            firstitr = False
        else:
            loss = loss+torch.trace(jac)
    # endfor
    if divergence is not None:
        loss = loss/divergence.shape[0]
        loss = loss + 0.5*torch.mean(divergence@score)
    #endif
    loss = loss + torch.mean(torch.norm(score))
    return loss
#end

if __name__ == '__main__':
    #load data 
    train_set, test_loader, data_shape = get_dataset(args)

    #build model
    model = ManifoldLearning(data_shape[0], data_shape[1], args.manifold_dim)
    model = model.to(device)

    # logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    
    if args.pretrain1:
        checkpt = torch.load(os.path.join(args.save,args.resume1))
        model.load_state_dict(checkpt['model_state_dict'])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2], gamma=0.2)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')

    model.train()
    if args.skip_manifold_training is False:
        for epoch in range(1, args.num_epochs + 1):
            with torch.autograd.set_detect_anomaly(False):
                train_loader = get_train_loader(train_set)
                model.train()
                for itr, (x,y) in enumerate(train_loader):

                    # cast data and move to device
                    x = cvt(x)

                    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    loss, loss_embedding, reconst = model(x)
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    time_meter.update(time.time() - end)
                    log_message = (
                            'Iter {} | Time {:.4f}({:.4f}) | loss {}'.format(
                                str(itr), time_meter.val, time_meter.avg, loss.item()
                            )
                        )
                    
                    # logger.info(log_message)
                    end = time.time()
                    # breakpoint()
                    if (itr) % args.viz_freq == 0:
                        fig_filename = os.path.join(args.save, "figs", "{}-{:04d}.jpg".format(epoch, itr))
                        orgfig_filename = os.path.join(args.save, "figs", "{}-{:04d}-1.jpg".format(epoch, itr))
                        utils.makedirs(os.path.dirname(fig_filename))
                        save_image(reconst, fig_filename, nrow=10)
                        save_image(x,orgfig_filename, nrow = 10 )
                        # torch.save({
                        #     'model_state_dict': model.state_dict(),
                        #     'optimizer_state_dict': optimizer.state_dict(),
                        #     'loss': loss,
                        # }, os.path.join(args.save, 'epoch-{}-itr-{}-1.pth'.format(epoch, itr))) 
                        # lr_scheduler.step()


                # compute test loss
                model.eval()
                if epoch % args.val_freq == 0:
                    start = time.time()
                    logger.info("validating...")
                    total_loss = 0
                    nitem = 0
                    for itr, (x, y) in enumerate(test_loader):
                        x.requires_grad = True
                        x = cvt(x)
                        loss = model(x)[0]
                        total_loss = total_loss+loss.item()
                        nitem = nitem + 1

                    mloss = total_loss/nitem
                    logger.info("Epoch {:04d} | Time {:.4f}, loss {:.4f}".format(epoch, time.time() - start, mloss))
                    if mloss < best_loss:
                        best_loss = loss
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                        }, os.path.join(args.save, 'epoch-{}-itr-1-1.pth'.format(epoch))) 
                    #endif
                #endif
            #end
        #endfor
    #endif
    
    # Phase 2 use the trained manifold mapping to calculate the jacobian and hessian
    model.train()
    data_exist = True
    if os.path.exists('divergence_data.pt'):
        divergence = torch.load('divergence_data.pt')
        data_precaled = divergence.shape[0]
        print('{} data points already calculated'.format(data_precaled))
    else:
        data_exist = False
        data_precaled = 0
    tmp_train_set = [train_set[i] for i in range(data_precaled,len(train_set))]
    pre_cal_sampler = torch.utils.data.SequentialSampler(tmp_train_set)
    pre_cal_sampler = torch.utils.data.BatchSampler(pre_cal_sampler, 50, False)
    if args.pre_cal:
        for itr, idxs in enumerate(pre_cal_sampler):
            # cast data and move to device
            x = cvt(torch.stack([train_set[i][0] for i in idxs]))
            x = model.encoder(x)[:, 0:model.md]
            x.requires_grad_(True)
            for i in range(0,len(idxs)):
                print("processing {}-th sample".format(data_precaled))
                xnow = x[i:(i+1),:].requires_grad_()
                jac = jacobian(model.decoder(xnow).view(-1), xnow, create_graph=True)
                jac = jac.squeeze()
                metric_tensor = torch.matmul(jac.t(),jac)
                mtinv = torch.inverse(metric_tensor.detach())
                tmp = hessian(xnow, mtinv, metric_tensor, model.md).unsqueeze(0)
                if data_exist:
                    divergence = torch.cat([divergence, tmp])
                else:
                    divergence = tmp
                print(metric_tensor)
                print(tmp)
                data_precaled = data_precaled+1
                torch.save(divergence, 'divergence_data.pt')
            #endfor
        #endfor

    divergence = torch.load('divergence_data.pt')
    data_precaled = divergence.shape[0]
    divergence = cvt(divergence)
    # breakpoint()

    # Phase 3 score matching.
    if args.pre_cal is not True:
        data_precaled = 60000

    model2 = MLP([model.md,1], [model.md,1], [128,128,128]).to(device)
    optimizer = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    
    if args.pretrain2:
        checkpt = torch.load(os.path.join(args.save,args.resume2))
        model2.load_state_dict(checkpt['model_state_dict'])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2], gamma=0.2)
    subsest_sampler = torch.utils.data.SubsetRandomSampler([i for i in range(0, data_precaled)])
    train_loader = torch.utils.data.BatchSampler(subsest_sampler, args.batch_size_sm, False)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')

    model2.train()
    if args.skip_score_matching is not True:
        for epoch in range(1, args.num_epochs + 1):
            with torch.autograd.set_detect_anomaly(False):
                for itr, idxs in enumerate(train_loader):

                    # cast data and move to device
                    x = torch.stack([train_set[i][0] for i in idxs])
                    x = cvt(x)

                    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    latent_x = model.encoder(x)[:,0:model.md].unsqueeze(2)
                    score = model2(latent_x)

                    if args.pre_cal is True:
                        loss = compute_loss(score, latent_x, divergence[idxs, :])
                    else:
                        loss = compute_loss(score, latent_x, None)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    time_meter.update(time.time() - end)
                    loss_meter.update(loss.item())
                    
                    end = time.time()
                    if itr % args.viz_freq == 0 and loss_meter.avg<best_loss:
                        best_loss = loss_meter.avg
                        torch.save({
                            'model_state_dict': model2.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                        }, os.path.join(args.save, 'epoch-{}-itr-{}-2.pth'.format(epoch, itr))) 
                        lr_scheduler.step()
                    #endif

                    log_message = (
                            'Epoch {} Itr {}| Time {:.4f}({:.4f}) | loss {}'.format(
                                str(epoch), str(itr), time_meter.val, time_meter.avg, loss_meter.avg
                            )
                        )                    
                    logger.info(log_message)
                #endfor                
            #end
        #endfor

    #image generation
    model2.eval()
    model.eval()
    batch_size = 100
    cur = cvt(torch.randn(batch_size, model.md,1))
    alp = args.alp
    maxitr = 1000
    with torch.no_grad():
        for i in range(1,maxitr):
            noise = torch.randn(batch_size, model.md,1).to(cur)
            newimage = cur+model2(cur)*alp + math.sqrt(alp)*noise
            newimage = model.decoder(newimage)
            # breakpoint()
            newimage = model.encoder(newimage)[:,0:model.md]
            cur = newimage.unsqueeze(2)
            if i%4==0:
                filename = os.path.join(args.save, 'gif_{}'.format(alp),'generate-itr-{}.jpg'.format(i))
                utils.makedirs(os.path.dirname(filename))
                save_image(model.decoder(cur), filename, nrow = 10)
    #endfor
