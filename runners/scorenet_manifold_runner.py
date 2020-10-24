import numpy as np
from losses.sliced_sm import sliced_score_estimation_vr
from losses.dsm import dsm_score_estimation
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models.scorenet import ManifoldMLPScore
from models.generator import Generator
from models.resnet import ResNet, ResNetBasicBlock

__all__ = ['ScoreNetMRunner']


class ScoreNetMRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999))
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.8)], indices[int(num_items * 0.8):]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.7)], indices[
                                                                          int(num_items * 0.7):int(num_items * 0.8)]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4)

        test_iter = iter(test_loader)
        # self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        encoder = ResNet(in_channels=self.config.data.channels, n_classes=self.config.model.manifold_dim, block=ResNetBasicBlock, deepths=[2, 2, 2]).to(self.config.device)
        score = ManifoldMLPScore(self.config).to(self.config.device)
        decoder = Generator(md = self.config.model.manifold_dim, nc = self.config.data.channels).to(self.config.device)

        params = list(encoder.parameters()) + list(score.parameters()) + list(decoder.parameters())
        optimizer = self.get_optimizer(params)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1

                X = X.to(self.config.device)
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                scaled_score = lambda x: score(x)

                if self.config.training.algo == 'ssm':
                    # X = X + torch.randn_like(X) * sigma
                    latent_X = encoder(X)
                    # breakpoint()
                    rec_X = decoder(latent_X)
                    loss_sm, *_ = sliced_score_estimation_vr(scaled_score, latent_X.detach(), n_particles=1)
                    loss_rec = ((X-rec_X)**2).mean()  + 0.5*(latent_X**2).mean()
                    loss = loss_sm + self.config.training.lbd * loss_rec


                elif self.config.training.algo == 'dsm':
                    latent_X = encoder(X)
                    rec_X = Generator(latent_X)
                    loss_sm = dsm_score_estimation(scaled_score, X, sigma=self.config.training.noise_std)
                    loss_rec = ((X-rec_X)**2).mean() + 0.5*(latent_X**2).mean()
                    loss = loss_sm + self.config.training.lbd * loss_rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                # tb_logger.add_scalar('sigma', sigma, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    if self.config.training.algo == 'ssm':
                        test_latent_X = encoder(test_X)
                        test_rec_X = decoder(test_latent_X)
                        loss_sm, *_ = sliced_score_estimation_vr(scaled_score, test_latent_X.detach(), n_particles=1)
                        loss_rec = ((X-rec_X)**2).mean() + 0.5*(latent_X**2).mean()
                        test_loss = loss_sm + self.config.training.lbd * loss_rec
                    elif self.config.training.algo == 'dsm':
                        print('Havent implemented yet!')
                        test_loss = dsm_score_estimation(scaled_score, test_X, sigma=self.config.training.noise_std)

                    tb_logger.add_scalar('test_loss', test_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict()
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

                if step == self.config.training.n_iters:
                    return 0
