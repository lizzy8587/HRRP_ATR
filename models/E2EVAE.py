#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File         :   VAE.py
@Version      :   1.0
@Time         :   2023/02/16 10:54:47
@E-mail       :   daodao123@sjtu.edu.cn
@Introduction :   end-to-end VAE for hrrp, generate hrrp data, classify targets
'''


import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple


class E2EVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 cls_num: int = 3,
                 length: int = 256,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.cls_num = cls_num
        self.encoded_shape = None
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.flatten_dims = int(length*hidden_dims[0]/2) # hidden_dims[-1]*8
        self.fc_mu = nn.Linear(self.flatten_dims, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dims, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dims)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size= 3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        self.encoded_shape = result.shape #  view(-1,512,8)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # result = result.view(-1, 512, 8)
        result = result.view(self.encoded_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def classifier(self, input: torch.Tensor) -> torch.Tensor:
        '''
        description: 对隐变量z进行分类
        param: input指重参数化返回的隐变量z， torch.Size([bs,latent_dim])
        return: torch.Size([bs,cls_num])
        '''
        fc = nn.Sequential(
            nn.Linear(self.latent_dim,self.cls_num),
            nn.Softmax(dim=1)
        )
        return fc(input)

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)    # torch.Size([bs,latent_dim])
        c = self.classifier(z)                  # torch.Size([bs,cls_num])
        return  [self.decode(z), input, mu, log_var, c]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        predict = args[4]

        labels = kwargs['labels']
        kld_weight = kwargs['kld_weight'] if 'kld_weight' in kwargs.keys() else 1
        rec_weight = kwargs['rec_weight'] if 'rec_weight' in kwargs.keys() else 1
        cls_weight = kwargs['cls_weight'] if 'cls_weight' in kwargs.keys() else 1
        recons_loss =F.mse_loss(recons, input)
        cls_loss = F.cross_entropy(predict,labels)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = rec_weight * recons_loss + kld_weight * kld_loss + cls_weight * cls_loss
        return {'loss': loss, 'rec_loss':rec_weight * recons_loss.detach(), 'kld':-kld_weight * kld_loss.detach(), 'cls_loss':cls_weight * cls_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]