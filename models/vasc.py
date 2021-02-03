# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VASC(nn.Module):
    def __init__(self, x_dim, z_dim, var, dropout, isTrain):
        super(VASC, self).__init__()
        self.x_dim = x_dim
        self.var = var
        self.dropout = dropout
        self.isTrain = isTrain

        # encoder part
        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.z_mean = nn.Linear(32, z_dim)
        self.z_log_var = nn.Linear(32, z_dim)

        # decoder part
        self.fc4 = nn.Linear(z_dim, 32)
        self.fc5 = nn.Linear(32, 128)
        self.fc6 = nn.Linear(128, 512)
        self.dc = nn.Linear(512, x_dim)

    def encoder(self, x):
        if self.isTrain:
            x = nn.Dropout(p=self.dropout)(x)
        h1 = self.fc1(x)
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        z_mean = self.z_mean(h3)
        if self.var:
            z_log_var = F.softplus(self.z_log_var(h3))
        else:
            z_log_var = self.z_log_var(h3)
        return z_mean, z_log_var # mu, log_var

    def sampling(self, mu, log_var):
        eps = torch.randn_like(mu).cuda()
        if self.var:
            std = torch.exp(0.5 * log_var).cuda()
        else:
            empt = torch.ones(log_var.size()).cuda()
            std = torch.exp(torch.tensor(0.5 * empt)).cuda()
        return torch.add(eps.mul(std), mu)  # return z sample

    def sampling_gumbel(self, sizes, eps=1e-8):
        u = torch.rand(sizes).cuda()
        epsilon = torch.ones(u.size()).mul(eps).cuda()
        sample = -torch.log( -torch.log(u + epsilon) + epsilon)
        return sample

    def compute_softmax(self, logits, tau):
        z = logits + self.sampling_gumbel(logits.size())
        return F.softmax(torch.div(z, torch.ones(z.size()).mul(tau).cuda()), dim=-1)

    def decoder(self, z, tau):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        expr_x = F.sigmoid(self.dc(h6))
        if self.isTrain:
            p = torch.exp(-torch.pow(expr_x, 2)) # p: dropout rate
            q = torch.ones(p.size()).cuda() - p

            log_q = torch.log(q + torch.ones(q.size()).mul(1e-20).cuda())
            log_p = torch.log(p + torch.ones(p.size()).mul(1e-20).cuda())

            log_p = log_p.unsqueeze(0).permute(1, 2, 0)
            log_q = log_q.unsqueeze(0).permute(1, 2, 0)
            logits = torch.cat((log_p, log_q), dim=-1)

            samples = self.compute_softmax(logits, tau)
            samples = samples[:, :, 1]
            out = expr_x.mul(samples)
        else:
            out = expr_x
        return out

    def forward(self, x, tau):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z, tau), mu, log_var
