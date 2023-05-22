# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE

class GAE_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, model_type):
        super(GAE_Encoder, self).__init__()
        self.model_type = model_type
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if model_type == 'GAE':
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif model_type == 'VGAE':
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.model_type == 'GAE':
            return self.conv2(x, edge_index)
        elif self.model_type == 'VGAE':
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class VGAEsc(VGAE):
    def __init__(self, encoder, in_dim, zdim):
        super(VGAEsc, self).__init__(encoder)
        # decoder part
        self.fc4 = nn.Linear(zdim, 32)
        self.fc5 = nn.Linear(32, 128)
        self.fc6 = nn.Linear(128, 512)
        self.dc = nn.Linear(512, in_dim)


    def sampling_gumbel(self, sizes, eps=1e-8):
        u = torch.rand(sizes).cuda()
        epsilon = torch.ones(u.size()).mul(eps).cuda()
        sample = -torch.log( -torch.log(u + epsilon) + epsilon)
        return sample

    def compute_softmax(self, logits, tau):
        z = logits + self.sampling_gumbel(logits.size())
        return F.softmax(torch.div(z, torch.ones(z.size()).mul(tau).cuda()), dim=-1)

    def get_decode(self, z, tau=1.0):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        expr_x = F.sigmoid(self.dc(h6))
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
        return out

    def expr_recon_loss(self, z, x):
        recon_x = self.get_decode(z)
        n, in_dim = x.shape
        loss = nn.BCELoss(reduction="mean")
        BCE = 50 * loss(recon_x, x)
        KLD = torch.mean(-0.5 * torch.sum(1 + self.__logstd__ - self.__mu__.pow(2) - self.__logstd__.exp(), dim=-1))
        VAE_Loss = BCE + KLD
        return VAE_Loss
