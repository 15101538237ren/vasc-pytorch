# -*- coding: utf-8 -*-
import torch
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