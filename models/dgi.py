# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DeepGraphInfomax
import numpy as np



class DGI_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(DGI_Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x