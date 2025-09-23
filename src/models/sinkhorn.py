import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class WassBarycenter(nn.Module):
    def __init__(self, feat_dim):
        super(WassBarycenter, self).__init__()
        self.feat_dim = feat_dim 
        self.barycenter_weight = nn.Parameter

class Critic(nn.Module): 
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
