import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FeatureMLP(torch.nn.Module):
    def __init__(self, in_dim=50, hidden_dim=50, out_dim=50, num_layer=1, init_identity=False, mlp_bias=True):
        super().__init__()
        out_dim = out_dim or in_dim
        
        layers = [torch.nn.Linear(in_dim, hidden_dim, bias=mlp_bias)]
        for i in range(1, num_layer): 
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(hidden_dim, out_dim, bias=mlp_bias))
        
        self.net = torch.nn.Sequential(*layers)
        self.mlp_bias = mlp_bias

        if init_identity:
            self._init_as_identity()

    def forward(self, x):
        return self.net(x)
    
    def _init_as_identity(self):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                in_dim, out_dim = layer.weight.shape
                if in_dim == out_dim:
                    torch.nn.init.eye_(layer.weight)
                else:
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                if self.mlp_bias: 
                    torch.nn.init.constant_(layer.bias, 0.0)
    
