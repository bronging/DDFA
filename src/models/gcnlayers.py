import torch.nn as nn

from torch_geometric.nn import GCNConv


class Gcn_PyG(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super().__init__()
        self.conv = GCNConv(in_ft, out_ft, bias=bias)
        self.act = nn.PReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)  # aggregation + linear
        return self.act(x)

class GcnLayers_PyG(nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout):
        super().__init__()
        self.num_layers_num = num_layers_num
        self.convs = nn.ModuleList()

        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        # self.lns = nn.ModuleList()
        for i in range(num_layers_num):
            in_dim = n_h if i > 0 else n_in
            self.convs.append(Gcn_PyG(in_dim, n_h))
            self.bns.append(nn.BatchNorm1d(n_h))

    def forward(self, x, edge_index, sparse=True, prompt_layers=None, LP=False):
        xs = []
        for i in range(self.num_layers_num):
            res = x  # for residual
            
            x = self.convs[i](x, edge_index)
            
            if i > 0:
                x = x + res  # residual connection
            
            if prompt_layers:
                x = prompt_layers[i](x)

            x = self.bns[i](x) # batch norm 
            x = self.dropout(x)

            xs.append(x)
        return  x.unsqueeze(0)  # [1, N, d]
    
  