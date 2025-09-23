import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCN, Encoder
from layers.prompt import *

from torch_geometric.nn import GCNConv
from torch.nn.utils import spectral_norm

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

        for i in range(num_layers_num):
            in_dim = n_h if i > 0 else n_in
            self.convs.append(Gcn_PyG(in_dim, n_h))
            self.bns.append(nn.BatchNorm1d(n_h))

    def forward(self, x, edge_index, sparse=True, prompt_layers=None, LP=False, ):
        # x: [N, d], edge_index: [2, num_edges]
        # xs = []
        for i in range(self.num_layers_num):
            res = x  # for residual
            x = self.convs[i](x, edge_index)
            
            if i > 0:
                x = x + res  # residual connection
            
            if prompt_layers:
                x = prompt_layers[i](x)

            # if LP:
            x = self.bns[i](x)
            x = self.dropout(x)

            # xs.append(x)

        return x.unsqueeze(0)  # [1, N, d]
    
class GNAELayers(torch.nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout, scaling_factor=1.8):
        super(GNAELayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num, scaling_factor)
        
        self.dropout = torch.nn.Dropout(p = dropout)

    def create_net(self,input_dim, hidden_dim, num_layers, scaling_factor):
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
    
        for i in range(num_layers):

            if i:
                nn = Encoder(hidden_dim, hidden_dim, scaling_factor=scaling_factor)
            else:
                nn = Encoder(input_dim, hidden_dim, scaling_factor=scaling_factor)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)
        
        return self.convs, self.bns


    def forward(self, seq, adj, sparse, LP = False, prompt_layers = None):


        graph_output = torch.squeeze(seq, dim=0)
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        xs = []
        if prompt_layers:
            assert(len(prompt_layers) == self.num_layers_num)

        for i in range(self.num_layers_num):
            # print("i",i)
            if i:
                graph_output = self.convs[i](graph_output, adj) + graph_output
            else:
                graph_output = self.convs[i](graph_output, adj)

            if prompt_layers:
                graph_output = prompt_layers[i](graph_output)
            # print("graphout1",graph_output)
            # print("graphout1",graph_output.shape)
            if LP:
                # print("graphout1",graph_output.shape)
                graph_output = self.bns[i](graph_output)
                # print("graphout2",graph_output.shape)
                graph_output = self.dropout(graph_output)
            # print("graphout2",graph_output)
            # print("graphout2",graph_output.shape)
            xs.append(graph_output)

        return graph_output.unsqueeze(dim=0)
    
class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num)
        
        self.dropout = torch.nn.Dropout(p = dropout)

    def create_net(self,input_dim, hidden_dim, num_layers):
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
    
        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)
        
        return self.convs, self.bns


    def forward(self, seq, adj, sparse, LP = False, prompt_layers = None):


        graph_output = torch.squeeze(seq, dim=0)
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        xs = []
        if prompt_layers:
            assert(len(prompt_layers) == self.num_layers_num)

        for i in range(self.num_layers_num):
            # print("i",i)
            input=(graph_output,adj)
            if i:
                graph_output = self.convs[i](input) + graph_output
            else:
                graph_output = self.convs[i](input)

            if prompt_layers:
                # print('structure prompt ! ')
                graph_output = prompt_layers[i](graph_output)
            # print("graphout1",graph_output)
            # print("graphout1",graph_output.shape)
            # if LP:
                # print('LP !')
                # print("graphout1",graph_output.shape)
                # graph_output = self.bns[i](graph_output)
                # print("graphout2",graph_output.shape)
                # graph_output = self.dropout(graph_output)
            graph_output = self.bns[i](graph_output)
                # print("graphout2",graph_output.shape)
            graph_output = self.dropout(graph_output)
            # print("graphout2",graph_output)
            # print("graphout2",graph_output.shape)
            xs.append(graph_output)

        return graph_output.unsqueeze(dim=0)
    
# def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
#     bsz = graph_sizes.size(0)
#     dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

#     min_size, max_size = graph_sizes.min(), graph_sizes.max()
#     mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

#     if min_size == max_size:
#         return batched_graph_feats.view(bsz, max_size, -1), mask
#     else:
#         graph_sizes_list = graph_sizes.view(-1).tolist()
#         unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
#         for i, l in enumerate(graph_sizes_list):
#             if l == max_size:
#                 continue
#             elif l > max_size:
#                 unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
#             else:
#                 mask[i, l:].fill_(0)
#                 zeros = torch.zeros((max_size-l, dim), dtype=dtype, device=device, requires_grad=False)
#                 unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
#         return torch.stack(unbatched_graph_feats, dim=0), mask
