import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP
# GNAE 
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index=None, model='GNAE', scaling_factor=1.8):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)
        self.model = model
        self.scaling_factor = scaling_factor

    def forward(self, x, adj, not_prop=0):
        if self.model == 'GNAE':
            x = self.linear1(x)
            x = F.normalize(x,p=2,dim=1)  * self.scaling_factor
            # x = self.propagate(x, edge_index)
            x = torch.matmul(adj, x) # edge index -> adj 형태로 수정 
            return x

        if self.model == 'VGNAE':
            x_ = self.linear1(x)
            # x_ = self.propagate(x_, edge_index)
            x_ = torch.matmul(adj, x_) # edge index -> adj 형태로 수정 

            x = self.linear2(x)
            x = F.normalize(x,p=2,dim=1) * self.scaling_factor # 정규화 텀 추가 됨. 
            # x = self.propagate(x, edge_index)
            x = torch.matmul(adj, x) # edge index -> adj 형태로 수정 

            return x, x_

        return x
    
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=None, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        # print("act",type(self.act))
        # print("fc",self.fc.weight)
        # print("fc",self.fc.weight.shape)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, input, sparse=True):
        # print("input",input)
        seq = input[0]#.to(input.device)
        adj = input[1]#.to(input.device)
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        seq_fts = self.fc(seq) # 1. 노드 피처 선형 변환
        if sparse:
            out = torch.spmm(adj, seq_fts) # 2. aggregation 
        else:
            # print("adj",adj.shape)
            # print("seqft",seq_fts.shape)
            out = torch.mm(adj.squeeze(dim=0), seq_fts)
        if self.bias is not None:
            out += self.bias

        # print("out",out)
        # print("act",self.act)

        return self.act(out)
        # return out.type(torch.float)
