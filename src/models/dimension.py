import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

class DimensionNN_V2(nn.Module):
    def __init__(self, n_in, n_h, n_out, activator, layers=1):
        # n_in = sample_size
        # n_out = k, reduced dimension 
        super(DimensionNN_V2, self).__init__()
        self.act = activator()
        self.layers = layers
        if layers == 1: 
            self.lin_in = nn.Linear(n_in, n_out) # 1 layer 만 사용  
        elif layers == 2: 
            self.lin_in = nn.Linear(n_in, n_h)
            self.lin_h1 = nn.Linear(n_h, n_h)
            self.lin_out = nn.Linear(n_h, n_out)
        self.sample = []

    def encode(self, x):
        if self.layers == 1: 
            return self.lin_in(x) 
        elif self.layers == 2: 
            z = self.act(self.lin_in(x))
            z = self.act(self.lin_h1(z))
            return self.lin_out(z)

    def forward(self, x):
        '''
        x size: (sample_size, feature) 
        '''
        self.sample = x
        self.out = F.normalize(self.encode(x.T)) #(feature, sig_embed)
        return self.out
         

    def dimensional_loss(self):
        return self.out.mean(dim=0).pow(2).mean()

class GCN_encoder(nn.Module):
    def __init__(self, n_in, n_h, activator):
        super(GCN_encoder, self).__init__()
        self.gcn_in = GCNConv(n_in, n_h)
        self.gcn_out = GCNConv(n_h, n_h)
        self.act = activator()

    def encode(self, x, edge_index):
        out = self.act(self.gcn_in(x, edge_index))
        out = self.gcn_out(out, edge_index)
        return out

    def proj(self, z):
        return self.lin_2(self.act(self.lin_1(z)))
        
    def forward(self, x, edge_index):
        out = self.encode(x,edge_index)
        return out

    def embed(self, x, edge_index):
        self.eval()
        return self.encode(x, edge_index)


class FUG(nn.Module):
    def __init__(self, D_NN, G_NN, S_mtd, sample_size):
        super(FUG, self).__init__()
        self.dnn = D_NN
        self.gnn = G_NN
        self.smtd = S_mtd
        self.sample_size = sample_size
        self.d_sample_matrix = []

    def update_sample(self, x, edge_index, if_rand=False):
        with torch.no_grad():
            self.d_sample_matrix = self.smtd(self.sample_size, x, edge_index, if_rand)

    def forward(self, x, edge_index):
        dimension_sig = self.dnn(self.d_sample_matrix)
        x = self.feature_sig_propagate(x, dimension_sig)
        return self.gnn(x, edge_index)

    def embed(self, x, edge_index):
        with torch.no_grad():
            self.eval()
            dimension_sig = self.dnn(self.d_sample_matrix)
            x = self.feature_sig_propagate(x, dimension_sig)
            return self.gnn.embed(x, edge_index)

    def reduced_feature(self, x):
        with torch.no_grad():
            self.eval()
            dimension_sig = self.dnn(self.d_sample_matrix)
            x = self.feature_sig_propagate(x, dimension_sig)
            return x
        
    def ssl_loss_fn_infoNCE(self, z):
        z = F.normalize(z, dim=1)
        return z.mean(dim=0).pow(2).mean()

    def ssl_loss_fn_pos(self, z, edge_index):
        return (z[edge_index[0]]-z[edge_index[1]]).pow(2).mean()

    def dim_loss_fn(self):
        return self.dnn.dimensional_loss()

    def feature_sig_propagate(self, x, dimension_sig):
        return F.normalize(x @ dimension_sig)

import torch.nn.init as init
class DimensionNN_FUG(nn.Module):
    def __init__(self, n_in, n_h, n_out, activator, layers=1):
        # n_in = sample_size
        # n_out = k, reduced dimension 
        super(DimensionNN_FUG, self).__init__()
        self.act = activator()
        self.layers = layers
        
        self.n_in = n_in
        self.n_h = n_h
        self.n_out = n_out 

        if layers == 1: 
            self.lin_in = nn.Linear(n_in, n_out) # 1 layer 만 사용  
        elif layers == 2: 
            self.lin_in = nn.Linear(n_in, n_h)
            self.lin_out = nn.Linear(n_h, n_out)
        elif layers == 3: 
            self.lin_in = nn.Linear(n_in, n_h)
            self.lin_h1 = nn.Linear(n_h, n_h)
            self.lin_out = nn.Linear(n_h, n_out)
        self.sample = []

        self.sample_size = n_in
        self.d_sample_matrix = []
        
        self.reset_parameters()  # 초기화 수행

    def reset_parameters(self):
        if self.layers == 1:
            init.xavier_uniform_(self.lin_in.weight)
            if self.lin_in.bias is not None:
                init.zeros_(self.lin_in.bias)
        elif self.layers == 2:
            init.xavier_uniform_(self.lin_in.weight)
            init.xavier_uniform_(self.lin_out.weight)
            if self.lin_in.bias is not None:
                init.zeros_(self.lin_in.bias)
                init.zeros_(self.lin_out.bias)
        elif self.layers == 3:
            init.xavier_uniform_(self.lin_in.weight)
            init.xavier_uniform_(self.lin_h1.weight)
            init.xavier_uniform_(self.lin_out.weight)
            if self.lin_in.bias is not None:
                init.zeros_(self.lin_in.bias)
                init.zeros_(self.lin_h1.bias)
                init.zeros_(self.lin_out.bias)
    def update_sample(self, sample):
        self.d_sample_matrix = sample

    def encode(self, x):
        # ✅ 입력 정규화 (especially for binary vs float feature scale difference)
        # x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        # x = F.normalize(x, p=2, dim=1)

        if self.layers == 1: 
            return self.lin_in(x) 
        elif self.layers == 2: 
            z = self.act(self.lin_in(x))
            return self.lin_out(z)
        elif self.layers == 3:
            z = self.act(self.lin_in(x))
            z = self.act(self.lin_h1(z))
            return self.lin_out(z)

    def forward(self, x):
        '''
        x size: (sample_size, feature) 
        '''
        self.sample = x

        # ✅ 입력 정규화 (especially for binary vs float feature scale difference)
        # x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        # x = F.normalize(x, p=2, dim=1)

        self.out = F.normalize(self.encode(x.T)) #(feature, sig_embed)
        return self.out

    def dimensional_loss(self):
        # print(self.out.shape)
        # print(self.out.mean(dim=0).shape)
        # print(self.out.mean(dim=0).pow(2).shape)
        return self.out.mean(dim=0).pow(2).mean()

    def feature_sig_propagate(self, x, dimension_sig):
        # ✅ 입력 정규화 (especially for binary vs float feature scale difference)
        # x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        # x = F.normalize(x, p=2, dim=1)

        return F.normalize(x @ dimension_sig)
    
    def reduced_feature(self, x):
        with torch.no_grad():
            self.eval()
            dimension_sig = self.dnn(self.d_sample_matrix)
            x = self.feature_sig_propagate(x, dimension_sig)
            return x
        
    def basis_matrix(self): 
        return self.out

    def mean_basis_vector(self): 
        return self.out.mean(dim=0) 
        
        
    