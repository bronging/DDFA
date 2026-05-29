import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.sample = x
        self.out = F.normalize(self.encode(x.T)) #(feature, sig_embed)
        return self.out

    def dimensional_loss(self):
        return self.out.mean(dim=0).pow(2).mean()
        
    def feature_sig_propagate(self, x, dimension_sig):
        return F.normalize(x @ dimension_sig)
        
    def reduced_feature(self, x):
        with torch.no_grad():
            self.eval()
            dimension_sig = self.dnn(self.d_sample_matrix)
            x = self.feature_sig_propagate(x, dimension_sig)
            return x
    def mean_basis_vector(self): 
        return self.out.mean(dim=0) 
        
        
    