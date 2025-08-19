import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(Predictor, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        h = self.linears[self.num_layers - 1](h)
        return h
    
class Sampler:
    def __init__(self, sample_size, if_rand, sampling='random'):
        self.sample_size = sample_size
        self.if_rand = if_rand
        self.fixed_indices = None
        self.sampling = sampling 
        self.sample_feature = None 

    def __call__(self, x, edge_index=None):
        if self.if_rand:
            # 매번 새로운 랜덤 샘플링
            idx = torch.randperm(x.shape[0])[:self.sample_size]
        else:
            # 고정된 인덱스를 처음에 한 번만 생성
            if self.fixed_indices is None:
                if self.sampling == 'random': 
                    self.fixed_indices = torch.randperm(x.shape[0])[:self.sample_size]
                elif self.sampling == 'feat_norm': 
                    norms = torch.norm(x, p=2, dim=1)  # ℓ2-norm
                    self.fixed_indices = torch.topk(norms, k=self.sample_size).indices
                    
                    # print(f'feat norm topk: ')
                    # print(self.fixed_indices)
                elif self.sampling == 'degree': 
                    deg = torch.bincount(edge_index[0], minlength=x.shape[0])
                    # self.fixed_indices = torch.topk(deg, k=self.sample_size).indices
                    self.fixed_indices = torch.argsort(deg)[:self.sample_size]  # deg 작은 순서로 선택 
                elif self.sampling == 'front': 
                    self.fixed_indices = torch.arange(self.sample_size)
                elif self.sampling == 'random_readout': 
                    samples = []
                    k=5
                    for _ in range(self.sample_size):
                        idx = torch.randperm(x.shape[0])[:k]
                        pseudo_feat = x[idx].mean(dim=0)  # or weighted_mean
                        samples.append(pseudo_feat)
                    self.fixed_indices = torch.stack(samples, dim=0)

                # print('New sampling!')
            idx = self.fixed_indices

        if self.sampling == 'random_readout': 
            return self.fixed_indices
            
        return x[idx, :]

       
    def reset_indices(self):
        self.fixed_indices = None 

