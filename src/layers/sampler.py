import torch
import torch.nn as nn
import torch.nn.functional as F


class Sampler:
    def __init__(self, sample_size, if_rand, sampling='random'):
        self.sample_size = sample_size
        self.if_rand = if_rand
        self.fixed_indices = None
        self.sampling = sampling 

    def __call__(self, x, edge_index=None, include_idx=None):
        if self.if_rand:
            # Re-sample randomly at every call
            self.fixed_indices = torch.randperm(x.shape[0])[:self.sample_size]
        else:
            # Generate fixed indices only once at the first call
            if self.fixed_indices is None:
                if self.sampling == 'random': 
                    self.fixed_indices = torch.randperm(x.shape[0])[:self.sample_size]
                
                elif self.sampling == 'feat_norm': 
                    norms = torch.norm(x, p=2, dim=1)  # ℓ2-norm
                    self.fixed_indices = torch.topk(norms, k=self.sample_size).indices

                elif self.sampling == 'degree': 
                    deg = torch.bincount(edge_index[0], minlength=x.shape[0])
                    self.fixed_indices = torch.topk(deg, k=self.sample_size).indices

            idx = self.fixed_indices
            if include_idx is not None:
                include_idx = include_idx.to(x.device)
        
                if isinstance(self.fixed_indices, torch.Tensor) and self.fixed_indices.dim() == 1:
                    self.fixed_indices = self.fixed_indices.to(x.device)
                    self.fixed_indices = torch.unique(torch.cat([self.fixed_indices, include_idx]))
                    if self.fixed_indices.size(0) > self.sample_size:
                        self.fixed_indices = self.fixed_indices[:self.sample_size]

        return x[self.fixed_indices, :]

       
    def reset_indices(self):
        self.fixed_indices = None 

    def get_indices(self): 
        return self.fixed_indices
    
    def set_indices(self, indices): 
        self.fixed_indices = indices
