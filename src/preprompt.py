import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Sampler
from layers.prompt import *
from models import DimensionNN_FUG, GcnLayers_PyG, Lp


class PrePromptBaryBasis(nn.Module):
    def __init__(self, unify_dim, hid_units, num_pretrain_dataset, num_layers, dropout, type_, 
                alpha=3.0, beta=100, n_sample=256, if_rand=False, num_de_layers=2, sampling='random'):
        super(PrePromptBaryBasis, self).__init__()

        self.gcn = GcnLayers_PyG(unify_dim, hid_units, num_layers, dropout)
        self.lp = Lp(unify_dim, hid_units)

        self.samplers = [Sampler(sample_size=n_sample, if_rand=if_rand, sampling=sampling) for _ in range(num_pretrain_dataset)]

        self.domain_token_layers = nn.ModuleList([textprompt(unify_dim, type_=type_) for _ in range(num_pretrain_dataset)])
        self.dimension_encoder_layers = nn.ModuleList([DimensionNN_FUG(n_sample, unify_dim*2, unify_dim, nn.PReLU, layers=num_de_layers) 
                                                        for _ in range(num_pretrain_dataset)])

        self.alpha = alpha 
        self.beta = beta 

    def get_reduction(self, seq_list, adj_list, aggregated_feat):
        xt_list = []
        for dim_pretext, domain_token, seq, adj, sampler, agg_feat, in \
            zip(self.dimension_encoder_layers, self.domain_token_layers, seq_list, adj_list, self.samplers, aggregated_feat):
           
            sample = sampler(agg_feat, adj)               
            basis_trans_mat = dim_pretext(sample)
            xt = dim_pretext.feature_sig_propagate(seq, basis_trans_mat)
            
            xt = sampler(xt, adj)
            xt = domain_token(xt)
            
            xt_list.append(xt)

        return xt_list  
    
    def compute_prelogits_LP(self, seq_list, adj_list, sparse=False, aggregated_feat=None):
        for dim_pretext, domain_token, seq, adj, sampler, agg_feat, in \
            zip(self.dimension_encoder_layers, self.domain_token_layers, seq_list, adj_list, self.samplers, aggregated_feat):
            
            # 1. Intra-Domain Dimension Alignment 
            sample = sampler(agg_feat, adj)   
            basis_trans_mat = dim_pretext(sample)
            preseq = dim_pretext.feature_sig_propagate(seq, basis_trans_mat)

            # 2. Inter-Domain Semantic Alignment 
            preseq = domain_token(preseq)
            
            yield self.lp(gcn=self.gcn, seq=preseq, adj=adj, sparse=sparse) 

    def get_weights(self):
        domain_tokens = [layer.weight.detach() for layer in self.domain_token_layers]
        return domain_tokens
    
    def ssl_loss_fn_infoNCE(self, z):
        z = F.normalize(z, dim=1)
        return z.mean(dim=0).pow(2).mean()

    def forward(self, seq_list, adj_list, sparse, aggregated_feat=None, samples=None):
        lp_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        logits = list(self.compute_prelogits_LP(
            seq_list, 
            adj_list, 
            sparse, 
            aggregated_feat))

        if isinstance(samples, list):
            samples = [torch.tensor(sample, dtype=torch.int64).to(seq_list[0].device)
                for sample in samples] 
            for i in range(len(logits)):    
                loss = compareloss(logits[i], samples[i], temperature=1)
                lp_loss += loss
        else:
            samples = torch.tensor(samples, dtype=torch.int64).to(seq_list[0].device)
            logits = torch.cat(logits, dim=0)
            lp_loss = compareloss(logits, samples, temperature=1)

        

        intra_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        domain_basis = []

        for idx, dim_pretext in enumerate(self.dimension_encoder_layers): 
            intra_loss += dim_pretext.dimensional_loss()
            domain_basis.append(self.domain_token_layers[idx].weight.squeeze(0))
        
        domain_basis = torch.stack(domain_basis, dim=0)  # [m, d]
        inter_loss = domain_basis.mean(dim=0).pow(2).mean()
        
        lp_diversity_loss = lp_loss + self.beta * (intra_loss + inter_loss) 

        return lp_diversity_loss


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))

def compareloss(feature,tuples,temperature): #  InfoNCE
    h_tuples=mygather(feature,tuples) # 각 anchor에 대응하는 sample embedding 추출 (각 sample(index)에 대응하는 feature 가져옴)
    temp = torch.arange(0, len(tuples)).to(feature.device)
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp) # anchor embedding 

    sim = F.cosine_similarity(h_i, h_tuples, dim=2) # anchor - tuple 간 유사도 계산 
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)  # positive score (tuple 중 첫번째가 pos)
    denominator = exp[1:exp.size(0)]  # 나머지는 negative samples - negative score
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True) # 모든 negative score 합 

    res = -1 * torch.log(numerator / denominator) # - log (exp(pos) / exp(neg))
    return res.mean()
