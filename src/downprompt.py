import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from layers.prompt import *
from models import DimensionNN_FUG


class downstreamprompt(nn.Module):
    def __init__(self, feature_dim, type_='mul', sample_size=256, 
                agg_feat=None, domain_tokens=None, num_de_layers=2):
        super(downstreamprompt, self).__init__()

        self.dimension_encoder = DimensionNN_FUG(sample_size, feature_dim*2, feature_dim, nn.PReLU, layers=num_de_layers)

        self.specific_prompt = textprompt(feature_dim, type_)
        self.mixture_prompt = composedtoken(domain_tokens, type_=type_)
        self.combineprompt = combineprompt()
        self.agg_feat = agg_feat

        
    def forward(self, seq, gcn, adj, sparse):
        sample = seq[1]
        seq = seq[0]

        # 1. Dual-Space Dimension Alignment 
        basis_trans_mat = self.dimension_encoder(sample)
        dim_seq = F.normalize((seq + self.agg_feat) @ basis_trans_mat)
        
        # 2. Semantic Alignment 
        dim_seq1 = self.specific_prompt(dim_seq)
        dim_seq2 = self.mixture_prompt(dim_seq)
        h = self.combineprompt(dim_seq1, dim_seq2)
        emb = gcn(h, adj, sparse, None)
        return emb
        

class downprompt(nn.Module):
    def __init__(self, hid_units, nb_classes, unify_dim, type_='add', sample_size=256, 
                temp=0.5, agg_feat=None, domain_tokens=None, num_de_layers=2):
        super(downprompt, self).__init__()        
        
        self.downstreamPrompt = downstreamprompt(unify_dim, type_, sample_size, agg_feat, domain_tokens, num_de_layers=num_de_layers)

        self.sample_size = sample_size
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, hid_units)
        self.ave = torch.FloatTensor(nb_classes, hid_units)
        self.agg_feat = agg_feat
        self.temp = temp 

    def forward(self, features, adj, sparse, gcn, idx, labels=None, train=0):
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)  # [nodes, emb_dim]
        uniform_loss = self.ssl_loss_fn_infoNCE(embeds)
        rawret = embeds[idx]  # rawret: [num_nodes, emb_dim]
        num = rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)  # class prototypes: [num_classes, emb_dim]
        rawret = torch.cat((rawret, self.ave), dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num, num:]  # Select only node-to-prototype similarities
        ret = F.softmax(ret / self.temp, dim=1) 
        return ret, uniform_loss

    def ssl_loss_fn_infoNCE(self, z):
        z = F.normalize(z, dim=1)
        return z.mean(dim=0).pow(2).mean()


def averageemb(labels, rawret):
    retlabel = torch_scatter.scatter(src=rawret, index=labels, dim=0, reduce='mean')
    return retlabel

    