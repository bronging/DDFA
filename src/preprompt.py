import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers, MLP, GatLayers, FeatureMLP, GNAELayers, DimensionNN_V2, DimensionNN_FUG, GcnLayers_PyG
from layers import AvgReadout, NodeFeaturePermMLP,build_hard_permutation_from_logits, Sampler, EMA, Predictor
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from layers.prompt import *
import copy

import copy
import torch.nn.functional as F
# import dgl.function as fn

from torch_scatter import scatter_mean

def split_features_by_frequency(seq_tensor, adj, top_k_low_freq=0.5):
    """
    Input: seq_tensor (torch.Tensor) - 노드 피처 (num_nodes, num_features)
    Output:
        low_freq_feat (torch.Tensor) - 저주파 피처 (num_nodes, num_low_features)
        high_freq_feat (torch.Tensor) - 고주파 피처 (num_nodes, num_high_features)
    """
    # 1. 각 피처 차원(컬럼)의 분산을 계산
    variances = torch.var(seq_tensor, dim=0)

    # 1. 모든 엣지에 걸쳐 각 피처 차원의 변화량을 계산
    # adj[0]은 소스 노드 인덱스, adj[1]은 타겟 노드 인덱스
    # src_nodes = adj[0]
    # dst_nodes = adj[1]
    
    # 각 엣지에 대한 피처 값의 차이
    # feat_diff = seq_tensor[src_nodes] - seq_tensor[dst_nodes]
    
    # 2. 각 피처 차원의 총 변화량(total variation)을 계산
    # 모든 엣지에 걸친 절댓값 차이의 합
    # variances = torch.sum(torch.abs(feat_diff), dim=0)


    # 2. 분산이 낮은 순서대로 피처 차원 인덱스를 정렬
    sorted_indices = torch.argsort(variances)
    
    # 3. 상위 K% (저주파)와 나머지 (고주파)로 인덱스 분리
    num_features = seq_tensor.shape[1]
    num_low_freq = int(num_features * top_k_low_freq)

    low_freq_indices = sorted_indices[:num_low_freq]
    high_freq_indices = sorted_indices[num_low_freq:]

    return low_freq_indices, high_freq_indices

def normalize_adjacency(edge_index, num_nodes):
    
    # 1. build sparse adjacency matrix
    adj = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1), device=edge_index.device), 
        (num_nodes, num_nodes)
    )

    # 2. compute degree and D^{-1/2}
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # 3. compute normalized adjacency: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = deg_inv_sqrt.view(-1, 1)
    row, col = edge_index
    norm_vals = D_inv_sqrt[row] * D_inv_sqrt[col]
    
    norm_adj = torch.sparse_coo_tensor(
        edge_index, norm_vals.squeeze(), (num_nodes, num_nodes)
    )

    return norm_adj  


def get_low_pass_filter(X, edge_index):
    num_nodes = X.size(0)
    A_hat = normalize_adjacency(edge_index, num_nodes)    
    return A_hat 

def get_high_pass_filter(X, edge_index):
    num_nodes = X.size(0)
    A_hat = normalize_adjacency(edge_index, num_nodes)    
    I = torch.eye(num_nodes, device=X.device)
    return (I-A_hat)

class FilterbankFUG(nn.Module):
    def __init__(self, in_dim, hid_dim, activation, num_pretrain_dataset, gcn_layers, 
        dropout, type_,  alpha=1.0, ablation='all', 
        n_sample=183, if_rand=False, sampling='random', 
        de_loss=1.0,  de_layers=1,  de_input='x', shared=False):
        super(FilterbankFUG, self).__init__()
        self.graphcledge = GraphCL(in_dim, hid_dim, activation)
        self.gcn = GcnLayers_PyG(in_dim, hid_dim, gcn_layers, dropout)
        self.loss = nn.BCEWithLogitsLoss()

        self.prompttype = type_
        self.ablation_choice = ablation
        self.combine = alpha

        self.de_loss = de_loss 
        self.de_input = de_input
        
        self.if_rand = if_rand
        self.sample_size = n_sample
        self.sampling = sampling 
        self.samplers = [Sampler(sample_size=n_sample, if_rand=if_rand, sampling=sampling) for _ in range(num_pretrain_dataset)]


        de_in_dim = n_sample * 2 if self.de_input == 'concat' else n_sample
        
        self.shared_low_dimension_encoder =  DimensionNN_FUG(de_in_dim, de_in_dim//2, in_dim, nn.PReLU, layers=de_layers) 
        
        self.low_dimension_encoder =  nn.ModuleList([DimensionNN_FUG(de_in_dim, de_in_dim//2, in_dim, nn.PReLU, layers=de_layers) 
                                                        for _ in range(num_pretrain_dataset)])
        self.high_dimension_encoder = nn.ModuleList([DimensionNN_FUG(de_in_dim, de_in_dim//2, in_dim, nn.PReLU, layers=de_layers) 
                                                        for _ in range(num_pretrain_dataset)])
        self.identity_dimension_encoder = nn.ModuleList([DimensionNN_FUG(de_in_dim, de_in_dim//2, in_dim, nn.PReLU, layers=de_layers) 
                                                        for _ in range(num_pretrain_dataset)])
        
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(num_pretrain_dataset)])

        self.shared = shared
        self.shared_token = textprompt(in_dim, type_)
        self.balancetoken_layers = nn.ModuleList([balanceprompt(in_dim, type_) for _ in range(num_pretrain_dataset)])

    def compute_prelogits_GRAPHCL(self,seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None, aggregated_feat=None):

        if aggregated_feat is None: 
            aggregated_feat = [None for _ in range(len(self.high_dimension_encoder))]

        for high_de, low_de, identity_de, weight, sampler, seq, adj in \
            zip(self.high_dimension_encoder, self.low_dimension_encoder, self.identity_dimension_encoder, self.weights, self.samplers, seq_list, adj_list):
            
            # X = AX
            low_seq = get_low_pass_filter(seq[0], adj[0]) @ seq[0] 
            # X = (I-A)X
            high_seq = get_high_pass_filter(seq[0], adj[0]) @ seq[0]

            # identity 
            identity_sample = sampler(seq[0], adj[0])
            low_sample = sampler(low_seq, adj[0])
            high_sample = sampler(high_seq, adj[0])


            identity_basis = identity_de(identity_sample)
            # low_basis = low_de(low_sample)
            low_basis = self.shared_low_dimension_encoder(low_sample)
            high_basis = high_de(high_sample)
            
            H_identity = [F.normalize(seq[i] @ identity_basis) for i in range(len(seq))]
            H_low = [F.normalize(seq[i] @ low_basis) for i in range(len(seq))]
            H_high = [F.normalize(seq[i] @ high_basis) for i in range(len(seq))]
            
            # 임베딩 가중합 
            # H_identity = self.graphcledge(self.gcn, 
            #                 H_identity[0], H_identity[1], H_identity[2], H_identity[3], 
            #                 adj[0], adj[1], adj[2], sparse, msk,
            #                 samp_bias1, samp_bias2, 'edge')
            
            # H_low = self.graphcledge(self.gcn, 
            #                 H_low[0], H_low[1], H_low[2], H_low[3], 
            #                 adj[0], adj[1], adj[2], sparse, msk,
            #                 samp_bias1, samp_bias2, 'edge')
            
            # H_high = self.graphcledge(self.gcn, 
            #                 H_high[0], H_high[1], H_high[2], H_high[3], 
            #                 adj[0], adj[1], adj[2], sparse, msk,
            #                 samp_bias1, samp_bias2, 'edge')
            
            # w = F.softmax(weight, dim=0)
            # emb_final = w[0]*H_low + w[1]*H_high + w[2]*H_identity
            # yield emb_final 

            H_identity = torch.stack(H_identity, dim=0)
            H_low = torch.stack(H_low, dim=0)
            H_high = torch.stack(H_high, dim=0)

            # 가중합 
            w = F.softmax(weight, dim=0)
            seq = w[0]*H_low + w[1]*H_high + w[2]*H_identity

            # 7. shared token 
            if self.shared: 
                seq = [F.relu(seq[i]) for i in range(len(seq))] # activation 
                seq = [self.shared_token(seq[i]) for i in range(len(seq))]# shared token 

            yield self.graphcledge(self.gcn, 
                                    seq[0], seq[1], seq[2], seq[3], 
                                    adj[0], adj[1], adj[2], sparse, msk,
                                    samp_bias1, samp_bias2, 'edge')
            


    def get_weights(self):
        combines = [self.combine]
        shared_token = self.shared_token.weight.detach()
        return self.high_dimension_encoder, self.shared_low_dimension_encoder, self.identity_dimension_encoder, combines, shared_token
    
    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl,  aggregated_feat=None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        
        logits = list(self.compute_prelogits_GRAPHCL(
            seq_list, 
            adj_list,  
            sparse, msk, samp_bias1, samp_bias2, 
            aggregated_feat))
        
        for i in range(len(logits)):
            loss = self.loss(logits[i], lbl[i]) # [1, 2*nodes] => [positive, negative]
            total_loss += loss
        
        l_de = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        h_de = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        i_de = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        for idx in range(len(self.high_dimension_encoder)): 
            # l_de += self.low_dimension_encoder[idx].dimensional_loss()
            h_de += self.high_dimension_encoder[idx].dimensional_loss()
            i_de += self.identity_dimension_encoder[idx].dimensional_loss()

        total_loss = total_loss + self.de_loss * (l_de + h_de + i_de)

        return total_loss
    
class PrePromptFilterFUG(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='all', scaling_factor=1.8, 
        n_sample=183, if_rand=False, de_loss=1.0, de_weight=False, n_mlp_layer=1, sampling='random', de_input='x', shared=False):
        super(PrePromptFilterFUG, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.gcn = GcnLayers_PyG(n_in, n_h, num_layers_num, dropout)
        self.loss = nn.BCEWithLogitsLoss()

        self.prompttype = type_
        self.ablation_choice = ablation
        self.combine = alpha

        self.de_loss = de_loss 
        self.de_weight = de_weight
        self.de_input = de_input
        
        self.if_rand = if_rand
        self.sample_size = n_sample
        self.sampling = sampling 
        self.samplers = [Sampler(sample_size=n_sample, if_rand=if_rand, sampling=sampling) for _ in range(num_pretrain_dataset_num)]

        self.shared = shared
        self.shared_token = textprompt(n_in, type_)

        if self.de_input == 'concat': 
            dim_in = n_sample * 2
        else: 
            dim_in = n_sample
        
        self.low_dimension_encoder = DimensionNN_FUG(dim_in, dim_in//2, n_in//2, nn.PReLU, layers=n_mlp_layer)
        self.high_dimension_encoder = nn.ModuleList([DimensionNN_FUG(dim_in, dim_in//2, n_in//2, nn.PReLU, layers=n_mlp_layer) 
                                                        for _ in range(num_pretrain_dataset_num)])
        
        self.balancetoken_layers = nn.ModuleList([balanceprompt(n_in, type_) for _ in range(num_pretrain_dataset_num)])


    def get_sample(self, sampler, bal_layer, seq, adj, agg_feat): 
        if self.de_input == 'x': 
            sample = sampler(seq, adj)
        elif self.de_input == 'ax': 
            sample = sampler(agg_feat, adj)   
        elif self.de_input == 'concat': 
            sample_AX = sampler(agg_feat, adj)  
            sample_X = sampler(seq, adj)
            sample = torch.cat([sample_X, sample_AX], dim=0)
            sample = bal_layer(sample.T).T
        return sample 
    
    def compute_prelogits_GRAPHCL(self,seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None, aggregated_feat=None):

        str_prelogits =  None 
        if aggregated_feat is None: 
            aggregated_feat = [None for _ in range(len(self.high_dimension_encoder))]

        for domain_encoder, bal_layer, seq, adj, sampler, agg_feat in \
            zip(self.high_dimension_encoder, self.balancetoken_layers, seq_list, adj_list, self.samplers, aggregated_feat):
            
            # 1. 피처 분류 
            if self.de_input == 'x': 
                low_freq_indices, high_freq_indices = split_features_by_frequency(seq[0], adj[0]) 
            elif self.de_input == 'ax':
                low_freq_indices, high_freq_indices = split_features_by_frequency(agg_feat, adj[0]) 
            elif self.de_input == 'concat': 
                low_idx_x, high_idx_x = split_features_by_frequency(seq[0], adj[0]) 
                low_idx_ax, high_idx_ax = split_features_by_frequency(agg_feat, adj[0]) 
            
            # 2. 샘플링 
            sample = self.get_sample(sampler, bal_layer, seq[0], adj[0], agg_feat)
            
            # 3. 저주파 DE 
            t_low = self.low_dimension_encoder(sample[:, low_freq_indices])
            H_low = [self.low_dimension_encoder.feature_sig_propagate(seq[i][:, low_freq_indices], t_low) for i in range(len(seq))]
            H_low = torch.stack(H_low, dim=0)

            # 4. 고주파 DE 
            t_high = domain_encoder(sample[:, high_freq_indices])
            H_high = [domain_encoder.feature_sig_propagate(seq[i][:, high_freq_indices], t_high) for i in range(len(seq))]
            H_high = torch.stack(H_high, dim=0)
            
            # 5. low DE, high DE concat 
            seq = torch.cat([H_low, H_high], dim=2) # [4, N, K]

            # 6. balance token 
            # seq = [bal_layer(seq[i]) for i in range(len(seq))]
            # sample = bal_layer(sample.T).T

            # 7. shared token 
            if self.shared: 
                seq = [self.shared_token(seq[i]) for i in range(len(seq))]# shared token 

            yield self.graphcledge(self.gcn, 
                                    seq[0], seq[1], seq[2], seq[3], 
                                    adj[0], adj[1], adj[2], sparse, msk,
                                    samp_bias1, samp_bias2, 'edge')

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_weights(self):
        balance_layers = [layer.weight.detach() for layer in self.balancetoken_layers]
        combines = [self.combine]
        shared_token = self.shared_token.weight.detach()
        return self.high_dimension_encoder, self.low_dimension_encoder, balance_layers, combines, shared_token
    
    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl,  aggregated_feat=None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        
        logits = list(self.compute_prelogits_GRAPHCL(
            seq_list, 
            adj_list,  
            sparse, msk, samp_bias1, samp_bias2, 
            aggregated_feat))
        
        for i in range(len(logits)):
            loss = self.loss(logits[i], lbl[i]) # [1, 2*nodes] => [positive, negative]
            total_loss += loss

        
        basis_mean = []
        de = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        for idx, dim_pretext in enumerate(self.high_dimension_encoder): 
            de += dim_pretext.dimensional_loss()
            basis_mean.append(dim_pretext.mean_basis_vector())
        basis_mean_loss = torch.stack(basis_mean).mean(dim=0).pow(2).mean()

        total_loss = total_loss + self.de_loss * (de +  basis_mean_loss + self.low_dimension_encoder.dimensional_loss() )

        return total_loss
    
class PrePromptSharedFUG(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='all', scaling_factor=1.8, 
        n_sample=183, if_rand=False, de_loss=1.0, de_weight=False, n_mlp_layer=1, sampling='random', de_input='x'):
        super(PrePromptSharedFUG, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.gcn = GcnLayers_PyG(n_in, n_h, num_layers_num, dropout)
        self.loss = nn.BCEWithLogitsLoss()

        self.prompttype = type_
        self.ablation_choice = ablation
        self.combine = alpha

        self.de_loss = de_loss 
        self.de_weight = de_weight
        self.de_input = de_input
        
        self.if_rand = if_rand
        self.sample_size = n_sample
        self.sampling = sampling 
        self.samplers = [Sampler(sample_size=n_sample, if_rand=if_rand, sampling=sampling) for _ in range(num_pretrain_dataset_num)]


        if self.de_input == 'concat': 
            dim_in = n_sample * 2
        else: 
            dim_in = n_sample
        
        # self.dimension_encoder_layers = nn.ModuleList([DimensionNN_FUG(dim_in, dim_in//2, n_in, nn.PReLU, layers=n_mlp_layer) 
        #                                                 for _ in range(num_pretrain_dataset_num)])
        # self.shared_dimension_encoder = DimensionNN_FUG(dim_in, dim_in, n_in, nn.PReLU, layers=n_mlp_layer)
        self.dimension_encoder_layers = nn.ModuleList([DimensionNN_FUG(dim_in, dim_in, n_in*2, nn.PReLU, layers=n_mlp_layer) 
                                                        for _ in range(num_pretrain_dataset_num)])
        self.shared_dimension_encoder = DimensionNN_FUG(dim_in, dim_in, n_in, nn.PReLU, layers=n_mlp_layer)
        self.balancetoken_layers = nn.ModuleList([balanceprompt(n_sample, type_) for _ in range(num_pretrain_dataset_num)])


    def get_sample(self, sampler, bal_layer, seq, adj, agg_feat): 
        if self.de_input == 'x': 
            sample = sampler(seq, adj)
        elif self.de_input == 'ax': 
            sample = sampler(agg_feat, adj)   
        elif self.de_input == 'concat': 
            sample_AX = sampler(agg_feat, adj)  
            sample_X = sampler(seq, adj)
            sample = torch.cat([sample_X, sample_AX], dim=0)
            sample = bal_layer(sample.T).T
        return sample 
    
        
    def compute_prelogits_GRAPHCL(self, dimension_encoder_layers,seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None, aggregated_feat=None):

        str_prelogits =  None 
        if aggregated_feat is None: 
            aggregated_feat = [None for _ in range(len(dimension_encoder_layers))]

        for dim_pretext, bal_layer, seq, adj, sampler, agg_feat in \
            zip(dimension_encoder_layers, self.balancetoken_layers, seq_list, adj_list, self.samplers, aggregated_feat):
            
           
            if self.ablation_choice == 'None':
                sample = self.get_sample(sampler, bal_layer, seq, adj, agg_feat)
                dimension_sig = dim_pretext(sample)
                
                seq = [dim_pretext.feature_sig_propagate(seq[i], dimension_sig) for i in range(len(seq))]


                agg_feat = dim_pretext.feature_sig_propagate(agg_feat, dimension_sig)

                sample_for_share = self.get_sample(sampler, bal_layer, seq, adj, agg_feat)    
                shared_dimension_sig = self.shared_dimension_encoder(sample_for_share)
                seq = [self.shared_dimension_encoder.feature_sig_propagate(seq[i], shared_dimension_sig) for i in range(len(seq))]

                yield self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
            
            elif self.ablation_choice == 'PCA': 
                yield self.graphcledge(self.gcn, 
                    seq[0], seq[1], seq[2], seq[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge')
            

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_weights(self):
        balance_layers = [layer.weight.detach() for layer in self.balancetoken_layers]
        combines = [self.combine]
        return self.dimension_encoder_layers, self.shared_dimension_encoder, balance_layers, combines
    
    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl,  aggregated_feat=None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        
        logits = list(self.compute_prelogits_GRAPHCL(
            self.dimension_encoder_layers, 
            seq_list, 
            adj_list,  
            sparse, msk, samp_bias1, samp_bias2, 
            aggregated_feat))
        
        for i in range(len(logits)):
            loss = self.loss(logits[i], lbl[i]) # [1, 2*nodes] => [positive, negative]
            total_loss += loss

        
        basis_mean = []
        de = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        for idx, dim_pretext in enumerate(self.dimension_encoder_layers): 
            de += dim_pretext.dimensional_loss()
            basis_mean.append(dim_pretext.mean_basis_vector())
        basis_mean_loss = torch.stack(basis_mean).mean(dim=0).pow(2).mean()
        total_loss = total_loss + self.de_loss * (de +  basis_mean_loss + self.shared_dimension_encoder.dimensional_loss() )

        return total_loss
    

class PrePromptACL(nn.Module):
    def __init__(self, n_in, n_h, num_pretrain_dataset_num, num_layers_num, 
                dropout, type_, temp, moving_average_decay=1.0, num_MLP=1,
                alpha=1.0, n_sample=183, if_rand=False, de_loss=1.0, 
                de_weight=False, n_mlp_layer=1, sampling='random', ablation='all', proj_mode='domain'):
        super(PrePromptACL, self).__init__()
        
        self.encoder = GcnLayers_PyG(n_in, n_h, num_layers_num, dropout)
        self.encoder_target = copy.deepcopy(self.encoder)
        set_requires_grad(self.encoder_target, False)  
        self.target_ema_updater = EMA(moving_average_decay) #EMA updater

        self.combine = alpha
        self.prompttype = type_
        self.ablation_choice = ablation
        self.de_weight = de_weight
        self.de_loss = de_loss 
        
        self.temp = temp                                    #contrastive temperature
        self.out_dim = n_h
        self.num_MLP = num_MLP

        self.if_rand = if_rand
        self.sample_size = n_sample
        self.sampling = sampling 
        self.samplers = [Sampler(sample_size=n_sample, if_rand=if_rand, sampling=sampling) for _ in range(num_pretrain_dataset_num)]
        
        self.dimension_encoder_layers = nn.ModuleList([DimensionNN_FUG(n_sample, n_sample//2, n_in, nn.PReLU, layers=n_mlp_layer) 
                                                       for _ in range(num_pretrain_dataset_num)])

        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) 
                                                    for _ in range(num_pretrain_dataset_num)])
        
        self.projector_mode = proj_mode

        if self.projector_mode == 'all': 
            self.projector = Predictor(n_h, n_h, num_MLP) 
        else: 
            self.projectors = nn.ModuleList([Predictor(n_h, n_h, num_MLP) 
                                        for _ in range(num_pretrain_dataset_num)])

        # if backbone == 'norm_mdgpt': 
        #     self.encoder = GNAELayers(n_in, n_h, num_layers_num, dropout, scaling_factor=scaling_factor)
        #     self.encoder_target = copy.deepcopy(self.encoder)


    def get_reduction_feat(self, feat, idx): 
        sample = self.samplers[idx](feat) 
        dimension_sig = self.dimension_encoder_layers[idx](sample) # [dimension size, unify_dim] basis matrix 
        seq = self.dimension_encoder_layers[idx].feature_sig_propagate(feat, dimension_sig)
        return seq 
    
    def get_target_embedding(self, feat, edge_index, idx):
        seq = self.get_reduction_feat(feat, idx)
        h = self.encoder_target(feat, edge_index)
        return h.detach()

    def get_embedding(self, feat, edge_index, idx):
        seq = self.get_reduction_feat(feat, idx)
        h = self.encoder(feat, edge_index)
        return h.detach()
    
    def get_projector_embedding(self, feat, edge_index, idx):
        seq = self.get_reduction_feat(feat, idx)
        h = self.encoder(feat, edge_index)
        if self.projector_mode == 'all': 
            h = self.projector(h)
        else: 
            h = self.projectors[idx](h)
        return h.detach()
    
    def pos_score(self, x, edge_index, v, u, idx):
        if self.projector_mode == 'all': 
            q = F.normalize(self.projector(v), dim=-1)
        else: 
            q = F.normalize(self.projectors[idx](v), dim=-1)
        u = F.normalize(u, dim=-1)
        src, dst = edge_index  # [2, num_edges]
        sim = (u[src] * q[dst]).sum(dim=1) / self.temp
        pos_score = scatter_mean(sim, dst, dim=0, dim_size=x.size(0))  # x.size(0) = num_nodes
        return pos_score
    
        graph.ndata['q'] = F.normalize(self.projectors[idx](v))
        graph.ndata['u'] = F.normalize(u, dim=-1)
        graph.apply_edges(fn.u_mul_v('u', 'q', 'sim'))
        graph.edata['sim'] = graph.edata['sim'].sum(1) / self.temp
        graph.update_all(fn.copy_e('sim', 'm'), fn.mean('m', 'pos'))
        pos_score = graph.ndata['pos']
        return pos_score, graph

    def neg_score(self, z):
        z = F.normalize(z, dim=-1)
        sim_matrix = torch.exp(torch.mm(z, z.t()) / self.temp)
        
        # 자기 자신 제외 
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.size(0), -1)
        neg_score = sim_matrix.mean(dim=1) 
        return neg_score
    
        z = F.normalize(h, dim=-1)
        graph.edata['sim'] = torch.exp(graph.edata['sim'])
        neg_sim = torch.exp(torch.mm(z, z.t()) / self.temp)
        neg_score = neg_sim.sum(1)
        graph.ndata['neg_sim'] = neg_score
        graph.update_all(udf_u_add_log_e, fn.mean('m', 'neg'))
        neg_score = graph.ndata['neg']
        return neg_score

    def update_moving_average(self):
        # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.encoder_target is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.encoder_target, self.encoder)
    
    def get_weights(self):
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        combines = [self.combine]
        return self.dimension_encoder_layers, fea_pretext_weights, combines
    
    def forward(self, feat_list, edge_list, agg_feat_list=None): 
        total_loss_cl = torch.tensor(0.0, dtype=torch.float32).to(feat_list[0].device)
        total_loss_de = torch.tensor(0.0, dtype=torch.float32).to(feat_list[0].device)

        if agg_feat_list is None: 
            agg_feat_list = [None for _ in range(len(feat_list))]

        for idx, (dim_pretext, fea_pretext, feat, edge_index, sampler, agg_feat) in \
            enumerate(zip(self.dimension_encoder_layers, self.feature_prompt_layers, feat_list, edge_list, self.samplers, agg_feat_list)):
            # PCA 면 feat 그대로 사용. 
            if self.ablation_choice != 'PCA': 
                # 샘플 추출 
                if agg_feat is not None: 
                    sample = sampler(agg_feat)
                    dimension_sig = dim_pretext(sample)
                    # (AX+X)T
                    feat = dim_pretext.feature_sig_propagate(agg_feat, dimension_sig)
                    
                    # XT
                    # feat = dim_pretext.feature_sig_propagate(feat, dimension_sig)
                else: 
                    sample = sampler(feat) 
                    
                    # 샘플 대상으로 basis 벡터 추출 
                    dimension_sig = dim_pretext(sample) # [dimension size, unify_dim] basis matrix 
                    feat = dim_pretext.feature_sig_propagate(feat, dimension_sig)

            if self.ablation_choice != 'None' and self.ablation_choice != 'PCA': # domain token 적용
                feat = fea_pretext(feat) 
            
            v = self.encoder(feat, edge_index).squeeze(0) 
            u = self.encoder_target(feat, edge_index).squeeze(0) 
            pos_score = self.pos_score(feat, edge_index, v, u, idx)
            neg_score = self.neg_score(v)
            # print("pos_score mean:", pos_score.mean().item())
            # print("neg_score mean:", neg_score.mean().item())
            loss_acl = (- pos_score + neg_score).mean()
            loss_de = dim_pretext.dimensional_loss()
            # loss 계산 
            total_loss_cl += loss_acl 
            total_loss_de += loss_de

        total_loss = total_loss_cl +  self.de_loss * total_loss_de

        return total_loss

class PrePromptFUG(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='all', scaling_factor=1.8, 
        n_sample=183, if_rand=False, de_loss=1.0, de_weight=False, n_mlp_layer=1, sampling='random', de_input='x', shared=False):
        super(PrePromptFUG, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_
        
        self.if_rand = if_rand
        self.sample_size = n_sample
        self.de_loss = de_loss 

        self.sampling = sampling 
        self.samplers = [Sampler(sample_size=n_sample, if_rand=if_rand, sampling=sampling) for _ in range(num_pretrain_dataset_num)]

        self.de_input = de_input

        if self.de_input == 'concat': 
            self.dimension_encoder_layers = nn.ModuleList([DimensionNN_FUG(n_sample*3, n_sample, n_in, nn.PReLU, layers=n_mlp_layer) 
                                                           for _ in range(num_pretrain_dataset_num)])
        else : # X or AX 만 사용 
            self.dimension_encoder_layers = nn.ModuleList([DimensionNN_FUG(n_sample, n_sample//2, n_in, nn.PReLU, layers=n_mlp_layer) 
                                                        for _ in range(num_pretrain_dataset_num)])

        # self.balancetoken_layers = nn.ModuleList([textprompt(n_sample*2, type_) for _ in range(num_pretrain_dataset_num)])
        self.balancetoken_layers = nn.ModuleList([balanceprompt(n_sample, type_) for _ in range(num_pretrain_dataset_num)])
        
        self.shared = shared
        self.shared_token = textprompt(n_in, type_)

        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) 
            for _ in range(num_pretrain_dataset_num)])

        self.structure_prompt_layers = nn.ModuleList([
            nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)])
            for _ in range(num_pretrain_dataset_num)])

        if backbone == 'norm_mdgpt': 
            self.gcn = GNAELayers(n_in, n_h, num_layers_num, dropout, scaling_factor=scaling_factor)
        elif backbone == 'gcn': 
            # self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
            self.gcn = GcnLayers_PyG(n_in, n_h, num_layers_num, dropout)

            

        self.combine = alpha

        self.loss = nn.BCEWithLogitsLoss()

        self.ablation_choice = ablation

        self.de_weight = de_weight

    def ablation(self, fea_prelogits, str_prelogits):
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits
        
    def compute_prelogits_GRAPHCL(self, dimension_encoder_layers, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None, aggregated_feat=None):

        str_prelogits =  None 
        if aggregated_feat is None: 
            aggregated_feat = [None for _ in range(len(dimension_encoder_layers))]

        for dim_pretext, fea_pretext, bal_layer, str_layers, seq, adj, sampler, agg_feat in \
            zip(dimension_encoder_layers, feature_prompt_layers, self.balancetoken_layers, structure_prompt_layers, seq_list, adj_list, self.samplers, aggregated_feat):
            
           
            if self.ablation_choice == 'None':
                # 샘플 추출 
                # if aggregated_feat :
                    # sample = sampler(agg_feat, adj[0])
                if self.de_input == 'x': 
                    sample = sampler(seq[0], adj[0])
                elif self.de_input == 'ax': 
                    sample = sampler(agg_feat, adj[0])   
                elif self.de_input == 'concat': 
                    # sample_AX = sampler(agg_feat, adj[0])  
                    

                    high_X = get_high_pass_filter(seq[0], adj[0]) @ seq[0]
                    low_X = get_low_pass_filter(seq[0], adj[0]) @ seq[0]

                    sample_I = sampler(seq[0], adj[0])
                    sample_L = sampler(low_X, adj[0])
                    sample_H = sampler(high_X, adj[0])

                    sample = torch.cat([sample_I, sample_L, sample_H], dim=0)
                    sample = bal_layer(sample.T).T

                dimension_sig = dim_pretext(sample)
                
                seq = [dim_pretext.feature_sig_propagate(seq[i], dimension_sig) for i in range(len(seq))]
                if self.shared: 
                    seq = [F.relu(seq[i]) for i in range(len(seq))] # activation 
                    seq = [self.shared_token(seq[i]) for i in range(len(seq))]# shared token 

                yield self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
            
            elif self.ablation_choice == 'PCA': 
                yield self.graphcledge(self.gcn, 
                    seq[0], seq[1], seq[2], seq[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge')
            elif self.ablation_choice == 'DEst': 
                # 샘플 추출 
                if self.sampling == 'degree': 
                    sample = sampler(seq[0], adj[0])
                else: 
                    sample = sampler(seq[0])
                # 샘플 대상으로 basis 벡터 추출 
                dimension_sig = dim_pretext(sample) # [dimension size, unify_dim] basis matrix 
                seq = [dim_pretext.feature_sig_propagate(seq[i], dimension_sig) for i in range(len(seq))]
                
                yield self.graphcledge(self.gcn, 
                        seq[0], seq[1], seq[2], seq[3], 
                        adj[0], adj[1], adj[2], sparse, msk,
                        samp_bias1, samp_bias2, 'edge', str_layers)

            else:
                # 샘플 추출 
                # sample = dimensional_sample_random(self.sample_size, seq[0], if_rand=self.if_rand)
                sample = sampler(seq[0])
                # 샘플 대상으로 basis 벡터 추출 
                dimension_sig = dim_pretext(sample)
                preseq = [dim_pretext.feature_sig_propagate(seq[i], dimension_sig) for i in range(len(seq))]
                
                preseq_list = [fea_pretext(preseq[i]) for i in range(len(preseq))] 
                #print(f'(((((preseq_list)))))\n{preseq_list[0][0]}\n\n{preseq_list[1][0]}')
                fea_prelogits = self.graphcledge(self.gcn, 
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')

                if self.ablation == 'all': 
                    str_prelogits = self.graphcledge(self.gcn, 
                        seq[0], seq[1], seq[2], seq[3], 
                        adj[0], adj[1], adj[2], sparse, msk,
                        samp_bias1, samp_bias2, 'edge', str_layers)
                
                yield self.ablation(fea_prelogits, str_prelogits)

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_weights(self):
        
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [
            [layer.weight.detach() for layer in structure_prompt_layer]
            for structure_prompt_layer in self.structure_prompt_layers
        ]
        balance_layers = [layer.weight.detach() for layer in self.balancetoken_layers]
        shared_token = self.shared_token.weight.detach()
        
        combines = [self.combine]
        return self.dimension_encoder_layers, fea_pretext_weights, balance_layers, combines, shared_token
    
    def get_token(self, seq): # 전체 그래프 한번에 입력 
        # 1. dimension encoder 
        arr = [] 
        for dim_pretext, graph in zip(self.dimension_encoder_layers, seq): 
            sample = dimensional_sample_random(self.sample_size, graph, if_rand=self.if_rand)
            # 샘플 대상으로 basis 벡터 추출 
            dimension_sig = dim_pretext(sample)
            seq = dim_pretext.feature_sig_propagate(graph, dimension_sig)
            arr.append(seq)
        return arr 

    def get_forward(self, seq, adj, sparse, msk, LP, i): 
        # print('seq: ', seq.size())
        # print('adj: ', adj.size())

        # 1. dimension encoder 
        sample = dimensional_sample_random(self.sample_size, seq, if_rand=self.if_rand)

        # 샘플 대상으로 basis 벡터 추출 
        dimension_sig = self.dimension_encoder_layers[i](sample)
        seq = self.dimension_encoder_layers[i].feature_sig_propagate(seq, dimension_sig)
        
        # 2. feature prompt 
        if self.ablation_choice != 'None':
            print('Not None')
            seq = self.feature_prompt_layers[i](seq)
            # print('token: ', token.size())

        emb = self.embed(seq, adj, sparse, msk, LP)[0].squeeze(0)
        print('emb: ', emb.size())
        return emb 
    
    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl, samples = None, aggregated_feat=None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        loss_list = []
        de_list = []
        
        logits = list(self.compute_prelogits_GRAPHCL(
            self.dimension_encoder_layers, 
            self.feature_prompt_layers, 
            self.structure_prompt_layers,
            seq_list, 
            adj_list,  
            sparse, msk, samp_bias1, samp_bias2, 
            aggregated_feat))
        for i in range(len(logits)):
            #print(f'lbl[i]: {lbl[i].size()}\n {lbl[i]}')
            loss = self.loss(logits[i], lbl[i]) # [1, 2*nodes] => [positive, negative]
            # loss_list.append(loss.item())
            total_loss += loss
        
        if self.de_weight: 
            dims = [seq[0].shape[1] for seq in seq_list]
            m = len(dims)

            inv_dims = [1.0 / d for d in dims]
            norm = sum(inv_dims)
            weights = [m * (inv_d / norm) for inv_d in inv_dims]

        de = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)

        basis_mean = []

        for idx, dim_pretext in enumerate(self.dimension_encoder_layers): 
            if self.de_weight: 
                de += weights[idx] * dim_pretext.dimensional_loss()
            else: 
                de += dim_pretext.dimensional_loss()
                # de_list.append(dim_pretext.dimensional_loss().item())
            basis_mean.append(dim_pretext.mean_basis_vector())
        # basis_mean_loss = torch.stack(basis_mean).pow(2).mean()
        basis_mean_loss = torch.stack(basis_mean).mean(dim=0).pow(2).mean()
        total_loss = total_loss + self.de_loss * de + self.de_loss * basis_mean_loss
        # print(f'cl loss: {total_loss}\n')
        # print(f'de loss: {de}\n')
        # print(f'basis mean loss: {basis_mean_loss}\n')

        return total_loss
    
class PrePromptNorm(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='all', scaling_factor=1.8):
        super(PrePromptNorm, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_
        
        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) 
            for _ in range(num_pretrain_dataset_num)])

        self.structure_prompt_layers = nn.ModuleList([
            nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)])
            for _ in range(num_pretrain_dataset_num)])

        self.gcn = GNAELayers(n_in, n_h, num_layers_num, dropout, scaling_factor=scaling_factor)

        self.combine = alpha

        self.loss = nn.BCEWithLogitsLoss()

        self.ablation_choice = ablation

    def ablation(self, fea_prelogits, str_prelogits):
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits
         
    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list, 
        sparse = False):
        for fea_pretext, str_layers, seq, adj in \
            zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq) , adj, sparse) 
                str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
                yield self.ablation(fea_prelogits, str_prelogits)
        
    def compute_prelogits_GRAPHCL(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None):
        for fea_pretext, str_layers, seq, adj in \
            zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
            else:
                preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))] 
                #print(f'(((((preseq_list)))))\n{preseq_list[0][0]}\n\n{preseq_list[1][0]}')
                fea_prelogits = self.graphcledge(self.gcn, 
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')

                str_prelogits = self.graphcledge(self.gcn, 
                    seq[0], seq[1], seq[2], seq[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge', str_layers)
                
                yield self.ablation(fea_prelogits, str_prelogits)

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_weights(self):
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [
            [layer.weight.detach() for layer in structure_prompt_layer]
            for structure_prompt_layer in self.structure_prompt_layers
        ]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines
    
    def get_token(self, seq): # 전체 그래프 한번에 입력 
        arr = [] 
        for fea_pretext, graph in zip(self.feature_prompt_layers, seq):
            # print(graph.size())
            tokened = fea_pretext(graph[0])
            arr.append(tokened)
        return arr 
    
    def get_forward(self, seq, adj, sparse, msk, LP, i): 
        print('seq: ', seq.size())
        print('adj: ', adj.size())
        token = self.feature_prompt_layers[i](seq)
        print('token: ', token.size())
        emb = self.embed(token, adj, sparse, msk, LP)[0].squeeze(0)
        print('emb: ', emb.size())
        return emb 
    
    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl, samples = None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        if samples == None:
            logits = list(self.compute_prelogits_GRAPHCL(
                self.feature_prompt_layers, 
                self.structure_prompt_layers,
                seq_list, 
                adj_list,  
                sparse, msk, samp_bias1, samp_bias2))
            for i in range(len(logits)):
                #print(f'lbl[i]: {lbl[i].size()}\n {lbl[i]}')
                loss = self.loss(logits[i], lbl[i]) # [1, 2*nodes] => [positive, negative]
                total_loss += loss
        else:
            logits = list(self.compute_prelogits_LP(
                self.feature_prompt_layers, 
                self.structure_prompt_layers,
                seq_list, 
                adj_list, 
                sparse))
            if type(samples) == list:
                samples = [torch.tensor(sample, dtype=torch.int64).to(seq_list[0].device)
                    for sample in samples] 
                for i in range(len(logits)):    
                    loss = compareloss(logits[i], samples[i], temperature=1)
                    total_loss += loss
            else:
                samples = torch.tensor(samples, dtype=torch.int64).to(seq_list[0].device)
                logits = torch.cat(logits, dim=0)
                total_loss = compareloss(logits, samples, temperature=1)

        return total_loss
    
class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='all', shared=False):
        super(PrePrompt, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_
        
        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) 
            for _ in range(num_pretrain_dataset_num)])

        self.structure_prompt_layers = nn.ModuleList([
            nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)])
            for _ in range(num_pretrain_dataset_num)])

        # self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        self.gcn = GcnLayers_PyG(n_in, n_h, num_layers_num, dropout)

        if backbone == 'gat':
            self.gcn = GatLayers(n_in, n_h, num_layers_num, dropout)
            str_prompt = [textprompt(n_h * self.gcn.heads, type_) for _ in range(num_layers_num)]
            #str_prompt.append(textprompt(n_h, type_))
            self.structure_prompt_layers =  nn.ModuleList([
                nn.ModuleList(copy.deepcopy(str_prompt))
                for _ in range(num_pretrain_dataset_num)])

        self.combine = alpha

        self.loss = nn.BCEWithLogitsLoss()

        self.ablation_choice = ablation

        self.shared = shared
        self.shared_token = textprompt(n_in, type_)


    def ablation(self, fea_prelogits, str_prelogits):
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits
         
    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list, 
        sparse = False):
        for fea_pretext, str_layers, seq, adj in \
            zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq) , adj, sparse) 
                str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
                yield self.ablation(fea_prelogits, str_prelogits)
        
    def compute_prelogits_GRAPHCL(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None):
        for fea_pretext, str_layers, seq, adj in \
            zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
            else:
                preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))] 
                #print(f'(((((preseq_list)))))\n{preseq_list[0][0]}\n\n{preseq_list[1][0]}')
                if self.shared: 
                    seq = [F.relu(seq[i]) for i in range(len(seq))] # activation 
                    preseq_list = [self.shared_token(seq[i]) for i in range(len(seq))]

                fea_prelogits = self.graphcledge(self.gcn, 
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')

                str_prelogits = self.graphcledge(self.gcn, 
                    seq[0], seq[1], seq[2], seq[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge', str_layers)
                
                yield self.ablation(fea_prelogits, str_prelogits)

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_weights(self):
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [
            [layer.weight.detach() for layer in structure_prompt_layer]
            for structure_prompt_layer in self.structure_prompt_layers
        ]
        combines = [self.combine]
        shared_token = self.shared_token.weight.detach()
        return fea_pretext_weights, str_pretext_weights, combines, shared_token
    
    def get_token(self, seq): # 전체 그래프 한번에 입력 
        arr = [] 
        for fea_pretext, graph in zip(self.feature_prompt_layers, seq):
            # print(graph.size())
            tokened = fea_pretext(graph[0])
            arr.append(tokened)
        return arr 
    
    def get_forward(self, seq, adj, sparse, msk, LP, i): 
        # print('seq: ', seq.size())
        # print('adj: ', adj.size())
        token = self.feature_prompt_layers[i](seq)
        # print('token: ', token.size())
        emb = self.embed(token, adj, sparse, msk, LP)[0].squeeze(0)
        # print('emb: ', emb.size())
        return emb 
    
    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl, samples = None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        if samples == None:
            logits = list(self.compute_prelogits_GRAPHCL(
                self.feature_prompt_layers, 
                self.structure_prompt_layers,
                seq_list, 
                adj_list,  
                sparse, msk, samp_bias1, samp_bias2))
            for i in range(len(logits)):
                #print(f'lbl[i]: {lbl[i].size()}\n {lbl[i]}')
                loss = self.loss(logits[i], lbl[i]) # [1, 2*nodes] => [positive, negative]
                total_loss += loss
        else:
            logits = list(self.compute_prelogits_LP(
                self.feature_prompt_layers, 
                self.structure_prompt_layers,
                seq_list, 
                adj_list, 
                sparse))
            if type(samples) == list:
                samples = [torch.tensor(sample, dtype=torch.int64).to(seq_list[0].device)
                    for sample in samples] 
                for i in range(len(logits)):    
                    loss = compareloss(logits[i], samples[i], temperature=1)
                    total_loss += loss
            else:
                samples = torch.tensor(samples, dtype=torch.int64).to(seq_list[0].device)
                logits = torch.cat(logits, dim=0)
                total_loss = compareloss(logits, samples, temperature=1)

        return total_loss


def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def udf_u_add_log_e(edges):
    return {'m': torch.log(edges.dst['neg_sim'] + edges.data['sim'])}

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def dimensional_sample_random(sample_size, x, if_rand=False):
    with torch.no_grad():
        if if_rand != True:
            d_sample_matrix = x[:sample_size, :]
        else:
            d_sample_matrix = x[torch.randperm(x.shape[0]),:][:sample_size, :]
        return d_sample_matrix

def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
 
    return res

def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))

def compareloss(feature,tuples,temperature):
    h_tuples=mygather(feature,tuples)
    temp = torch.arange(0, len(tuples)).to(feature.device)
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp)

    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))

    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return torch.tensor(res.astype(int) )

def sliced_wasserstein_torch(X, Y, n_proj=100):
        d = X.shape[1]
        projections = torch.randn(n_proj, d, device=X.device)
        projections = projections / torch.norm(projections, dim=1, keepdim=True)  # normalize

        X_proj = X @ projections.T  # [N, n_proj]
        Y_proj = Y @ projections.T  # [N, n_proj]

        min_len = min(X.shape[0], Y.shape[0])

        if X.shape[0] > min_len:
            idx = torch.randperm(X.shape[0], device=X.device)[:min_len]
            X_proj = X_proj[idx]

        if Y.shape[0] > min_len:
            idx = torch.randperm(Y.shape[0], device=Y.device)[:min_len]
            Y_proj = Y_proj[idx]

        # ✅ 샘플링된 후 정렬
        X_proj_sorted = X_proj.sort(dim=0)[0].clone()
        Y_proj_sorted = Y_proj.sort(dim=0)[0].clone()

        wasserstein_1d = (X_proj_sorted - Y_proj_sorted).abs().mean(dim=0)  # [n_proj]
        return wasserstein_1d.mean()