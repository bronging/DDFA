import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers, MLP, GatLayers, FeatureMLP, GNAELayers
from layers import AvgReadout, NodeFeaturePermMLP,build_hard_permutation_from_logits
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from layers.prompt import *
import copy

class PrePromptwithMLPToken(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, 
                 num_layers_num, dropout, type_='add', alpha=1.0, ablation='all', 
                 n_mlp_layer=1, init_identity=False):
        super(PrePromptwithMLPToken, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        
        self.feature_MLP_layers = nn.ModuleList(
            [FeatureMLP(in_dim=n_in, hidden_dim=n_in, out_dim=n_in, num_layer=n_mlp_layer, init_identity=init_identity) 
                for _ in range(num_pretrain_dataset_num)])

        self.feature_prompt_layers = nn.ModuleList(
            [textprompt(n_in, type_) 
                for _ in range(num_pretrain_dataset_num)])
        
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)

        self.combine = alpha

        self.loss = nn.BCEWithLogitsLoss()

        self.ablation_choice = ablation

         
    def compute_prelogits_GRAPHCL(self, seq_list, adj_list, graphid,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None):

        seq = seq_list[0]
        adj = adj_list[0]

        if graphid == 0: 
            return self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
        else: 
            preseq_list = [self.feature_MLP_layers[graphid](seq[i]) for i in range(len(seq))]
            token_list = [self.feature_prompt_layers[graphid](preseq_list[i]) for i in range(len(preseq_list))]
            return self.graphcledge(self.gcn, 
                    token_list[0], token_list[1], token_list[2], token_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')

    def embed(self, seq, adj, sparse, msk, LP, graphid, grad=True):
        if graphid == 0: 
            h_1 = self.gcn(seq, adj, sparse, LP)
            c = self.read(h_1, msk)
        else: 
            feat = self.feature_MLP_layers[graphid](seq)
            feat2 = self.feature_prompt_layers[graphid](seq)
            h_1 = self.gcn(feat, adj, sparse, LP)
            c = self.read(h_1, msk)

        if grad: 
            return h_1, c
        else: 
            return h_1.detach(), c.detach()

    def forward(self, seq_list, adj_list, sparse, msk, graphid,
        samp_bias1, samp_bias2, lbl, samples = None,):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        
        if samples == None:
            logits = list(self.compute_prelogits_GRAPHCL(
                seq_list, 
                adj_list,  
                graphid,
                sparse, msk, samp_bias1, samp_bias2))
            
            for i in range(len(logits)):
                loss = self.loss(logits[i], lbl[i].squeeze(0))
                total_loss += loss

        return total_loss
    
    def get_weights(self):
        fea_domain_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]

        return fea_domain_weights 

class PrePromptwithMLP(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, alpha=1.0, ablation='all', n_mlp_layer=1, init_identity=False, mlp_bias=True):
        super(PrePromptwithMLP, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        
        self.feature_MLP_layers = nn.ModuleList(
            [FeatureMLP(in_dim=n_in, hidden_dim=n_in, out_dim=n_in, num_layer=n_mlp_layer, init_identity=init_identity, mlp_bias=mlp_bias) 
                for _ in range(num_pretrain_dataset_num)])

        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)

        self.combine = alpha

        self.loss = nn.BCEWithLogitsLoss()

        self.ablation_choice = ablation

        for param in self.feature_MLP_layers[0].parameters():
            param.requires_grad = False
         
    def compute_prelogits_GRAPHCL(self, seq_list, adj_list, graphid,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None):

        seq = seq_list[0]
        adj = adj_list[0]

        if graphid == 0: 
            return self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
        else: 
            preseq_list = [self.feature_MLP_layers[graphid](seq[i]) for i in range(len(seq))]
            return self.graphcledge(self.gcn, 
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')

    def embed(self, seq, adj, sparse, msk, LP, graphid, grad=True):
        if graphid == 0: 
            h_1 = self.gcn(seq, adj, sparse, LP)
            c = self.read(h_1, msk)
        else: 
            feat = self.feature_MLP_layers[graphid](seq)
            h_1 = self.gcn(feat, adj, sparse, LP)
            c = self.read(h_1, msk)

        if grad: 
            return h_1, c
        else: 
            return h_1.detach(), c.detach()

    def forward(self, seq_list, adj_list, sparse, msk, graphid,
        samp_bias1, samp_bias2, lbl, samples = None,):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        
        if samples == None:
            logits = list(self.compute_prelogits_GRAPHCL(
                seq_list, 
                adj_list,  
                graphid,
                sparse, msk, samp_bias1, samp_bias2))
            
            for i in range(len(logits)):
                loss = self.loss(logits[i], lbl[i].squeeze(0))
                total_loss += loss

        return total_loss

class MDGPTwithPerm(nn.Module): 
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='ft', perm_hid_dim=128, perm_n_layers=2, mlp_init=True):
        super(MDGPTwithPerm, self).__init__()
        self.n_in = n_in
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_
        self.n_graphs = num_pretrain_dataset_num
        
        self.perm_layers = nn.ModuleList([NodeFeaturePermMLP(d=n_in, hidden_dim=perm_hid_dim, n_layers=perm_n_layers, mlp_init=mlp_init)
            for _ in range(num_pretrain_dataset_num)])
        
        self.feature_prompt_layers = nn.ModuleList([textprompt(n_in, type_) 
            for _ in range(num_pretrain_dataset_num)])

        self.structure_prompt_layers = nn.ModuleList([
            nn.ModuleList([textprompt(n_h, type_) for _ in range(num_layers_num)])
            for _ in range(num_pretrain_dataset_num)])

        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
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

    def ablation(self, fea_prelogits, str_prelogits):
        if self.ablation_choice == 'all':
            return fea_prelogits + self.combine * str_prelogits
        elif self.ablation_choice == 'st':
            return str_prelogits
        elif self.ablation_choice == 'ft':
            return fea_prelogits
        else:
            return fea_prelogits + self.combine * str_prelogits
        
    def compute_prelogits_GRAPHCL(self, perm_layers, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None):
        
        results = []
        row_losss = [] 
        col_losss = [] 
        var_losss = []
        
        for perm, fea_pretext, str_layers, seq, adj in \
            zip(perm_layers, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                logit =  self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
            else:
                # permutation 추가 
                # 원본 피처만 입력으로 넣음. 
                # print("Original: \n", seq[0][0])  
                perm_matrix = perm(seq[0])
                # perm_matrix = build_hard_permutation_from_logits(perm_matrix)
                permuted_seq = [seq[i] @ perm_matrix.T for i in range(len(seq))]
                # print("Permutation: ", perm_matrix.argmax(dim=0))   
                # print("After Perm: ", permuted_seq[0][0])
                
                preseq_list = [fea_pretext(permuted_seq[i]) for i in range(len(permuted_seq))] 
                fea_prelogits = self.graphcledge(self.gcn, 
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')

                str_prelogits = self.graphcledge(self.gcn, 
                    seq[0], seq[1], seq[2], seq[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge', str_layers)
                
                logit =  self.ablation(fea_prelogits, str_prelogits)

                row_sum = perm_matrix.sum(dim=1)  # (d,)
                col_sum = perm_matrix.sum(dim=0)  # (d,)
                row_loss = F.mse_loss(row_sum, torch.ones_like(row_sum))
                col_loss = F.mse_loss(col_sum, torch.ones_like(col_sum))
                var_loss = -perm_matrix.var()
            results.append(logit)
            row_losss.append(row_loss)
            col_losss.append(col_loss)
            var_losss.append(var_loss)
        
        return results, torch.stack(row_losss).mean(), torch.stack(col_losss).mean(), torch.stack(var_losss).mean()

    def embed(self, seq, adj, sparse, msk, LP, graph_id):
        perm_matrix = self.perm_layers[graph_id](seq)
        seq = seq @ perm_matrix.T
        h_1 = self.gcn(seq, adj, sparse, LP)
        #c = self.read(h_1, msk)

        return h_1.detach()#, c.detach()

    def get_weights(self):
        fea_pretext_weights = [layer.weight.detach() for layer in self.feature_prompt_layers]
        str_pretext_weights = [
            [layer.weight.detach() for layer in structure_prompt_layer]
            for structure_prompt_layer in self.structure_prompt_layers
        ]
        combines = [self.combine]
        return fea_pretext_weights, str_pretext_weights, combines

    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl, samples = None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        if samples == None:
            logits, row_losss, col_losss, var_losss  = list(self.compute_prelogits_GRAPHCL(
                self.perm_layers,
                self.feature_prompt_layers, 
                self.structure_prompt_layers,
                seq_list, 
                adj_list,  
                sparse, msk, samp_bias1, samp_bias2))
            for i in range(len(logits)):
                loss = self.loss(logits[i], lbl[i])
                total_loss += loss
           
        else:
            logits= list(self.compute_prelogits_LP(
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

        return total_loss, row_losss, col_losss, var_losss

class PrePrompt2(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='all', 
        n_mlp_layer=1, init_identity=False, mlp_bias=True):
        super(PrePrompt2, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.gcn = GcnLayers(n_in, n_h, num_layers_num, dropout)
        self.loss = nn.BCEWithLogitsLoss()
        self.feature_prompt_layers = nn.ModuleList(
            [FeatureMLP(in_dim=n_in, hidden_dim=n_in, out_dim=n_in, num_layer=n_mlp_layer, 
                        init_identity=init_identity, mlp_bias=mlp_bias) 
                for _ in range(num_pretrain_dataset_num)])
        
        self.prompttype = type_
        self.combine = alpha
        self.ablation_choice = ablation

    def compute_prelogits_LP(self, feature_prompt_layers, seq_list, adj_list, 
        sparse = False):
        results = []

        for fea_pretext, seq, adj in \
            zip(feature_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                logit = self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq) , adj, sparse) 
                logit = fea_prelogits
            results.append(logit)
        return results 
                    
    def compute_prelogits_GRAPHCL(self, feature_prompt_layers,  seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None):
        
        results = []

        for fea_pretext, seq, adj in \
            zip(feature_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                logit = self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
            else:
                preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))] 
                logit = self.graphcledge(self.gcn, 
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')
                
            results.append(logit)
        return results
    
    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def get_token(self, seq): # 전체 그래프 한번에 입력 
        arr = [] 
        for i, graph in enumerate(seq): 
            print(graph.size())
            tokened = self.feature_prompt_layers[i](graph[0])
            arr.append(tokened)
        return arr 
    
    def get_weights(self):
        combines = [self.combine]
        return self.feature_prompt_layers, combines

    def forward(self, seq_list, adj_list, sparse, msk, 
        samp_bias1, samp_bias2, lbl, samples = None):        
        total_loss = torch.tensor(0.0, dtype=torch.float32).to(seq_list[0].device)
        if samples == None:
            logits = self.compute_prelogits_GRAPHCL(
                self.feature_prompt_layers, 
                seq_list, 
                adj_list,  
                sparse, msk, samp_bias1, samp_bias2)
            # logits_tensor = torch.stack(logits)        # shape: (N, D)
            # labels_tensor = torch.stack(lbl)           # shape: (N, D)
            # loss = self.loss(logits_tensor, labels_tensor)
            # total_loss.add_(loss)
            for i in range(len(logits)):
                loss = self.loss(logits[i], lbl[i])
                total_loss += loss
        else:
            logits = self.compute_prelogits_LP(
                self.feature_prompt_layers, 
                seq_list, 
                adj_list, 
                sparse)
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
