import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP, FeatureMLP, DimensionNN_FUG
from layers import GCN, AvgReadout, NodeFeaturePermMLP, build_hard_permutation_from_logits
import torch_scatter
from layers.prompt import *
from preprompt import dimensional_sample_random, Sampler, split_features_by_frequency

def compute_x_ax_weight(x, ax):
    # Normalize
    x_norm = F.normalize(x, dim=1)
    ax_norm = F.normalize(ax, dim=1)

    sim = (x_norm * ax_norm).sum(dim=1)  # cosine similarity per node
    sim_mean = sim.mean().item()  # 전체 평균

    # Higher similarity → less X importance
    alpha = sim_mean # X weight
    beta = 1.0 - sim_mean      # AX weight

    # Normalize
    alpha, beta = alpha / (alpha + beta), beta / (alpha + beta)
    return alpha, beta

def compute_degree_weight(edge_index, num_nodes):
    row, _ = edge_index
    deg = torch.bincount(row, minlength=num_nodes).float()
    norm_deg = deg / deg.sum()
    deg_entropy = -(norm_deg * norm_deg.clamp_min(1e-10).log()).sum().item()

    # 높은 entropy → 균등한 degree → AX 비중 ↑
    beta = deg_entropy / torch.log(torch.tensor(num_nodes)).item()  # normalize to [0,1]
    alpha = 1.0 - beta
    return alpha, beta

def compute_variance_weight(x, ax):
    var_x = x.var(dim=0).mean()
    var_ax = ax.var(dim=0).mean()

    alpha = var_x / (var_x + var_ax)
    beta = 1 - alpha
    return alpha.item(), beta.item()

def balance_compute(graph_embedding, alpha, beta): 
    half = graph_embedding.shape[1] // 2

    x_part = graph_embedding[:, :half] * alpha
    ax_part = graph_embedding[:, half:] * beta
    graph_embedding = torch.cat([x_part, ax_part], dim=1)
    return graph_embedding

from torch_geometric.utils import get_laplacian
from torch_sparse import SparseTensor

def spectral_energy_histogram(x, edge_index, n_bins=256):
    """
    x: [n, d] feature matrix
    eigvals: [n] Laplacian eigenvalues (오름차순)
    eigvecs: [n, n] Laplacian eigenvectors
    n_bins: histogram bin 개수 (default=256)
    
    return: [d, n_bins] 각 feature dimension별 normalized energy ratio
    """
    n, d = x.shape

    # 1. Laplacian 만들기
    # PyG: get_laplacian -> (edge_index, edge_weight)
    edge_index, edge_weight = get_laplacian(edge_index, normalization="sym", num_nodes=n)

    # SparseTensor 변환
    L = SparseTensor(row=edge_index[0], col=edge_index[1],
                     value=edge_weight, sparse_sizes=(n, n)).to_dense()

    # 2. 고유분해
    eigvals, eigvecs = torch.linalg.eigh(L) 

    # Fourier transform: U^T x
    coeffs = eigvecs.T @ x  # [n, d]
    energy = coeffs.pow(2)  # [n, d]

    # bin 정의 (λ_max = 2 고정)
    bins = torch.linspace(0, 2.0, n_bins+1, device=x.device)

    # feature별 histogram 초기화
    hist = torch.zeros(d, n_bins, device=x.device)
    for i in range(n_bins):
        mask = (eigvals >= bins[i]) & (eigvals < bins[i+1])
        if mask.any():
            hist[:, i] = energy[mask].sum(dim=0)  # bin별 에너지 합산

    # normalize (각 feature별 합이 1)
    # hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-9)

    return hist

def spectral_energy_distribution(x, edge_index, num_nodes=None, k=None):
    """
    x: [n, d] 노드 feature matrix
    edge_index: [2, E] PyG edge_index
    num_nodes: 노드 개수 (None이면 x.shape[0]으로 설정)
    k: (optional) 상위 k개의 고유값만 사용, None이면 전체 사용
    
    return:
        energy_dists: [d, k or n] 각 feature dimension별 spectral energy 분포
        eigvals: [k or n] 대응되는 라플라시안 고유값
    """
    if num_nodes is None:
        num_nodes = x.size(0)

    # 1. Laplacian 만들기
    # PyG: get_laplacian -> (edge_index, edge_weight)
    edge_index, edge_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)

    # SparseTensor 변환
    L = SparseTensor(row=edge_index[0], col=edge_index[1],
                     value=edge_weight, sparse_sizes=(num_nodes, num_nodes)).to_dense()

    # 2. 고유분해
    eigvals, eigvecs = torch.linalg.eigh(L)  # L 대칭행렬 -> 안정적

    # k개만 쓸 경우
    if k is not None and k < num_nodes:
        eigvals = eigvals[:k]
        eigvecs = eigvecs[:, :k]

    # 3. 각 feature column을 Fourier 변환
    #    hat{x} = U^T x
    #    energy = |hat{x}|^2
    energy_dists = []
    for j in range(x.size(1)):
        feat_j = x[:, j]  # [n]
        coeffs = torch.matmul(eigvecs.T, feat_j)  # [k or n]
        energy = coeffs.pow(2)  # spectral energy
        energy_dists.append(energy.unsqueeze(0))

    energy_dists = torch.cat(energy_dists, dim=0)  # [d, k or n]

    return energy_dists, eigvals 

from preprompt import get_low_pass_filter, get_high_pass_filter

class downstreampromptW1MLP(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, dim_pretexts,
                combines, type_ = 'mul', ablation = 'all', sample_size = 182, if_rand=False, 
                gamma=0.5, n_mlp_layer=1, sampling='random', agg_feat=None, de_input='x', shared=False, shared_token=None):
        super(downstreampromptW1MLP, self).__init__()

        self.sample_size = sample_size
        self.if_rand = if_rand 

        self.composed_mlp = composedW1MLP(len(dim_pretexts))
        self.open_mlp = FeatureMLP(in_dim=feature_dim, hidden_dim=feature_dim, out_dim=feature_dim, num_layer=n_mlp_layer, init_identity=False) 
        self.src_mlp = dim_pretexts

        for param in self.src_mlp.parameters():
            param.requires_grad = False

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.gamma = gamma 
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

        self.agg_feat = agg_feat

    def forward(self, seq, gcn, adj, sparse):
        
        composed_seq = self.composed_mlp(seq, self.src_mlp)
        open_seq = self.open_mlp(seq)
        seq = open_seq + composed_seq
        return gcn(seq, adj, sparse, None)

    def get_emb(self, seq): 
        open_seq = self.open_mlp(seq)
        return open_seq
    
class downpromptW1MLP(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  dim_pretext_weight, 
                  combines, type_='mul', ablation = 'all', sample_size = 182, 
                  if_rand=False, gamma=0.5, n_mlp_layer=1, 
                  sampling='random', agg_feat=None, shared=False, shared_token=None, de_input='x'):
        super(downpromptW1MLP, self).__init__()

        self.num_pretrain_datasets = len(dim_pretext_weight)
        

        self.downstreamPrompt = downstreampromptW1MLP(feature_dim, ft_in, num_layers_num, 
            dim_pretext_weight, 
            combines, type_, ablation, sample_size, if_rand, gamma,
            n_mlp_layer, sampling, agg_feat, de_input, shared, shared_token)

        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)
        self.agg_feat = agg_feat

    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0,batch=None):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        
        if batch != None: # graph classification 
            rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
        else: # node classification 
            rawret = embeds[idx]
        num =  rawret.shape[0]

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype 
        
        # 코사인 유사도 기반 분류 
        rawret = torch.cat((rawret, self.ave.to(rawret.device)), dim=0)  # shape: (B+C, D) -> Query node B 개와 C개의 prototype  비교 
        rawret = F.normalize(rawret, dim=1)            # 모든 row 벡터 L2 정규화
        rawret = rawret @ rawret.T                     # (N+K) x (N+K) 유사도 행렬 (cosine similarity)

        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

from preprompt import pca_compression
class downstreampromptBaryceter(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, dim_pretexts,
                combines, type_ = 'mul', ablation = 'all', sample_size = 182, if_rand=False, 
                gamma=0.5, n_mlp_layer=1, sampling='random', agg_feat=None, de_input='x', shared=False, shared_token=None, basis_matrix=None):
        super(downstreampromptBaryceter, self).__init__()

        self.sample_size = sample_size
        self.if_rand = if_rand 

        self.domain_token = textprompt(feature_dim, 'add')

        if ablation[-2:] == 'no': 
            self.task_prompt = composedBasisNode(feature_dim, len(basis_matrix), basis_matrix)
            self.task_prompt2 = composedBasisNode(feature_dim, len(basis_matrix), basis_matrix)
            self.task_prompt3 = composedBasisNode(feature_dim, len(basis_matrix), basis_matrix)
        else: 
            self.task_prompt = composedBasisNode(feature_dim, len(basis_matrix), None)
            self.task_prompt2 = composedBasisNode(feature_dim, len(basis_matrix), None)
            self.task_prompt3 = composedBasisNode(feature_dim, len(basis_matrix), None)
        
        self.composedprompt_dim = composedFUG(len(dim_pretexts), dim_pretexts, shared_token)
        self.open_prompt_dim = DimensionNN_FUG(sample_size, sample_size//2, feature_dim, nn.PReLU, layers=n_mlp_layer)
        # self.open_prompt_dim = DimensionNN_FUG(sample_size, sample_size, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.de_input = de_input 

        self.sampler = Sampler(sample_size=sample_size, if_rand=if_rand, sampling=sampling)
        self.sampling = sampling 
        self.sample_size = sample_size
        
        self.open_prompt_fea = textprompt(feature_dim, type_='add')
        

        self.shared = shared
        self.shared_token = shared_token

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.gamma = gamma 
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

        self.agg_feat = agg_feat
        # print(f'gamma: {self.gamma}')
    def get_composed(self): 
        return self.sampler(self.composed_feat)
    def forward(self, seq, gcn, adj, sparse):

        # if self.de_input == 'x': 
        #     sample = self.sampler(seq, adj)
        # elif self.de_input == 'ax': 
        #     sample = self.sampler(self.agg_feat, adj)
        sample = seq[1]
        seq = seq[0]
        
        dimension_sig_open = self.open_prompt_dim(sample)

        open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open)
        open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(self.agg_feat, dimension_sig_open)
        composed_dim_seq = self.composedprompt_dim(sample, seq, 'ww')
        # composed_dim_seq = self.composedprompt_dim(sample, self.agg_feat, 'ww')
        self.composed_feat = composed_dim_seq

        if self.ablation_choice[-4:-2] == 'do': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1)
            return gcn(open_dim_seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'd2': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq2)
            return gcn(open_dim_seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'd3': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq2)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'dq': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            return gcn(open_dim_seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'dz': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'hh': 
            open_dim_seq1 = self.domain_token(open_dim_seq1 + open_dim_seq2)
            composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'dn': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            # composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'no': 
            emb_o = gcn(open_dim_seq1 + open_dim_seq2, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'zx': 
            emb_o = gcn(open_dim_seq1 + open_dim_seq2, adj, sparse,  None)
            return emb_o
        elif self.ablation_choice[-4:-2] == 'z2': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            return emb_o
        elif self.ablation_choice[-4:-2] == 'kl': 
            composed_dim_seq = self.task_prompt2(composed_dim_seq)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_t
        elif self.ablation_choice[-4:-2] == 'dl': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            # composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return torch.cat([emb_o, emb_t], dim=2) 
        elif self.ablation_choice[-4:-2] == 'db': 
            # open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            # composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return torch.cat([emb_o, emb_t], dim=2) 
        elif self.ablation_choice[-4:-2] == 'km': 
            # open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            # composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(open_dim_seq2, adj, sparse,  None)
            return emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'cp': 
            # composed_dim_seq = self.task_prompt(open_dim_seq1)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_t
        elif self.ablation_choice[-4:-2] == 'de': 
            # open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            return gcn(open_dim_seq1 + open_dim_seq2, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'd3': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1)
            open_dim_seq2 = self.task_prompt3(open_dim_seq2)
            composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_2 = gcn(open_dim_seq2, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_2 + emb_t
        elif self.ablation_choice[-4:-2] == 'dj': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1)
            open_dim_seq2 = self.task_prompt(open_dim_seq2)
            # composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_x = gcn(open_dim_seq2, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_x, emb_t], dim=2)
        elif self.ablation_choice[-4:-2] == 'o2': 
            open_dim_seq1 = self.task_prompt(open_dim_seq1)
            open_dim_seq2 = self.task_prompt2(open_dim_seq2)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(open_dim_seq2, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=2)
        elif self.ablation_choice[-4:-2] == 'dc': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            return gcn(composed_dim_seq, adj, sparse,  None)
        elif self.ablation_choice[-4:-2] == 'dx': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            
            return gcn(open_dim_seq2, adj, sparse,  None)
        elif self.ablation_choice[-4:-2] == 'dt': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            seq_oc = (open_dim_seq1 + composed_dim_seq) 
            emb = gcn(seq_oc, adj, sparse,  None)
            return emb 
        elif self.ablation_choice[-4:-2] == 'xt': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            seq_oc = (open_dim_seq2 + composed_dim_seq) 
            emb = gcn(seq_oc, adj, sparse,  None)
            return emb 
        elif self.ablation_choice[-4:-2] == 'ct': 
            # composed_dim_seq = self.open_fea(composed_dim_seq)

            emb_o = gcn(open_dim_seq2, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        elif self.ablation_choice[-4:-2] == 'ot': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(open_dim_seq2, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        elif self.ablation_choice[-4:-2] == 'jj': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        elif self.ablation_choice[-4:-2] == 'jx': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1)
            composed_dim_seq = self.task_prompt(composed_dim_seq)
            # composed_dim_seq = self.open_prompt_fea(composed_dim_seq)

            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=2)
        elif self.ablation_choice[-4:-2] == 'tt': 
            emb_o1 = gcn(open_dim_seq1, adj, sparse,  None)
            emb_o2 = gcn(open_dim_seq2, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            # return emb_o1 + emb_o2 + emb_t
            return torch.cat([emb_o1, emb_o2, emb_t], dim=1)
        elif self.ablation_choice[-4:-2] == 'oc': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_oc = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_oc
        elif self.ablation_choice[-4:-2] == 'co': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            emb_o = gcn(open_dim_seq2, adj, sparse,  None)
            emb_oc = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_oc
        elif self.ablation_choice[-4:-2] == 'ot': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(open_dim_seq2, adj, sparse,  None)
            return emb_o + emb_t
        if self.ablation_choice[-2:] == 'no': 
            embed_fea = gcn(seq, adj, sparse, None)
            return embed_fea
    def get_emb(self, seq): 
        if self.de_input == 'x': 
            sample = self.sampler(seq)
        elif self.de_input == 'ax': 
            sample = self.sampler(self.agg_feat)

        # sample = self.domain_token(sample.T).T
        # dimension_sig_open = self.open_prompt_dim(sample)

        # open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open)
        # open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(self.agg_feat, dimension_sig_open)
        # open_dim_seq1 = self.task_prompt2(open_dim_seq1)
        # composed_dim_seq = self.task_prompt(composed_dim_seq)
        composed_dim_seq = self.composedprompt_dim(sample, seq, 'ww')
        
        return self.sampler(composed_dim_seq)
        # return self.sampler(self.task_prompt2(open_dim_seq1 + open_dim_seq2))
        # return self.sampler(open_dim_seq1 + open_dim_seq2)
        return open_dim_seq1 + open_dim_seq2
    def get_emb_2(self, seq, gcn, adj, sparse): 
        if self.de_input == 'x': 
            sample = self.sampler(seq)
        elif self.de_input == 'ax': 
            sample = self.sampler(self.agg_feat)

        # sample = self.domain_token(sample.T).T
        dimension_sig_open = self.open_prompt_dim(sample)

        open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open)
        open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(self.agg_feat, dimension_sig_open)
        composed_dim_seq = self.composedprompt_dim(sample, seq, 'ww')

        if self.ablation_choice[-4:-2] == 'do': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1)
            emb = gcn(open_dim_seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'd2': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq2)
            emb = gcn(open_dim_seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'dq': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            emb = gcn(open_dim_seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'dz': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            emb = emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'dn': 
            open_dim_seq1 = self.task_prompt2(open_dim_seq1 + open_dim_seq2)
            # composed_dim_seq = self.task_prompt(composed_dim_seq)
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            emb = emb_o + emb_t

        return self.sampler(emb.squeeze(0))
class downpromptBarycenter(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  dim_pretext_weight, 
                  combines, type_='mul', ablation = 'all', sample_size = 182, 
                  if_rand=False, gamma=0.5, n_mlp_layer=1, 
                  sampling='random', agg_feat=None, shared=False, shared_token=None, de_input='x', basis_matrix=None, ):
        super(downpromptBarycenter, self).__init__()

        self.num_pretrain_datasets = len(dim_pretext_weight)
        

        self.downstreamPrompt = downstreampromptBaryceter(feature_dim, ft_in, num_layers_num, 
            dim_pretext_weight, 
            combines, type_, ablation, sample_size, if_rand, gamma,
            n_mlp_layer, sampling, agg_feat, de_input, shared, shared_token, basis_matrix=basis_matrix)

        self.sample_size = sample_size
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)
        self.agg_feat = agg_feat

        self.gamma = gamma 
    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0,batch=None, pseudo_idx=None, weights=None):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   # [nodes, emb_dim]
        # print(f'embeds: {embeds.shape}')

        if weights is not None: # pseudo labeled node 사용 
            if batch != None: # graph classification 
                rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
            else: # node classification 
                # rawret = embeds[idx] # [idx]: [num_classes], rawret: [num_class, emb_dim]
                rawret = embeds[pseudo_idx] # [idx]: [num_classes], rawret: [num_class, emb_dim]
            num =  rawret.shape[0] 
            if train == 1:
                self.ave = weighted_averageemb(labels, rawret, weights)
            rawret = embeds[idx] 
            num =  rawret.shape[0] 
        else: 
            if batch != None: # graph classification 
                rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
            else: # node classification 
                rawret = embeds[idx] # [idx]: [num_classes], rawret: [num_class, emb_dim]
            num =  rawret.shape[0] 

            if train == 1:
                self.ave = averageemb(labels=labels, rawret=rawret) # prototype , [num_class, emb_dim]
        
        # 코사인 유사도 기반 분류 
        rawret = torch.cat((rawret, self.ave.to(rawret.device)), dim=0)  # shape: (B+C, D) -> Query node B 개와 C개의 prototype  비교 
        rawret = F.normalize(rawret, dim=1)            # 모든 row 벡터 L2 정규화
        rawret = rawret @ rawret.T                     # (N+C) x (N+C) 유사도 행렬 (cosine similarity)

        ret = rawret[:num,num:] # 각 노드 - 프로토타입 간의 유사도만 선택 
        ret = F.softmax(self.gamma * ret, dim=1) 
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def de_loss(self): 
        return self.downstreamPrompt.open_prompt_dim.dimensional_loss()

def weighted_averageemb(labels, rawret, weights):
    """
    labels: [N] 각 샘플의 class id
    rawret: [N, d] 샘플 embedding
    weights: [N] 각 샘플 weight
    """
    # (N, d) → (N, d) * (N, 1)
    weighted_emb = rawret * weights.unsqueeze(1)
    
    # 클래스별 weight 합
    weight_sum = torch_scatter.scatter(weights, labels, dim=0, reduce='sum').unsqueeze(1) + 1e-9
    
    # 클래스별 weighted sum
    proto_sum = torch_scatter.scatter(weighted_emb, labels, dim=0, reduce='sum')
    
    return proto_sum / weight_sum  # [num_classes, d]

class downstreampromptFilterbankFUG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, high_dimension_pretext, low_dimension_pretext, identity_dimension_pretext,
                combines, type_ = 'mul', ablation = 'all', sample_size = 182, if_rand=False, gamma=0.5, basis_matrix=None, 
                n_mlp_layer=1, sampling='random', agg_feat=None, shared=False, shared_token=None):
        super(downstreampromptFilterbankFUG, self).__init__()

        self.sample_size = sample_size
        self.if_rand = if_rand 

        self.iden_task_prompt = composedBasisNode(feature_dim, len(basis_matrix[0]), basis_matrix[0])
        self.low_task_prompt = composedBasisNode(feature_dim, len(basis_matrix[0]), basis_matrix[1])
        self.high_task_prompt = composedBasisNode(feature_dim, len(basis_matrix[0]), basis_matrix[2])

        self.shared_low_dimension_encoder = DimensionNN_FUG(sample_size, sample_size//2, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.shared_low_dimension_encoder.load_state_dict(low_dimension_pretext.state_dict())
        for param in self.shared_low_dimension_encoder.parameters():
            param.requires_grad = False

        self.low_dimension_encoder =  DimensionNN_FUG(sample_size, sample_size//2, feature_dim, nn.PReLU, layers=n_mlp_layer) 
        self.high_dimension_encoder = DimensionNN_FUG(sample_size, sample_size//2, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.identity_dimension_encoder = DimensionNN_FUG(sample_size, sample_size//2, feature_dim, nn.PReLU, layers=n_mlp_layer) 
        
        self.weight = nn.Parameter(torch.zeros(3))
        self.open_prompt_fea = textprompt(feature_dim, type_='add')

        self.sampler = Sampler(sample_size=sample_size, if_rand=if_rand, sampling=sampling)
        self.sampling = sampling 

        self.shared = shared 
        self.shared_token = shared_token
        

    def forward(self, seq, gcn, adj, sparse):
        low_seq = get_low_pass_filter(seq, adj) @ seq 
        high_seq = get_high_pass_filter(seq, adj) @ seq 

        # identity 
        identity_sample = self.sampler(seq, adj)
        low_sample = self.sampler(low_seq, adj)
        high_sample = self.sampler(high_seq, adj)

        identity_basis = self.identity_dimension_encoder(identity_sample)
        low_basis = self.low_dimension_encoder(low_sample)
        # low_basis = self.shared_low_dimension_encoder(low_sample)
        high_basis = self.high_dimension_encoder(high_sample)
        # high_basis = identity_basis - low_basis 
        
        # X 에 곱해주기 
        # H_identity = F.normalize(seq @ identity_basis)
        # H_low = F.normalize(seq @ low_basis)
        # H_high = F.normalize(seq @ high_basis)

        # filtered X에 곱해주기 
        H_identity = F.normalize(seq @ identity_basis)
        H_low = F.normalize(low_seq @ low_basis)
        H_high = F.normalize(high_seq @ high_basis)

        
        # H_identity = self.iden_task_prompt(H_identity)
        # H_low = self.low_task_prompt(H_low)
        # H_high = self.high_task_prompt(H_high)

        if self.shared: 
            H_identity = F.relu(H_identity) # activation 
            H_low = F.relu(H_low) # activation 
            H_high = F.relu(H_high) # activation 

            H_identity = self.shared_token * H_identity
            H_low = self.shared_token * H_low
            H_high = self.shared_token * H_high

        # H_identity = gcn(H_identity, adj, sparse, None)
        # H_low = gcn(H_low, adj, sparse, None)
        # H_high = gcn(H_high, adj, sparse, None)

        # 가중합 
        w = F.softmax(self.weight, dim=0)
        seq = w[0]*H_low + w[1]*H_high + w[2]*H_identity
        # return seq 
    
        seq = self.open_prompt_fea(seq)

        return gcn(seq, adj, sparse, None)
    
    def get_emb(self, seq, adj): 
        low_seq = get_low_pass_filter(seq, adj) @ seq 
        high_seq = get_high_pass_filter(seq, adj) @ seq 

        # identity 
        identity_sample = self.sampler(seq, adj)
        low_sample = self.sampler(low_seq, adj)
        high_sample = self.sampler(high_seq, adj)

        identity_basis = self.identity_dimension_encoder(identity_sample)
        # low_basis = self.low_dimension_encoder(low_sample)
        low_basis = self.shared_low_dimension_encoder(low_sample)
        high_basis = self.high_dimension_encoder(high_sample)
        # high_basis = identity_basis - low_basis 
        
        # X 에 곱해주기 
        H_identity = F.normalize(seq @ identity_basis)
        H_low = F.normalize(seq @ low_basis)
        H_high = F.normalize(seq @ high_basis)

        # filtered X에 곱해주기 
        # H_identity = F.normalize(seq @ identity_basis)
        # H_low = F.normalize(low_seq @ low_basis)
        # H_high = F.normalize(high_seq @ high_basis)
  
        if self.shared: 
            H_identity = F.relu(H_identity) # activation 
            H_low = F.relu(H_low) # activation 
            H_high = F.relu(H_high) # activation 

            H_identity = self.shared_token * H_identity
            H_low = self.shared_token * H_low
            H_high = self.shared_token * H_high


        # 가중합 
        w = F.softmax(self.weight, dim=0)
        seq = w[0]*H_low + w[1]*H_high + w[2]*H_identity
        return seq

class downstreampromptFilterFUG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, high_dimension_pretext, low_dimension_pretext, balance_weights,
                combines, type_ = 'mul', ablation = 'all', sample_size = 182, if_rand=False, gamma=0.5, basis_matrix=None, 
                n_mlp_layer=1, sampling='random', agg_feat=None, shared=False, shared_token=None):
        super(downstreampromptFilterFUG, self).__init__()

        self.sample_size = sample_size
        self.if_rand = if_rand 

        # high_basis = [tensor.mean(dim=0) for tensor in high_dimension_pretext]
        self.task_prompt = composedBasisNode(feature_dim//2, len(high_dimension_pretext), basis_matrix)
        self.task_prompt2 = composedBasisNode(feature_dim//2, len(high_dimension_pretext), basis_matrix)

        basis = [torch.cat([low_dimension_pretext.out.detach().mean(dim=0), b]) for b in basis_matrix]
        self.task_prompt3 = composedBasisNode(feature_dim, len(high_dimension_pretext), basis)
        self.task_prompt4 = composedBasisNode(feature_dim, len(high_dimension_pretext), basis)

        # self.domain_token_layers = nn.ModuleList([textprompt(feature_dim//2, type_='mul') for _ in range(num_pretrain_dataset_num)])
        # balance weights = low 통과전 붙은 애들 
        self.low_domain_token = textprompt(sample_size, type_='mul')
        self.high_shared_token = shared_token

        self.composedprompt_dim = composedFUG(len(high_dimension_pretext), high_dimension_pretext, balance_weights)
        
        # self.low_dim = DimensionNN_FUG(sample_size, sample_size, feature_dim//2, nn.PReLU, layers=n_mlp_layer)
        self.low_dim = DimensionNN_FUG(sample_size, sample_size//2, feature_dim//2, nn.PReLU, layers=n_mlp_layer)
        self.low_dim.load_state_dict(low_dimension_pretext.state_dict())

        # requires_grad 끊기
        for param in self.low_dim.parameters():
            param.requires_grad = False

        self.open_high_dim = DimensionNN_FUG(sample_size, sample_size//2, feature_dim//2, nn.PReLU, layers=n_mlp_layer)
        # self.open_high_dim = DimensionNN_FUG(sample_size, sample_size, feature_dim//2, nn.PReLU, layers=n_mlp_layer)

        self.sampler = Sampler(sample_size=sample_size, if_rand=if_rand, sampling=sampling)
        self.sampling = sampling 
        
        self.open_balance_token = balanceprompt(feature_dim, type_)
        self.open_prompt_fea = textprompt(feature_dim, type_='add')

        self.open_prompt_fea1 = textprompt(feature_dim//2, type_='add')
        self.open_prompt_fea2 = textprompt(feature_dim//2, type_='add')

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.gamma = gamma 
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation
        self.agg_feat = agg_feat

        self.shared = shared 
        self.shared_token = shared_token

    def forward(self, seq, gcn, adj, sparse):

        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
        
        # dimension reduction 먼저 
        
        # low_freq_indices, high_freq_indices = split_features_by_frequency(seq, adj)
        # sample = self.sampler(seq, adj)

        low_freq_indices, high_freq_indices = split_features_by_frequency(self.agg_feat, adj)
        sample = self.sampler(self.agg_feat, adj)

        # Token ! 
        low_sample = self.low_domain_token(sample[:, low_freq_indices].T).T
        # low_sample = sample[:, low_freq_indices]

        t_low = self.low_dim(low_sample)

        H_low_X = self.low_dim.feature_sig_propagate(seq[:, low_freq_indices], t_low)
        H_low = self.low_dim.feature_sig_propagate(self.agg_feat[:, low_freq_indices], t_low)

        t_high = self.open_high_dim(sample[:, high_freq_indices])

        H_high_X = self.open_high_dim.feature_sig_propagate(seq[:, high_freq_indices], t_high)
        H_high = self.open_high_dim.feature_sig_propagate(self.agg_feat[:, high_freq_indices], t_high)
        
        

        H_composed_high = self.composedprompt_dim(sample[:, high_freq_indices], self.agg_feat[:, high_freq_indices], 'wn')
        H_composed_high_X = self.composedprompt_dim(sample[:, high_freq_indices], seq[:, high_freq_indices], 'wn')

        # token ! 
        # H_high = self.high_shared_token * H_high
        # H_composed_high = self.high_shared_token * H_composed_high

        seq1 = torch.cat([H_low, H_high], dim=1)
        seq2 = torch.cat([H_low, H_composed_high], dim=1)
        seq3 = torch.cat([H_low, (H_high+H_composed_high)], dim=1)

        seq4 = torch.cat([H_low_X, H_high_X], dim=1)
        seq5 = torch.cat([H_low_X, H_composed_high_X], dim=1)
        seq6 = torch.cat([H_low_X, (H_high_X+H_composed_high_X)], dim=1)

        if self.shared: 
            seq1 = self.shared_token * seq1
            seq2 = self.shared_token * seq2
            seq3 = self.shared_token * seq3

            seq4 = self.shared_token * seq4
            seq5 = self.shared_token * seq5
            seq6 = self.shared_token * seq6

        # balance token 
        # seq1 = self.open_balance_token(seq1)
        # seq2 = self.open_balance_token(seq2)
        # seq3 = self.open_balance_token(seq3)
        # seq4 = self.open_balance_token(seq4)
        # seq5 = self.open_balance_token(seq5)


        # seq1 = self.open_prompt_fea(seq1)
        # seq2 = self.open_prompt_fea(seq2)
        # seq3 = self.open_prompt_fea(seq3)

        if self.ablation_choice[-4:-2] == 'do': 
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'dc':
            return gcn(seq2, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'dt': 
            return gcn(seq3, adj, sparse,  None)
        elif self.ablation_choice[-4:-2] == 'ct': 
            emb_o = gcn(seq1, adj, sparse,  None)
            emb_t = gcn(seq4, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        elif self.ablation_choice[-4:-2] == 'tt': 

            seq1 = torch.cat([H_low, H_high], dim=1)
            seq4 = torch.cat([H_low_X, H_high_X], dim=1)
            
            seq1 = self.open_prompt_fea(seq1)
            seq4 = self.open_prompt_fea(seq4)
            emb_o = gcn(seq1, adj, sparse,  None)
            emb_t = gcn(seq4, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        elif self.ablation_choice[-4:-2] == 'oc': 
            emb1 = gcn(seq1, adj, sparse,  None)
            emb2 = gcn(seq2, adj, sparse,  None)
            return emb1 + emb2
        elif self.ablation_choice[-4:-2] == 'll': 
            seq_ll = torch.cat([H_low, H_low], dim=1)
            return gcn(seq_oo, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'oo': 
            seq_oo = torch.cat([H_high, H_high], dim=1)
            return gcn(seq_oo, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'cc': 
            seq_cc = torch.cat([H_composed_high, H_composed_high], dim=1)
            return gcn(seq_cc, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'xo':
            return gcn(seq4, adj, sparse, None) 
        elif self.ablation_choice[-4:-2] == 'xc':
            return gcn(seq5, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'x6': 
            return gcn(seq5, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'xx': 
            return gcn(seq4+seq5, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'xp': 
            return gcn(seq4+seq1, adj, sparse, None)
        # [1, k] 짜리 프롬프트 
        if self.ablation_choice[-4:-2] == 'tp':
            seq3 = self.open_prompt_fea(seq3)
            return gcn(seq3, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'dp':
            seq1 = self.open_prompt_fea(seq1)
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'dd':
            seq2 = self.open_prompt_fea(seq2)
            return gcn(seq2, adj, sparse, None)
        # [k//2]짜리 프롬프트 
        elif self.ablation_choice[-4:-2] == 'lm': 
            # H_low = self.open_prompt_fea1(H_low)
            H_high = self.open_prompt_fea2(H_high)
            seq1 = torch.cat([H_low, H_high], dim=1)
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'kk': 
            seq6 = torch.cat([H_low+H_high, H_low+H_composed_high], dim=1)
            return gcn(seq6, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'op': 
            H_low = self.open_prompt_fea1(H_low)
            H_high = self.open_prompt_fea2(H_high)
            seq1 = torch.cat([H_low, H_high], dim=1)
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'sb': 
            # H_low = self.open_prompt_fea1(H_low)
            # H_high = self.open_prompt_fea2(H_high)
            H_low = self.task_prompt(H_low)
            H_high = self.task_prompt2(H_high)
            seq1 = torch.cat([H_low, H_high], dim=1)
            # print(f'seq: {seq1.shape}')
            # seq1 = self.task_prompt(seq1)
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 's2': 
            # H_low = self.open_prompt_fea1(H_low)
            # H_high = self.open_prompt_fea2(H_high)
            seq1 = self.task_prompt3(seq3)
            seq4 = self.task_prompt4(seq6)
            emb_1 = gcn(seq1, adj, sparse, None)
            emb_2 = gcn(seq2, adj, sparse, None)
            return emb_1 + emb_2 
        
        elif self.ablation_choice[-4:-2] == 'qw': 
            H_low = self.open_prompt_fea1(H_low_X)
            H_high = self.open_prompt_fea2(H_high_X)
            seq1 = torch.cat([H_low, H_high], dim=1)
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'cp': 
            H_low = self.open_prompt_fea1(H_low)
            seq1 = torch.cat([H_low, H_high+H_composed_high_X], dim=1)
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'pp': 
            H_low = self.open_prompt_fea1(H_low)
            H_cp = self.open_prompt_fea2(H_high+H_composed_high_X)
            seq1 = torch.cat([H_low, H_cp], dim=1)
            return gcn(seq1, adj, sparse, None)
        elif self.ablation_choice[-4:-2] == 'ff': 
            H_low_p = self.open_prompt_fea1(H_low)
            H_high_p = self.open_prompt_fea2(H_high)
            seq1 = torch.cat([H_low, H_high], dim=1)
            seq2 = torch.cat([H_low_X, H_high_X], dim=1)
            # return gcn(seq1, adj, sparse, None) + gcn(seq2, adj, sparse, None)
            return torch.cat([gcn(seq1, adj, sparse, None), gcn(seq2, adj, sparse, None)], dim=1)
        
    def get_emb(self, seq, adj): 
        low_freq_indices, high_freq_indices = split_features_by_frequency(self.agg_feat, adj)
        sample = self.sampler(self.agg_feat, adj)

        # Token ! 
        low_sample = self.low_domain_token(sample[:, low_freq_indices].T).T
        # low_sample = sample[:, low_freq_indices]

        t_low = self.low_dim(low_sample)

        # H_low_X = self.low_dim.feature_sig_propagate(seq[:, low_freq_indices], t_low)
        H_low = self.low_dim.feature_sig_propagate(self.agg_feat[:, low_freq_indices], t_low)

        # t_high = self.open_high_dim(sample[:, high_freq_indices])

        # H_high_X = self.open_high_dim.feature_sig_propagate(seq[:, high_freq_indices], t_high)
        # H_high = self.open_high_dim.feature_sig_propagate(self.agg_feat[:, high_freq_indices], t_high)
        
        return H_low
    
class downstreampromptSharedFUG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, dim_pretexts, shared_dimension_pretext, balance_weights,
                combines, type_ = 'mul', ablation = 'all', sample_size = 182, if_rand=False, gamma=0.5, basis_matrix=None, n_mlp_layer=1, sampling='random', agg_feat=None):
        super(downstreampromptSharedFUG, self).__init__()

        self.sample_size = sample_size
        self.if_rand = if_rand 

        self.composedprompt_dim = composedFUG(len(dim_pretexts), dim_pretexts, balance_weights)
        # self.composedprompt_dim = composedFUG(len(dim_pretexts), dim_pretexts, basis_matrix)

        
        # self.shared_dim = DimensionNN_FUG(sample_size, sample_size, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.shared_dim = DimensionNN_FUG(sample_size, sample_size, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.shared_dim.load_state_dict(shared_dimension_pretext.state_dict())

        # requires_grad 유지 (원하는 경우 여기서 False로 설정 가능)
        for param in self.shared_dim.parameters():
            param.requires_grad = False

        
        # self.open_prompt_dim = DimensionNN_FUG(sample_size*2, sample_size, feature_dim, nn.PReLU, layers=n_mlp_layer)
        # self.open_prompt_dim = DimensionNN_FUG(sample_size, sample_size//2, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.open_prompt_dim = DimensionNN_FUG(sample_size, sample_size, feature_dim*2, nn.PReLU, layers=n_mlp_layer)
        self.sampler = Sampler(sample_size=sample_size, if_rand=if_rand, sampling=sampling)
        self.sampling = sampling 
        
        self.open_balance_token = balanceprompt(sample_size, type_)
        

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.gamma = gamma 
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

        self.agg_feat = agg_feat

    def forward(self, seq, gcn, adj, sparse):

        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
        
        # dimension reduction 먼저 
        sample_AX = self.sampler(self.agg_feat, adj)
        sample_X = self.sampler(seq, adj)

        sample = torch.cat([sample_X, sample_AX], dim=0)
        sample_b = self.open_balance_token(sample.T).T

        # dimension_sig_open = self.open_prompt_dim(sample_b)

        dimension_sig_open = self.open_prompt_dim(sample_AX)

        open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open)
        open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(self.agg_feat, dimension_sig_open)

        composed_dim_seq = self.composedprompt_dim(sample_AX, seq, 'wn')


        # # shared 
        shared_dim_sig_open = self.shared_dim(sample_AX)
        shared_dim_open = self.shared_dim.feature_sig_propagate(seq, shared_dim_sig_open)

        sample = self.sampler(open_dim_seq1, adj)
        shared_dim_sig_open = self.shared_dim(sample)
        open_dim_seq1 = self.shared_dim.feature_sig_propagate(open_dim_seq1, shared_dim_sig_open)
        open_dim_seq2 = self.shared_dim.feature_sig_propagate(open_dim_seq2, shared_dim_sig_open)

        
        # Xc -> shared 
        sample = self.sampler(composed_dim_seq, adj)
        shared_dim_sig_open = self.shared_dim(sample)
        composed_dim_seq = self.shared_dim.feature_sig_propagate(composed_dim_seq, shared_dim_sig_open)


        if self.ablation_choice[-4:-2] == 'do': 
            return gcn(open_dim_seq1, adj, sparse, None)
        
        elif self.ablation_choice[-4:-2] == 'xt':
            return gcn(open_dim_seq2, adj, sparse, None)

        elif self.ablation_choice[-4:-2] == 'dc': 
            return gcn(composed_dim_seq, adj, sparse,  None)
        
        elif self.ablation_choice[-4:-2] == 'dt': 
            seq_oc = (open_dim_seq1 + composed_dim_seq) 
            emb = gcn(seq_oc, adj, sparse,  None)
            return emb 
        elif self.ablation_choice[-4:-2] == 'sh': 
            return gcn(shared_dim_open, adj, sparse, None)
        
        elif self.ablation_choice[-4:-2] == 'oc': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_oc = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_oc

        elif self.ablation_choice[-4:-2] == 'ot': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(open_dim_seq2, adj, sparse,  None)
            return emb_o + emb_t
        elif self.ablation_choice[-4:-2] == 'to': 
            seq_oc = (open_dim_seq1 + open_dim_seq2) 
            emb = gcn(seq_oc, adj, sparse,  None)
            return emb 
        elif self.ablation_choice[-4:-2] == 'tc': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_oc = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_oc
        elif self.ablation_choice[-4:-2] == 'al': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(open_dim_seq2, adj, sparse,  None)
            emb_oc = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_t + emb_oc
        elif self.ablation_choice[-4:-2] == 'as': 
            seq = open_dim_seq1 + open_dim_seq2 + composed_dim_seq
            return gcn(seq, adj, sparse,  None)

    
class downpromptSharedFUG(nn.Module):
    def __init__(self, model_type, ft_in, nb_classes, feature_dim, num_layers_num, 
                  dim_pretext_weight, shared_dimension_encoder, balance_weights,
                  combines, type_='mul', ablation = 'all', sample_size = 182, 
                  if_rand=False, gamma=0.5, basis_matrix=None, n_mlp_layer=1, sampling='random', agg_feat=None, shared=False, shared_token=None):
        super(downpromptSharedFUG, self).__init__()

        self.num_pretrain_datasets = len(balance_weights)
        
        self.model_type = model_type
        if model_type == 'sharedFUG': 
            self.downstreamPrompt = downstreampromptSharedFUG(feature_dim, ft_in, num_layers_num, 
                dim_pretext_weight, shared_dimension_encoder, balance_weights, 
                combines, type_, ablation, sample_size, if_rand, gamma, basis_matrix, 
                n_mlp_layer, sampling, agg_feat)
        elif model_type == 'filterFUG': 
            self.downstreamPrompt = downstreampromptFilterFUG(feature_dim, ft_in, num_layers_num, 
                dim_pretext_weight, shared_dimension_encoder, balance_weights, 
                combines, type_, ablation, sample_size, if_rand, gamma, basis_matrix, 
                n_mlp_layer, sampling, agg_feat, shared, shared_token)
            
        elif model_type == 'filterbank': 
            self.downstreamPrompt = downstreampromptFilterbankFUG(feature_dim, ft_in, num_layers_num, 
                dim_pretext_weight, shared_dimension_encoder, balance_weights, 
                combines, type_, ablation, sample_size, if_rand, gamma, basis_matrix, 
                n_mlp_layer, sampling, agg_feat, shared, shared_token)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)
        self.agg_feat = agg_feat

    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0,batch=None):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        
        if batch != None: # graph classification 
            rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
        else: # node classification 
            rawret = embeds[idx]
        num =  rawret.shape[0]

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype 
        
        # 코사인 유사도 기반 분류 
        rawret = torch.cat((rawret, self.ave.to(rawret.device)), dim=0)  # shape: (B+C, D) -> Query node B 개와 C개의 prototype  비교 
        rawret = F.normalize(rawret, dim=1)            # 모든 row 벡터 L2 정규화
        rawret = rawret @ rawret.T                     # (N+K) x (N+K) 유사도 행렬 (cosine similarity)

        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def de_loss(self): 
        if self.model_type  == 'sharedFUG':
            return self.downstreamPrompt.open_prompt_dim.dimensional_loss()
        elif self.model_type  == 'filterFUG': 
            return self.downstreamPrompt.open_high_dim.dimensional_loss()
        elif self.model_type == 'filterbank':
            return self.downstreamPrompt.identity_dimension_encoder.dimensional_loss() \
                    + self.downstreamPrompt.low_dimension_encoder.dimensional_loss() \
                    + self.downstreamPrompt.high_dimension_encoder.dimensional_loss()

class downstreampromptFUG(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, dim_pretexts, fea_pretext_weights, balance_weights,
                combines, type_ = 'mul', ablation = 'all', sample_size = 182, if_rand=False, gamma=0.5, basis_matrix=None, n_mlp_layer=1, sampling='random', agg_feat=None, shared=False, shared_token=None):
        super(downstreampromptFUG, self).__init__()

        self.sample_size = sample_size
        self.if_rand = if_rand 

        self.composedprompt_dim = composedFUG(len(dim_pretexts), dim_pretexts, balance_weights)
        # self.composedprompt_dim = composedFUG(len(dim_pretexts), dim_pretexts, basis_matrix)

        # self.open_prompt_dim = DimensionNN_FUG(sample_size*3, sample_size, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.open_prompt_dim = DimensionNN_FUG(sample_size, sample_size//2, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.sampler = Sampler(sample_size=sample_size, if_rand=if_rand, sampling=sampling)
        self.sampling = sampling 


        # self.local_composed_fea = composedBasisNode(feature_dim, len(dim_pretexts), basis_matrix)
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)

        
        self.open_prompt_fea = textprompt(feature_dim, type_='add')
        
        # self.open_balance_token = textprompt(sample_size*2, type_='mul')
        self.open_balance_token = balanceprompt(sample_size*3, type_)

        self.shared = shared
        self.shared_token = shared_token

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.gamma = gamma 
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

        self.open_prompt_str = nn.ModuleList()
        # if str_pretext_weights is not None: 
        #     for weight in str_pretext_weights[0]:
        #         in_features = weight.size(1)
        #         new_layer = textprompt(in_features, type_)
        #         self.open_prompt_str.append(new_layer)
        #     self.composedprompt_str = nn.ModuleList([
        #         composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
        #         for i in range(num_layers_num)
        #     ])

        self.agg_feat = agg_feat

    def forward(self, seq, gcn, adj, sparse):

        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
        
        exp_c = self.ablation_choice[-2:]
        exp_shared = self.ablation_choice[-3]
        exp_o = self.ablation_choice[-5:-3]

        # dimension reduction 먼저 
        sample_AX = self.sampler(self.agg_feat, adj)
        sample_X = self.sampler(seq, adj)
        # sample = torch.cat([sample_X, sample_AX], dim=0)
        # sample_b = self.open_balance_token(sample.T).T

        dimension_sig_open = self.open_prompt_dim(sample_AX)

        open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open)
        open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(self.agg_feat, dimension_sig_open)


        # high_X = get_high_pass_filter(seq, adj) @ seq
        # low_X = get_low_pass_filter(seq, adj) @ seq

        # sample_I = self.sampler(seq, adj)
        # sample_L = self.sampler(low_X, adj)
        # sample_H = self.sampler(high_X, adj)

        # sample = torch.cat([sample_I, sample_L, sample_H], dim=0)
        # sample_b = self.open_balance_token(sample.T).T
        
        # dimension_sig_open = self.open_prompt_dim(sample_b)
        
        # open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open)
        # open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(low_X, dimension_sig_open)
        # open_dim_seq3 = self.open_prompt_dim.feature_sig_propagate(high_X, dimension_sig_open)

        # open_dim_seq1 *= self.open_balance_token.weight[0,0]
        # open_dim_seq2 *= self.open_balance_token.weight[0,1]
        # open_dim_seq3 *= self.open_balance_token.weight[0,2]

        # open_dim_seq1 = open_dim_seq1 + open_dim_seq2 + open_dim_seq3

        # dimension_sig_open = self.open_prompt_dim(sample_b)

        

        if self.shared: 
            open_dim_seq1 = self.shared_token * open_dim_seq1
            open_dim_seq2 = self.shared_token * open_dim_seq2
            dimension_sig_open = self.shared_token * dimension_sig_open

        if exp_c[-1] == 'n': # target's balacne token 
            composed_dim_seq = self.composedprompt_dim(sample_b, seq, exp_c)
            if exp_shared == 't': # True
                composed_dim_seq = composed_dim_seq * self.shared_token
        elif exp_c[-1] == 'b': # source balance token 
            composed_dim_seq = self.composedprompt_dim(sample, seq, exp_c)
            if exp_shared == 't': # True
                composed_dim_seq = composed_dim_seq * self.shared_token
        elif exp_c == 'no': 
            # composed_dim_seq = self.composedprompt_dim(sample_X, seq)
            composed_dim_seq = self.composedprompt_dim(sample_AX, seq, 'ww')
            # composed_dim_seq = self.composedprompt_dim(sample, seq)

        if exp_o == 'do': # Xo only  
            return gcn(open_dim_seq1, adj, sparse, None)
        elif exp_o == 'xt': # Xt only 
            return gcn(open_dim_seq2, adj, sparse, None)
        elif exp_o == 'dc': 
            return gcn(composed_dim_seq, adj, sparse, None)
        elif exp_o == 'ds': 
            open_dim_seq1 = open_dim_seq1 * self.shared_token
            print(open_dim_seq1.shape, self.shared_token.shape)
            return gcn(open_dim_seq1, adj, sparse, None)
        elif exp_o == 'to':  # Xo + Xt
            return gcn(open_dim_seq1 + open_dim_seq2, adj, sparse, None)
        elif exp_o == 'ot': 
            return gcn(open_dim_seq1, adj, sparse, None) + gcn(open_dim_seq2, adj, sparse, None) 
        elif exp_o == 'oc': 
            # composed_dim_seq *= self.shared_token
            emb1 = gcn(open_dim_seq1, adj, sparse, None)
            emb2 = gcn(composed_dim_seq, adj, sparse, None)
            return emb1 + emb2 
        elif exp_o == 'sc': 
            emb1 = gcn(open_dim_seq1 + open_dim_seq2, adj, sparse, None)
            emb2 = gcn(composed_dim_seq, adj, sparse, None)
            return emb1 + emb2 
        elif exp_o == 'al': # All 
            emb1 = gcn(open_dim_seq1, adj, sparse, None)
            emb2 = gcn(open_dim_seq2, adj, sparse, None)
            emb3 = gcn(composed_dim_seq, adj, sparse, None)
            return emb1 + emb2 + emb3 
        elif exp_o == 'as': 
            open_dim_seq2 *= self.shared_token
            emb1 = gcn(open_dim_seq1, adj, sparse, None)
            emb2 = gcn(open_dim_seq2, adj, sparse, None)
            emb3 = gcn(composed_dim_seq, adj, sparse, None)
            return emb1 + emb2 + emb3
        

        if self.ablation_choice  == 'bal01': 
            # seq1 = self.open_balance_token.weight0(seq)
            # seq2 = self.open_balance_token.weight1(self.agg_feat)

            open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open) 
            open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(self.agg_feat, dimension_sig_open) 

            # open_dim_seq1 = self.open_prompt_dim.feature_sig_propagate(seq, dimension_sig_open) * a
            # open_dim_seq2 = self.open_prompt_dim.feature_sig_propagate(self.agg_feat, dimension_sig_open) * b
            
            # open_dim_seq1 = open_dim_seq1 * self.shared_token
            # open_dim_seq2 = open_dim_seq2 * self.shared_token # shared token

            # open_dim_seq1 = self.open_balance_token.weight0(open_dim_seq1)
            # open_dim_seq2 = self.open_balance_token.weight1(open_dim_seq2)
            # composed_dim_seq = composed_dim_seq * self.shared_token
            # return gcn(open_dim_seq1 + open_dim_seq2, adj, sparse, None) + gcn(composed_dim_seq, adj, sparse, None)
            return gcn(open_dim_seq1, adj, sparse, None) +  gcn(open_dim_seq2, adj, sparse, None) + gcn(composed_dim_seq, adj, sparse, None)
        
        if self.ablation_choice[-4:-2] == 'do': 
            # open_dim_seq1 = open_dim_seq1 * self.shared_token
            
            return gcn(open_dim_seq1, adj, sparse, None)
        
        elif self.ablation_choice[-4:-2] == 'dc': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            return gcn(composed_dim_seq, adj, sparse,  None)
        
        elif self.ablation_choice[-4:-2] == 'dt': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            return  gcn(open_dim_seq2, adj, sparse, None)
            seq_oc = (open_dim_seq + composed_dim_seq) 
            emb = gcn(seq_oc, adj, sparse,  None)
            return emb 

        elif self.ablation_choice[-4:-2] == 'oc': 
            # composed_dim_seq = composed_dim_seq * self.shared_token
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_oc = gcn(composed_dim_seq, adj, sparse,  None)
            return emb_o + emb_oc
        elif self.ablation_choice[-4:-2] == 'ot': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(open_dim_seq2, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        
        elif self.ablation_choice[-4:-2] == 'ct': 
            emb_o = gcn(open_dim_seq2, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        
        elif self.ablation_choice[-4:-2] == 'jj': 
            emb_o = gcn(open_dim_seq1, adj, sparse,  None)
            emb_t = gcn(composed_dim_seq, adj, sparse,  None)
            # return emb_o + emb_t
            return torch.cat([emb_o, emb_t], dim=1)
        
        # elif self.ablation_choice[-4:-2] == 'de':

        #     # Xc -> Xc + p_node-wise prompt vector
        #     p_c = self.local_composed_fea(composed_dim_seq)
        #     xcp = composed_dim_seq + p_c 
        #     gcn_cp = gcn(xcp, adj, sparse,  None)
        #     # gcn_c = gcn(composed_dim_seq, adj, sparse, None)

        #     # Xo -> Xo + p_node wiee prompt vector 
        #     p_o = self.local_composed_fea2(open_dim_seq)
        #     xop = open_dim_seq + p_o
        #     gcn_op = gcn(xop, adj, sparse,  None)
        #     # gcn_o = gcn(open_dim_seq, adj, sparse, None)

        #     return gcn_cp + gcn_op
        
        # if self.ablation_choice == 'dofo':
        #     op = self.local_composed_fea(open_dim_seq)
        #     return gcn(op, adj, sparse, None)
        # elif self.ablation_choice == 'dcfo': 
        #     op1 = self.local_composed_fea(open_dim_seq)
        #     op2 = self.local_composed_fea(composed_dim_seq)
        #     return gcn(op1, adj, sparse, None)# + gcn(op2, adj, sparse, None)
        # elif self.ablation_choice == 'do': 
        #     return gcn(open_dim_seq, adj, sparse, None)

        # if self.ablation_choice == 'xo_xt': 
        #     return gcn(open_dim_seq, adj, sparse,  None) + gcn(open_dim_seq2, adj, sparse, None)
        
        # elif self.ablation_choice == 'xt':
        #     return gcn(open_dim_seq2, adj, sparse, None)
        # elif self.ablation_choice == 'xc': 
        #     return gcn(composed_dim_seq2, adj, sparse, None)
        
        # elif self.ablation_choice == 'xo_xt_ge': 
        #     return gcn(open_dim_seq+open_dim_seq2, adj, sparse, None)
        # elif self.ablation_choice == 'xo_xt_xc': 
        #     return  gcn(open_dim_seq+open_dim_seq2, adj, sparse, None) + gcn(composed_dim_seq, adj, sparse, None)
        # elif self.ablation_choice == 'xo_ft': 
        #     op = self.local_composed_fea(open_dim_seq)
        #     return gcn(op, adj, sparse, None)
        # elif self.ablation_choice == 'o3':
        #     return gcn(open_dim_seq3, adj, sparse, None)
        if self.ablation_choice[-2:] == 'no': 
            embed_fea = gcn(seq, adj, sparse, None)
            return embed_fea
        

        # f_o_s_o_str = gcn(open_dim_seq, adj, sparse, None, self.open_prompt_str)
        # f_c_s_o_str = gcn(composed_dim_seq, adj, sparse, None, self.open_prompt_str)
        # f_o_s_c_str = gcn(open_dim_seq, adj, sparse, None, self.composedprompt_str)
        # f_c_s_c_str = gcn(composed_dim_seq, adj, sparse, None, self.composedprompt_str)

        # if self.ablation_choice == 'stoo': 
        #     return gcn(open_dim_seq, adj, sparse,  self.open_prompt_str)
        # elif self.ablation_choice == 'stco':
        #     return gcn(composed_dim_seq, adj, sparse,  self.open_prompt_str)
        # elif self.ablation_choice == 'stoc':
        #     return gcn(open_dim_seq, adj, sparse,  self.composedprompt_str)
        # elif self.ablation_choice == 'stcc':
        #     return gcn(composed_dim_seq, adj, sparse, self.composedprompt_str)
        
class downpromptFUG(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  dim_pretext_weight, fea_pretext_weights, balance_weights,
                  combines, type_='mul', ablation = 'all', sample_size = 182, 
                  if_rand=False, gamma=0.5, basis_matrix=None, n_mlp_layer=1, 
                  sampling='random', agg_feat=None, shared=False, shared_token=None):
        super(downpromptFUG, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreampromptFUG(feature_dim, ft_in, num_layers_num, 
            dim_pretext_weight, fea_pretext_weights, balance_weights, 
            combines, type_, ablation, sample_size, if_rand, gamma, basis_matrix, 
            n_mlp_layer, sampling, agg_feat, shared, shared_token)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)
        self.agg_feat = agg_feat

    def get_emb (self, features, gcn, adj, sparse): 
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0) 
        return embeds
    
    def ssl_loss_fn_infoNCE(self, z):
        z = F.normalize(z, dim=1)
        return z.mean(dim=0).pow(2).mean()

    def ssl_loss_fn_pos(self, z, edge_index):
        return (z[edge_index[0]]-z[edge_index[1]]).pow(2).mean()


    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0,batch=None):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        
        if batch != None: # graph classification 
            rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
        else: # node classification 
            rawret = embeds[idx]
        num =  rawret.shape[0]

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype 
        
        # 코사인 유사도 기반 분류 
        rawret = torch.cat((rawret, self.ave.to(rawret.device)), dim=0)  # shape: (B+C, D) -> Query node B 개와 C개의 prototype  비교 
        rawret = F.normalize(rawret, dim=1)            # 모든 row 벡터 L2 정규화
        rawret = rawret @ rawret.T                     # (N+K) x (N+K) 유사도 행렬 (cosine similarity)

        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)

        # 유클리디안 거리 기반 
        # prototype = self.ave.to(rawret.device)  # [C, D]
        # dists = torch.cdist(rawret, prototype, p=2)  # L2 distance
        # ret = F.softmax(-dists, dim=1)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def de_loss(self): 
        return self.downstreamPrompt.open_prompt_dim.dimensional_loss()

class finetune(nn.Module):
    def __init__(self, gcn, ft_in, nb_classes, feature_dim, 
                  num_source_domains,
                ablation = 'all', sample_size=182, if_rand=False, n_mlp_layer=1, sampling='random'):
        super(finetune, self).__init__()
        
        self.pre_train_model = gcn 
        self.open_prompt_dim = DimensionNN_FUG(sample_size, ft_in, feature_dim, nn.PReLU, layers=n_mlp_layer)
        self.sampler = Sampler(sample_size=sample_size, if_rand=if_rand, sampling=sampling)

        self.sampling = sampling 

        self.num_pretrain_datasets = num_source_domains
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)

        self.sample_size = sample_size
        self.if_rand = if_rand 
        self.ablation = ablation
        
    def forward(self,features,adj,sparse,idx,labels=None,train=0):
        if self.ablation == 'DEt_finetune':
            # dimension reduction 먼저 
            # sample = dimensional_sample_random(self.sample_size, features, if_rand=self.if_rand)
            
            if self.sampling == 'degree': 
                sample = self.sampler(features, adj)
            else: 
                sample = self.sampler(features)

            # 샘플 대상으로 basis 벡터 추출 
            dimension_sig_open = self.open_prompt_dim(sample)
            open_dim_seq = self.open_prompt_dim.feature_sig_propagate(features, dimension_sig_open)

            embeds = self.pre_train_model(open_dim_seq, adj, sparse, None).squeeze(0)   
        
        elif self.ablation == 'pca_finetune': 
            embeds = self.pre_train_model(features, adj, sparse, None).squeeze(0)  
        
        # embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = embeds[idx]
        num =  rawret.shape[0]

        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype 
        
        # 코사인 유사도 기반 분류 
        rawret = torch.cat((rawret, self.ave.to(rawret.device)), dim=0)  # shape: (B+C, D) -> Query node B 개와 C개의 prototype  비교 
        rawret = F.normalize(rawret, dim=1)            # 모든 row 벡터 L2 정규화
        rawret = rawret @ rawret.T                     # (N+K) x (N+K) 유사도 행렬 (cosine similarity)

        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def de_loss(self): 
        return self.downstreamPrompt.open_prompt_dim.dimensional_loss()
    
class downstreampromptPerm(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, fea_pretext_weights, str_pretext_weights, 
                combines, type_ = 'mul', ablation = 'ft', perm_hid_dim=128, perm_n_layers=2, mlp_init=True):
        super(downstreampromptPerm, self).__init__()
        self.feature_dim = feature_dim
        self.perm_layer = NodeFeaturePermMLP(d=feature_dim, hidden_dim=perm_hid_dim, n_layers=perm_n_layers, mlp_init=mlp_init)
        
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList([
            composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
            for i in range(num_layers_num)
        ])
        
        self.open_prompt_fea = textprompt(feature_dim)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            new_layer = textprompt(in_features, type_)
            self.open_prompt_str.append(new_layer)
        #nn.ModuleList([textprompt(hidden_dim, type) for _ in range(num_layers_num)])

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

    def forward(self, seq, gcn, adj, sparse):
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
        
        # logits = self.perm_layer(seq) # (D, D) from row to col
        # perm_matrix = build_hard_permutation_from_logits(logits.view(self.feature_dim, self.feature_dim))
        # perm_idx = perm_matrix.argmax(dim=0) # (D, )
        # # print(perm_idx)
        # seq = seq[:, perm_idx]

        perm_matrix = self.perm_layer(seq)
        seq = seq @ perm_matrix.T

        composed_seq_fea = self.composedprompt_fea(seq)
        open_seq_fea = self.open_prompt_fea(seq)
        
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)
        
        if self.ablation_choice == 'ft':
            embed_fea = gcn(seq_fea, adj, sparse, None)
        elif self.ablation_choice[-2:] == 'fo':
            embed_fea = gcn(open_seq_fea, adj, sparse, None)
        elif self.ablation_choice[-2:] == 'fc':
            embed_fea = gcn(composed_seq_fea, adj, sparse, None)        
        
        return embed_fea
        
        # composed_embed_str = gcn(seq, adj, sparse, None, self.composedprompt_str)
        # open_embed_str = gcn(seq, adj, sparse, None, self.open_prompt_str)
        # if self.beta < 0:
        #     embed_str = self.weighted_prompt([composed_embed_str, open_embed_str])
        # else:
        #     embed_str = composed_embed_str + self.beta * open_embed_str
        
        # if self.ablation_choice[:2] == 'so':
        #     embed_str = open_embed_str
        # elif self.ablation_choice[:2] == 'sc':
        #     embed_str = composed_embed_str
        # if self.ablation_choice == 'st':
        #     return embed_str
        
        # ret = embed_fea + self.alpha * embed_str
        # return ret

class downpromptPerm(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  fea_pretext_weights, str_pretext_weights,
                  combines, type_='mul', ablation = 'all', perm_hid_dim=128, perm_n_layers=2, mlp_init=True):
        super(downpromptPerm, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreampromptPerm(feature_dim, ft_in, num_layers_num, 
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation, perm_hid_dim=perm_hid_dim, perm_n_layers=perm_n_layers, mlp_init=mlp_init)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)


    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = embeds[idx]
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype [C x D]
        ret = torch.FloatTensor(num,self.nb_classes)

        # rawret = torch.cat((rawret,self.ave),dim=0)
        # rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)

        # 정규화 후 행렬 곱 -> 더 빠름. u*v만 구하면 됨.  
        rawret = torch.cat((rawret, self.ave), dim=0)  # shape: (B+C, D) -> Query node B 개와 C개의 prototype  비교 
        rawret = F.normalize(rawret, dim=1)            # 모든 row 벡터 L2 정규화
        rawret = rawret @ rawret.T                     # (N+K) x (N+K) 유사도 행렬 (cosine similarity)

        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downstreampromptMLP(nn.Module):
    def __init__(self, feature_dim, 
                 target_mlp, fea_domain_weights, 
                combines, type_ = 'mul', ablation = 'all'):
        super(downstreampromptMLP, self).__init__()

        self.target_mlp = target_mlp
        self.composedprompt_fea = composedtoken(fea_domain_weights, type_) # unifying token 
        self.open_prompt_fea = textprompt(feature_dim) # specific prompt 
    
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

    def forward(self, seq, gcn, adj, sparse):
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
        
        mlp_seq = self.target_mlp(seq)
        composed_seq_fea = self.composedprompt_fea(mlp_seq)
        open_seq_fea = self.open_prompt_fea(mlp_seq)
        
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(mlp_seq), self.open_prompt_fea(mlp_seq)])
        else:
            seq_fea = self.composedprompt_fea(mlp_seq) + self.beta * self.open_prompt_fea(mlp_seq)
        
        if self.ablation_choice[-2:] == 'fo':
            seq_fea == open_seq_fea
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = composed_seq_fea
        
        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == 'ft':
            return embed_fea
    
        ret = embed_fea 
        return ret

class downpromptMLP(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim,
                  target_mlp, fea_domain_weights,
                  combines, type_='mul', ablation = 'all'):
        super(downpromptMLP, self).__init__()

        self.num_pretrain_datasets = len(fea_domain_weights)
        
        self.downstreamPrompt = downstreampromptMLP(feature_dim, target_mlp, 
                                                    fea_domain_weights, combines, type_, ablation)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)


    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = embeds[idx]
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype 
        ret = torch.FloatTensor(num,self.nb_classes)
        rawret = torch.cat((rawret,self.ave),dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

class TargetAlignedModel(torch.nn.Module):
    def __init__(self, pretrained_model, featuremlp):
        super().__init__()
        self.model = pretrained_model  # frozen or partially frozen
        self.target_mlp = featuremlp

    def forward(self, x, adj, sparse, LP):
        x_proj = self.target_mlp(x)
        return self.model.gcn(x_proj, adj, sparse, LP)
    
class PrototypeClassifier(nn.Module):
    def __init__(self, nb_classes, hidden_dim):
        super().__init__()
        self.nb_classes = nb_classes
        self.hidden_dim = hidden_dim
        # self.class_prototypes = torch.zeros(nb_classes, hidden_dim)
        self.register_buffer("class_prototypes", torch.zeros(nb_classes, hidden_dim))

    def forward(self, embs, idx_query):
        # cosine similarity to prototypes
        query = embs[idx_query]  # [Q, D]
        sim = F.cosine_similarity(query.unsqueeze(1), self.class_prototypes.unsqueeze(0), dim=-1)
        return F.softmax(sim, dim=1)

    def update_prototypes(self, embs, idx_support, support_labels):
        # 평균 프로토타입 계산
        for cls in range(self.nb_classes):
            mask = support_labels == cls
            self.class_prototypes[cls] = embs[idx_support][mask].mean(dim=0)

class downstreamprompt2(nn.Module):
    def __init__(self, feature_dim, fea_pretext_weights, 
                combines, type_ = 'mul', ablation = 'all',
                n_mlp_layer=1, init_identity=False, mlp_bias=True, mixing_mlp=False):
        super(downstreamprompt2, self).__init__()
        self.composedprompt_fea = composedtoken2(fea_pretext_weights, type_)
        
        self.open_prompt_fea = FeatureMLP(in_dim=feature_dim, hidden_dim=feature_dim, out_dim=feature_dim, 
                                          num_layer=n_mlp_layer, init_identity=init_identity, mlp_bias=mlp_bias)
        self.open_prompt_str = nn.ModuleList()


        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)
        self.ablation_choice = ablation
        self.mixing_mlp = mixing_mlp

    def forward(self, seq, gcn, adj, sparse):
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
            
        composed_seq_fea = self.composedprompt_fea(seq)
        open_seq_fea = self.open_prompt_fea(seq)
        
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)

        if self.ablation_choice[-2:] == 'ft': 
            embed_fea = gcn(seq_fea, adj, sparse, None)
        elif self.ablation_choice[-2:] == 'fo':
            embed_fea = gcn(open_seq_fea, adj, sparse, None)
        elif self.ablation_choice[-2:] == 'fc':
            embed_fea = gcn(composed_seq_fea, adj, sparse, None)
        
        return embed_fea
        


class downprompt2(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, 
                  fea_pretext_weights,
                  combines, type_='mul', ablation = 'all',
                  n_mlp_layer=1, init_identity=False, 
                  mlp_bias=True, mixing_mlp=False):
        super(downprompt2, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        self.downstreamPrompt = downstreamprompt2(feature_dim=feature_dim, fea_pretext_weights=fea_pretext_weights, 
                                                  combines=combines, type_=type_, ablation=ablation,
                                                  n_mlp_layer=n_mlp_layer, init_identity=init_identity, 
                                                  mlp_bias=mlp_bias, mixing_mlp=mixing_mlp)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)
        self.mixing_mlp = mixing_mlp


    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = embeds[idx]
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype 
        ret = torch.FloatTensor(num,self.nb_classes)
        rawret = torch.cat((rawret,self.ave),dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)
        return ret
    
    def get_emb(self, features, gcn, adj, sparse): 
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        return embeds 
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downstreamprompt(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, fea_pretext_weights, str_pretext_weights, 
                combines, type_ = 'mul', ablation = 'all', shared=False, shared_token=None):
        super(downstreamprompt, self).__init__()
        self.composedprompt_fea = composedtoken(fea_pretext_weights, type_)
        self.composedprompt_str = nn.ModuleList([
            composedtoken([pretext[i] for pretext in str_pretext_weights], type_)
            for i in range(num_layers_num)
        ])
        
        self.open_prompt_fea = textprompt(feature_dim)
        self.open_prompt_str = nn.ModuleList()
        for weight in str_pretext_weights[0]:
            in_features = weight.size(1)
            new_layer = textprompt(in_features, type_)
            self.open_prompt_str.append(new_layer)
        #nn.ModuleList([textprompt(hidden_dim, type) for _ in range(num_layers_num)])

        self.alpha = combines[0]
        self.beta = 1.0 if len(combines) <= 1 else combines[1]
        self.weighted_prompt = weighted_prompt(2)

        self.ablation_choice = ablation

        self.shared = shared
        self.shared_token = shared_token 

    def forward(self, seq, gcn, adj, sparse):
        if self.ablation_choice == 'None':
            return gcn(seq, adj, sparse, None)
            
        composed_seq_fea = self.composedprompt_fea(seq)
        open_seq_fea = self.open_prompt_fea(seq)
        
        if self.beta < 0:
            seq_fea = self.weighted_prompt([self.composedprompt_fea(seq), self.open_prompt_fea(seq)])
        else:
            seq_fea = self.composedprompt_fea(seq) + self.beta * self.open_prompt_fea(seq)
        
        if self.ablation_choice[-2:] == 'fo':
            seq_fea = open_seq_fea
            embed_fea = gcn(seq_fea, adj, sparse, None)
            return embed_fea
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = composed_seq_fea
            embed_fea = gcn(seq_fea, adj, sparse, None)
            return embed_fea

        # 수정 
        elif self.ablation_choice == 'ft': 
            if self.shared: 
                seq_fea = F.relu(seq_fea)
                seq_fea = self.shared_token * seq_fea
            embed_fea = gcn(seq_fea, adj, sparse, None)
            return embed_fea
        
        embed_fea = gcn(seq_fea, adj, sparse, None)
        if self.ablation_choice == 'ft':
            return embed_fea
        
        composed_embed_str = gcn(seq, adj, sparse, None, self.composedprompt_str)
        open_embed_str = gcn(seq, adj, sparse, None, self.open_prompt_str)
        if self.beta < 0:
            embed_str = self.weighted_prompt([composed_embed_str, open_embed_str])
        else:
            embed_str = composed_embed_str + self.beta * open_embed_str
        
        if self.ablation_choice[:2] == 'so':
            embed_str = open_embed_str
        elif self.ablation_choice[:2] == 'sc':
            embed_str = composed_embed_str
        if self.ablation_choice == 'st':
            return embed_str
        
        ret = embed_fea + self.alpha * embed_str
        return ret


class downprompt(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  fea_pretext_weights, str_pretext_weights,
                  combines, type_='mul', ablation = 'all', shared=False, shared_token=None):
        super(downprompt, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreamprompt(feature_dim, ft_in, num_layers_num, 
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation, shared, shared_token)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)

    def get_emb (self, features, gcn, adj, sparse): 
        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0) 
        return embeds

    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = embeds[idx]
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret) # prototype 
        ret = torch.FloatTensor(num,self.nb_classes)
        # rawret = torch.cat((rawret,self.ave),dim=0)
        # rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)

        rawret = torch.cat((rawret, self.ave), dim=0)  # shape: (B+C, D) -> Query node B 개와 C개의 prototype  비교 
        rawret = F.normalize(rawret, dim=1)            # 모든 row 벡터 L2 정규화
        rawret = rawret @ rawret.T                     # (N+K) x (N+K) 유사도 행렬 (cosine similarity)

        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class downprompt_graph(nn.Module):
    def __init__(self, ft_in, nb_classes, feature_dim, num_layers_num, 
                  fea_pretext_weights, str_pretext_weights,
                  combines, type_='mul', ablation = 'all'):
        super(downprompt_graph, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreamprompt(feature_dim, ft_in, num_layers_num, 
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)


    def forward(self,features,adj,sparse,gcn,idx,batch,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = torch_scatter.scatter(src=embeds[idx],index=batch,dim=0,reduce='mean')
        num =  rawret.shape[0]
        if train == 1:
            self.ave = averageemb(labels=labels, rawret=rawret)
        ret = torch.FloatTensor(num,self.nb_classes)
        rawret = torch.cat((rawret,self.ave),dim=0)
        rawret = torch.cosine_similarity(rawret.unsqueeze(1), rawret.unsqueeze(0), dim=-1)
        ret = rawret[:num,num:]
        ret = F.softmax(ret, dim=1)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

def averageemb(labels, rawret):
    retlabel = torch_scatter.scatter(src=rawret,index=labels,dim=0,reduce='mean')
    return retlabel