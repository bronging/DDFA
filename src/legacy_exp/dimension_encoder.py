from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg, FeatureMLP, DimensionNN_V2, GCN_encoder, FUG
from preprompt import PrePrompt, PrePrompt2, pca_compression, PrePromptwithMLP, MDGPTwithPerm, PrePromptNorm
import preprompt
import pdb
import os
import sys
import tqdm
import argparse
from downprompt import downprompt, downprompt_graph, TargetAlignedModel, PrototypeClassifier
import csv
from tqdm import tqdm, trange

import torch.nn.functional as F
import torch
import logging
from utils.dataloader import PretrainDatasetAug
from torch.utils.data import DataLoader
import warnings
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug
from utils.logging_ import * 
import train 
import adapation 
import gc 

def set_seed(seed=42):
    random.seed(seed)                       # Python 내장 random
    np.random.seed(seed)                    # NumPy
    torch.manual_seed(seed)                 # PyTorch (CPU)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # GPU
        torch.cuda.manual_seed_all(seed)    # Multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu(gpu): 
    print("CUDA Available:", torch.cuda.is_available())
    print('gpu:', str(gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(gpu)
    device = torch.device("cuda")
    return device

torch.cuda.empty_cache()
parser = argparse.ArgumentParser("SAMGPT")
parser.add_argument('--dataset', type=str, default='FacebookPagePage', help='target data')
parser.add_argument('--pretrain_datasets', nargs='+', type=str, 
    help='pretrain datasets', default=['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'LastFMAsia'])
parser.add_argument('--downstream_task', type=str, default='node', help='node or graph')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", help='GRAPHCL or LP or splitLP')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')   
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help='target graph\'s id in one dataset')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative_samples_num')
parser.add_argument('--skip_pretrain', type=int, default=1, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='ft', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='ft', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
parser.add_argument('--batch_size', type=int, default=1, help='backbone')
parser.add_argument('--restart_epoch', type=int, default=0, help='이어서 학습 시작할 에포크')

parser.add_argument('--separate_learning', type=str2bool, default=True, help='mlp/gcn 학습 같이(f) or 따로(t)')
parser.add_argument('--w1alpha', type=float, default=1.0, help='graphcl loss + w1alpha*w1loss')

parser.add_argument('--experiment', type=str, default='EXP0527_vec2mlp', help='실험 종류')

parser.add_argument('--nb_epochs', type=int, default=200, help='pretraining epoch')
parser.add_argument('--mlp_init', type=str2bool, default=False, help='mlp identity init')
parser.add_argument('--n_mlp_layer', type=int, default=1, help='num of mlp layers')
parser.add_argument('--mlp_bias', type=str2bool, default=True, help='mlp bias or not')
parser.add_argument('--mixing_mlp', type=str2bool, default=True, help='adaptation-mlp 결과 믹싱 사용 여부')
parser.add_argument('--mlpalpha', type=float, default=1.0, help='perm regular 비율 ㄴ')
parser.add_argument('--down_mlp', type=str2bool, default=False, help='학습은 with perm, downstream은 w/o perm')
parser.add_argument('--if_rand', type=str2bool, default=False, help='sampling 방법')

parser.add_argument('--source_id', type=int, default=0, help='mlp bias or not')
parser.add_argument('--target_id', type=int, default=0, help='[Cora, Citeseer, Pubmed, Photo, Computers, FacebookPagePage, LastFMAsia]')
parser.add_argument('--model_type', type=str, default='permMDGPT', help='[samgpt, anchor_mlp, vec2mlp, permMDGPT]')

parser.add_argument('--graph_batch', type=str2bool, default=False, help='graph 하나씩 학습')
args = parser.parse_args()

warnings.filterwarnings("ignore")

seed = args.seed
shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
negative_samples_num = args.negative_samples_num
aug_type = args.aug_type
drop_percent = args.drop_percent
hid_units = args.hid_units
num_layers_num = args.layers_num
dataset = args.dataset
unify_dim = args.unify_dim
target_graph_id = args.graphId
lr = args.lr

pretrain_method = args.pretrain_method
graph_batch = args.graph_batch
mlp_init = args.mlp_init
experiment = args.experiment 
nb_epochs = args.nb_epochs
n_mlp_layer = args.n_mlp_layer 
mlp_bias = args.mlp_bias 
mixing_mlp = args.mixing_mlp

ablation_pre = args.ablation_pre
ablation_down = args.ablation_down 

source_id = args.source_id if args.model_type == 'anchor_mlp' else args.target_id
target_id = args.target_id
model_type = args.model_type
LP = (args.pretrain_method == 'LP')

#         0         1          2         3         4                5                6        
# data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']
data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'Texas', 'Cornell', 'Wisconsin', 'chameleon', 'squirrel']
dataset = data[args.target_id]
downstream = data[args.target_id]

#TODO anchor graph 사용 안 할 때는 target id만 제외해야 함. 
pretrain_dataset_names = get_pretrain_dataset_names(data, source_id, target_id)
print(pretrain_dataset_names)


device = set_gpu(args.gpu)
set_seed(seed)

patience = 50
l2_coef = 0.0
drop_prob = 0.0
sparse = True
best = 1e9
best_t = 0
firstbest = 0
cnt_wait = 0
test_idx_num = 100
negetive_sample = torch.tensor(0.0)

logfile, save_dir, result_dir, cache_dir = make_dir(experiment)
save_name, csv_name = get_save_name(args, pretrain_dataset_names, save_dir, result_dir)


logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=logfile,
                    filemode='a', 
                    encoding="utf-8",)

log_args_table(args, max_per_line=5, col_width=30)
write(f"pretrain_dataset: {pretrain_dataset_names}")
write(f"source id: {source_id}, target id: {target_id}")

torch.autograd.set_detect_anomaly(True)

num_pretrain_dataset_num = len(pretrain_dataset_names)

b_xent = nn.BCEWithLogitsLoss()

nonlinearity = 'prelu'  # special name to separate parameters

from scipy.optimize import linear_sum_assignment
def find_optimal_permutation(W1, W2, dim=50, normed=True):
    """
    W1, W2: torch tensors of shape (50, 50)
    Returns: permutation indices such that W1's columns are permuted to best match W2
    """
    W1_np = W1.detach().cpu().numpy()
    W2_np = W2.detach().cpu().numpy()
    
    # 1. (선택) 각 column L2 정규화
    if normed:
        W1_np = W1_np / (np.linalg.norm(W1_np, axis=0, keepdims=True) + 1e-8)
        W2_np = W2_np / (np.linalg.norm(W2_np, axis=0, keepdims=True) + 1e-8)
    
    # 2. cost matrix (여기선 cosine similarity 최대화 → -cosine 최소화)
    cost = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            # -cosine similarity
            cost[i, j] = -np.dot(W1_np[:, i], W2_np[:, j])
    
    # 3. Hungarian algorithm으로 최적 매칭
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind, col_ind는 모두 np.arange(50)와 동일하므로 col_ind가 곧 permutation 순서
    print(f'row: {row_ind[:10]}')  
    print(f'col: {col_ind[:10]}')
    return col_ind # W2에 적용하면 W1 처럼 됨.


def dimensional_sample_random(sample_size, x, edge_index, if_rand=False):
    with torch.no_grad():
        if if_rand != True:
            d_sample_matrix = x[:sample_size, :]
        else:
            d_sample_matrix = x[torch.randperm(x.shape[0]),:][:sample_size, :]
        return d_sample_matrix


if model_type == 'samgpt': 
    write(f"✅ SAMGPT - GraphCL")
    write(f'pretrain-dataset: {pretrain_dataset_names}')

    model = PrePrompt(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
            num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
            alpha = args.alpha, ablation = ablation_pre).cuda()
elif model_type == 'norm_mdgpt': 
    write(f"✅ Norm MDGPT - GraphCL")
    write(f'pretrain-dataset: {pretrain_dataset_names}')
    model = PrePromptNorm(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
            num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
            alpha = args.alpha, ablation = ablation_pre, scaling_factor=3).cuda()
      
sample_size = 183
perm = False 
write(f'Sample size: {sample_size}')
write(f'Permutation: {perm}')
dnn = DimensionNN_V2(n_in=sample_size, n_h=unify_dim*2, n_out=unify_dim, activator=nn.PReLU, layers=n_mlp_layer)
gnn = GCN_encoder(unify_dim, hid_units, nn.PReLU)
base_model = FUG(D_NN=dnn, G_NN=gnn, S_mtd=dimensional_sample_random, sample_size=sample_size)
base_state = base_model.state_dict()

try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:
    pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
    
    features, adjs, edge_indexs = process.get_features_adjs(pretrain_loaders,  \
                       cache_dir, pretrain_dataset_names, target_graph_id)
    
    set_seed(seed)
    
    if torch.cuda.is_available():
        print('Using CUDA')
        
        features = [tensors.cuda() for tensors in features]
        edge_indexs = [tensors.cuda() for tensors in edge_indexs]

    
    # DE 학습 
    # leng = [len(features[i]) for i in range(len(features))]
    # sample_size = min(leng)
    # print('sample size: ', sample_size)

    # base_model = DimensionNN_V2(n_in=sample_size, n_h=unify_dim*2, n_out=unify_dim, activator=nn.PReLU, layers=n_mlp_layer)
    # base_state = base_model.state_dict()


    dim_encoders = [] 
    dimension_sigs = []
    
    alpha = 1.0
    
    for i in range(len(features)): 
        print(pretrain_dataset_names[i])

        
        # encoder = DimensionNN_V2(n_in=sample_size, n_h=unify_dim*2, n_out=unify_dim, activator=nn.PReLU, layers=n_mlp_layer)

        dnn = DimensionNN_V2(n_in=sample_size, n_h=unify_dim*2, n_out=unify_dim, activator=nn.PReLU, layers=n_mlp_layer)
        gnn = GCN_encoder(unify_dim, hid_units, nn.PReLU)
        encoder = FUG(D_NN=dnn, G_NN=gnn, S_mtd=dimensional_sample_random, sample_size=sample_size)
        # encoder.load_state_dict(base_state)  # 동일 초기화
        optimiser = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=0.00001)

        

        feature = features[i]
        edge_index = edge_indexs[i]

        print('Before feature: ', feature.size())
        # 학습 
        encoder.cuda()
        loss_mi = 0 
        with tqdm(total=nb_epochs*2, desc='(T)') as pbar:
            total_loss = 0
            l_ssl = 0
            l_ssl_pos = 0
            l_sig_cross = 0
            losslam_ssl = 1
            losslam_sig_cross = 400
            losslam_ssl_pos = 1
            for epoch in range(nb_epochs*2):
                # encoder.train()
                # optimiser.zero_grad()
                
                # d_sample_matrix = dimensional_sample_random(sample_size, feature, if_rand=args.if_rand)
                # dimension_sig = encoder(d_sample_matrix)
                # reduced_feature = F.normalize(feature @ dimension_sig)
                
                # optimiser.zero_grad()
                # loss_de = encoder.dimensional_loss()
                # # loss_mi = mi_loss_infonce(feature, reduced_feature)
                # loss = loss_de + alpha * loss_mi
                # loss.backward()
                # optimiser.step()

                # total_loss += loss.item()

                # pbar.set_postfix({'loss': total_loss})
                # pbar.update()
                encoder.update_sample(feature, edge_index, if_rand=args.if_rand)
                encoder.train()
                optimiser.zero_grad()
                z = encoder(feature, edge_index)
                loss_ssl = encoder.ssl_loss_fn_infoNCE(z)
                loss_ssl_pos = encoder.ssl_loss_fn_pos(z, edge_index)
                loss_sig_cross = encoder.dim_loss_fn()
                loss = losslam_ssl * loss_ssl + losslam_sig_cross * loss_sig_cross + losslam_ssl_pos * loss_ssl_pos
                loss.backward()
                optimiser.step()
                l_ssl = l_ssl + loss_ssl.item()
                l_ssl_pos = l_ssl_pos + loss_ssl_pos.item()
                l_sig_cross = l_sig_cross + loss_sig_cross.item()
                
                pbar.set_postfix({'loss_ssl': l_ssl, 
                                    'loss_ssl_pos': l_ssl_pos,
                                    'loss_sig_cross': l_sig_cross
                                })
                pbar.update()
            

        reduced_feature = encoder.reduced_feature(feature)
        #dimension_sigs.append(reduced_feature)
        dim_encoders.append(encoder)

        features[i] = reduced_feature#.detach()
        print('After: ', features[i].size())
        print('dimension sig: ', reduced_feature.size(), '\n')

    # print('dim encoder weight: ', dim_encoders[0].lin_in.weight.size()) # [50, 3327]
    # print('dimension_sig: ', dimension_sigs[0].size())

    if perm: 
        for i in range(1, len(features)):
            s2t_perm = find_optimal_permutation(dim_encoders[0].dnn.lin_in.weight.T, dim_encoders[i].dnn.lin_in.weight.T, dim=unify_dim, normed=False)
            print(dim_encoders[i].dnn.lin_in.weight.T.size(), " -> ", dim_encoders[i].dnn.lin_in.weight.T)
            #s2t_perm = find_optimal_permutation(dimension_sigs[0], dimension_sigs[i], normed=False)
            print(f"{i} -> {0} perm: ", s2t_perm)
            features[i] = features[i][:, s2t_perm] # t를 s처럼 만드는 perm  

    aug_features, aug_adjs, lbls, negetive_samples, combinedadj = process.preprocess_dataset_w_DE(
                                                                    features, adjs, pretrain_method, \
                                                                    sparse, drop_percent, negative_samples_num)
    
    print('Aug feat[0]: ', aug_features[0].size())
    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()

        adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).cuda()  if sparse else torch.FloatTensor(adj.todense()).cuda() 
            for adj in adjs]
        
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()

        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]
        lbls = [tensors.cuda() for tensors in lbls]

    # DE training 

    if model_type == 'samgpt' or model_type == 'norm_mdgpt':
        if pretrain_method == 'GRAPHCL':

            train.train_samgpt_graphcl(model=model, lr=lr, weight_decay=l2_coef, 
                            start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                            aug_features=aug_features, aug_adjs=aug_adjs, 
                            lbls=lbls, sparse=sparse, save_name=save_name, patience=patience)


        
write('#'*50)
write('PreTrain datasets are ')
write(pretrain_dataset_names)
write('Downastream dataset is ')
write(f"✅ {downstream}")

downstream_dataset = load_dataset(downstream)
downstream_loader = DataLoader(downstream_dataset)

for data in downstream_loader:
    print(data)
    t_feature, adj= process.process_tu(data,data.x.shape[1])
    print('process done')
    edge_index = data.edge_index.cuda()
    # features = torch.FloatTensor(pca_compression(t_feature,k=unify_dim)).cuda()
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y
    data=np.array(data.y)
    np.unique(data)
    nb_classes=len(np.unique(data))
    print('nb_classes', nb_classes)
    if args.downstream_task == 'graph':
        test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()
    t_feature = t_feature.cuda()

print(f'loading model from {save_name}')
model.load_state_dict(torch.load(save_name))
model = model.cuda()

save_path = save_name + "_apapt"

# encoder = DimensionNN_V2(n_in=sample_size, n_h=unify_dim*2, n_out=unify_dim, activator=nn.PReLU, layers=n_mlp_layer)
# sample_size = 3327
dnn = DimensionNN_V2(n_in=sample_size, n_h=unify_dim*2, n_out=unify_dim, activator=nn.PReLU, layers=n_mlp_layer)
gnn = GCN_encoder(unify_dim, hid_units, nn.PReLU)
encoder = FUG(D_NN=dnn, G_NN=gnn, S_mtd=dimensional_sample_random, sample_size=sample_size)
# encoder.load_state_dict(base_state)  # 동일 초기화
optimiser = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=0.00001)

print('Before feature: ', t_feature.size())
# 학습 
encoder.cuda()
loss_mi = 0 
total_loss = 0
l_ssl = 0
l_ssl_pos = 0
l_sig_cross = 0
losslam_ssl = 1
losslam_sig_cross = 400
losslam_ssl_pos = 1

with tqdm(total=nb_epochs*2, desc='(T)') as pbar:
    for epoch in range(nb_epochs*2):
        encoder.update_sample(t_feature, edge_index, if_rand=args.if_rand)
        encoder.train()
        optimiser.zero_grad()
        z = encoder(t_feature, edge_index)
        loss_ssl = encoder.ssl_loss_fn_infoNCE(z)
        loss_ssl_pos = encoder.ssl_loss_fn_pos(z, edge_index)
        loss_sig_cross = encoder.dim_loss_fn()
        loss = losslam_ssl * loss_ssl + losslam_sig_cross * loss_sig_cross + losslam_ssl_pos * loss_ssl_pos
        loss.backward()
        optimiser.step()
        l_ssl = l_ssl + loss_ssl.item()
        l_ssl_pos = l_ssl_pos + loss_ssl_pos.item()
        l_sig_cross = l_sig_cross + loss_sig_cross.item()
        
        pbar.set_postfix({'loss_ssl': l_ssl, 
                            'loss_ssl_pos': l_ssl_pos,
                            'loss_sig_cross': l_sig_cross
                        })
        pbar.update()
    

    reduced_feature = encoder.reduced_feature(t_feature)

    t_feature = reduced_feature #.detach()
    print('After: ', t_feature.size())
    # print('dimension sig: ', reduced_feature.size(), '\n')

# print('dim encoder weight: ', dim_encoders[0].lin_in.weight.size()) # [50, 3327]
# print('dimension_sig: ', dimension_sigs[0].size())

if perm: 
    s2t_perm = find_optimal_permutation(dim_encoders[0].dnn.lin_in.weight.T, encoder.dnn.lin_in.weight.T, dim=unify_dim, normed=False)
    print(dim_encoders[i].dnn.lin_in.weight.T.size(), " -> ", dim_encoders[i].dnn.lin_in.weight.T)
    #s2t_perm = find_optimal_permutation(dimension_sigs[0], dimension_sigs[i], normed=False)
    print(f"{i} -> {0} perm: ", s2t_perm)
    t_feature = t_feature[:, s2t_perm] # t를 s처럼 만드는 perm  

if model_type == 'samgpt' or model_type == 'norm_mdgpt': 
    if ablation_down == 'None': 
        adapation.adaptation_samgpt_node(model, t_feature, adj, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                                     args.combinetype, ablation_down, 1, save_path, epoch=100)
    else: 
        adapation.adaptation_samgpt_node(model, t_feature, adj, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, epoch=100)
