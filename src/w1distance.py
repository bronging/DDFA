import numpy as np
import scipy.sparse as sp
import random

from preprompt import *
import os
import argparse

import torch
import logging
from torch.utils.data import DataLoader
import warnings
import torch.nn as nn

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils.logging_ import * 
from utils.figure import *
import train 
import adapation 

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

# exp setting  
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--experiment', type=str, default='EXP0527_vec2mlp', help='실험 종류')
parser.add_argument('--target_id', type=int, default=0, help='[Cora, Citeseer, Pubmed, Photo, Computers, FacebookPagePage, LastFMAsia]')
parser.add_argument('--downstream_task', type=str, default='node', help='node or graph')
parser.add_argument('--skip_pretrain', type=int, default=1, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='ft', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='ft', help='ablation_down')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help='target graph\'s id in one dataset')
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", help='GRAPHCL or LP or splitLP')

# Model
parser.add_argument('--model_type', type=str, default='norm_mdgpt', help='[samgpt, anchor_mlp, vec2mlp, permMDGPT]')
parser.add_argument('--n_mlp_layer', type=int, default=2, help='num of mlp layers')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')   
parser.add_argument('--shared', type=str2bool, default=False, help='shared token or DE 사용 여부')   

# Training 
parser.add_argument('--nb_epochs', type=int, default=200, help='pretraining epoch')
parser.add_argument('--adapt_ep', type=int, default=100, help='adpatation epoch')
parser.add_argument('--ep_aug', type=str2bool, default=False, help='True: 매 에포크마다 augmentation 생성')
parser.add_argument('--batch_size', type=int, default=1, help='backbone')
parser.add_argument('--restart_epoch', type=int, default=0, help='이어서 학습 시작할 에포크')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative_samples_num')

# DE
parser.add_argument('--de_loss', type=float, default=1.5, help='dim encoder loss 얼마나 반영할지')
parser.add_argument('--de_weight', type=str2bool, default=False, help='de loss 합산 시 가중치 반영 여부 ')
parser.add_argument('--de_input', type=str, default='x', help='[x, ax, concat]')
parser.add_argument('--sample_size', type=int, default=256, help='DE input dimension size')
parser.add_argument('--sampling', type=str, default='random', help='DE 에 입력할 노드 샘플링 방법')
parser.add_argument('--if_rand', type=str2bool, default=False, help='sampling 방법')
parser.add_argument('--a', type=float, default=0.0, help='adatation 때 open dim encoder loss 얼마나 반영할지')
parser.add_argument('--gamma', type=float, default=0.0, help='open+composed prompt 가중합 비율 - (1-r)*open + r*com')
parser.add_argument('--barycenter', type=str2bool, default=False, help='barycenter loss 사용 여부')


# GraphACL
parser.add_argument('--temp', type=float, default=0.5, help='Temperature hyperparameter.')
parser.add_argument('--moving_average_decay', type=float, default=0.9)
parser.add_argument('--proj_num_mlp', type=int, default=1, help='projector layer  수')
parser.add_argument('--proj_mode', type=str, default='domain', help='domain=따로따로, all=하나만')

args = parser.parse_args()

warnings.filterwarnings("ignore")

set_seed(args.seed)
device = set_gpu(args.gpu)

shot_num = args.shot_num
negative_samples_num = args.negative_samples_num
aug_type = args.aug_type
drop_percent = args.drop_percent
hid_units = args.hid_units
layers_num = args.layers_num
unify_dim = args.unify_dim
target_graph_id = args.graphId
lr = args.lr

pretrain_method = args.pretrain_method
experiment = args.experiment 
nb_epochs = args.nb_epochs
n_mlp_layer = args.n_mlp_layer 

ablation_pre = args.ablation_pre
ablation_down = args.ablation_down 

target_id = args.target_id
model_type = args.model_type
sample_size = args.sample_size

# LP = (args.pretrain_method == 'LP')
LP=True # batch norm, drop out 

# data = ['Cora', 'Photo']
# data = ['Cora', 'Pubmed','FacebookPagePage', 'LastFMAsia']
        # 0         1          2         3         4                5                6        
data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']
#          0        1           2        3          4          5         6           7            8           9
# data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'Texas', 'Cornell', 'Wisconsin', 'chameleon', 'squirrel']
# data = ['Cora', 'Texas', 'Cornell', 'Wisconsin', 'chameleon', 'squirrel']
# data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'Reddit']
# data = ['Cora', 'Citeseer', 'Pubmed', 'Cornell', 'squirrel', 'chameleon']

dataset = data[args.target_id]
downstream = data[args.target_id]

#TODO anchor graph 사용 안 할 때는 target id만 제외해야 함. 
pretrain_dataset_names = get_pretrain_dataset_names(data, target_id)

print(pretrain_dataset_names)

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

torch.autograd.set_detect_anomaly(True)
pretrain_dataset_num = len(pretrain_dataset_names)

b_xent = nn.BCEWithLogitsLoss()

nonlinearity = 'prelu'  # special name to separate parameters

if model_type == 'GraphACL': 
    write(f"✅ GraphACL")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   de loss         : {args.de_loss}')
    write(f'   gamma           : {args.gamma}')
    
    model = PrePromptACL(unify_dim, hid_units, pretrain_dataset_num, layers_num, 
                0.1, type_ = args.combinetype, temp=args.temp, moving_average_decay=args.moving_average_decay, 
                num_MLP=args.proj_num_mlp, alpha=args.alpha, n_sample=sample_size, 
                if_rand=args.if_rand, de_loss=args.de_loss, de_weight=args.de_weight, 
                sampling=args.sampling,n_mlp_layer=args.n_mlp_layer, ablation = ablation_pre, proj_mode=args.proj_mode).cuda()

elif model_type == 'samgpt': 
    write(f"✅ SAMGPT - GraphCL")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   shared          : {args.shared}')

    model = PrePrompt(unify_dim, hid_units, nonlinearity, pretrain_dataset_num, 
            layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
            alpha = args.alpha, ablation = ablation_pre, shared=args.shared).cuda()
    
elif model_type == 'norm_mdgpt': 
    write(f"✅ DE - GraphCL")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   Backbone        : {args.backbone}')
    write(f'   de loss         : {args.de_loss}')
    write(f'   gamma           : {args.gamma}')
    write(f'   shared          : {args.shared}')

    model = PrePromptFUG(unify_dim, hid_units, nonlinearity, pretrain_dataset_num, 
            layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone, #'norm_mdgpt',
            alpha = args.alpha, ablation = ablation_pre, scaling_factor=3, n_sample=sample_size, 
            if_rand=False,  de_loss=args.de_loss, de_weight=args.de_weight, sampling=args.sampling,
            n_mlp_layer=args.n_mlp_layer, de_input=args.de_input, shared=args.shared).cuda()

elif model_type == 'sharedFUG': 
    write(f"✅ SharedDE - GraphCL")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   Backbone        : {args.backbone}')
    write(f'   de loss         : {args.de_loss}')
    write(f'   gamma           : {args.gamma}')
    model = PrePromptSharedFUG(unify_dim, hid_units, nonlinearity, pretrain_dataset_num, 
            layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone, #'norm_mdgpt',
            alpha = args.alpha, ablation = ablation_pre, scaling_factor=3, n_sample=sample_size, 
            if_rand=False,  de_loss=args.de_loss, de_weight=args.de_weight, sampling=args.sampling,
            n_mlp_layer=args.n_mlp_layer, de_input=args.de_input).cuda()

elif model_type == 'filterFUG': 
    write(f"✅ FilterDE - GraphCL")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   Backbone        : {args.backbone}')
    write(f'   de loss         : {args.de_loss}')
    write(f'   gamma           : {args.gamma}')
    write(f'   shared          : {args.shared}')
    model = PrePromptFilterFUG(unify_dim, hid_units, nonlinearity, pretrain_dataset_num, 
            layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone, #'norm_mdgpt',
            alpha = args.alpha, ablation = ablation_pre, scaling_factor=3, n_sample=sample_size, 
            if_rand=False,  de_loss=args.de_loss, de_weight=args.de_weight, sampling=args.sampling,
            n_mlp_layer=args.n_mlp_layer, de_input=args.de_input, shared=args.shared).cuda()

elif model_type == 'filterbank': 
    write(f"✅ FilterbankDE - GraphCL")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   Backbone        : {args.backbone}')
    write(f'   de loss         : {args.de_loss}')
    write(f'   gamma           : {args.gamma}')
    write(f'   shared          : {args.shared}')
    model = FilterbankFUG(in_dim=unify_dim, hid_dim=hid_units, activation=nonlinearity, num_pretrain_dataset=pretrain_dataset_num, 
                          gcn_layers=layers_num, dropout=0.1, type_=args.combinetype,  alpha=args.alpha, ablation=ablation_pre, 
                            n_sample=sample_size, if_rand=False, sampling=args.sampling, 
                            de_loss=args.de_loss,  de_layers=args.n_mlp_layer,  de_input=args.de_input, shared=args.shared).cuda()
elif model_type == 'barycenter': 
    write(f"✅ PrePromptBarycenter")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   Backbone        : {args.backbone}')
    write(f'   de loss         : {args.de_loss}')
    write(f'   gamma           : {args.gamma}')
    write(f'   shared          : {args.shared}')

    args.barycenter = True 
    model = PrePromptBarycenter(unify_dim, hid_units, nonlinearity, pretrain_dataset_num, 
            layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone, #'norm_mdgpt',
            alpha = args.alpha, ablation = ablation_pre, scaling_factor=3, n_sample=sample_size, 
            if_rand=False,  de_loss=args.de_loss, de_weight=args.de_weight, sampling=args.sampling,
            n_mlp_layer=args.n_mlp_layer, de_input=args.de_input, shared=args.shared).cuda()
elif model_type == 'w1mlp': 
    write(f"✅ PrePromptMLPW1Bary")
    write(f'   pretrain-dataset: {pretrain_dataset_names}')
    write(f'   Backbone        : {args.backbone}')

    PrePromptMLPW1Bary(unify_dim, hid_units, nonlinearity, pretrain_dataset_num, 
            layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone, #'norm_mdgpt',
            alpha = args.alpha, ablation = ablation_pre, scaling_factor=3, n_sample=sample_size, 
            if_rand=False,  de_loss=args.de_loss, de_weight=args.de_weight, sampling=args.sampling,
            n_mlp_layer=args.n_mlp_layer, de_input=args.de_input, shared=args.shared)


try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:
    pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
    
    features, adjs, edge_indexs = process.get_features_adjs(pretrain_loaders,  \
                    cache_dir, pretrain_dataset_names, target_graph_id)
    
    if ablation_pre == 'PCA' or model_type == 'samgpt': 
        # aug_features, aug_adjs, lbls, negetive_samples, combinedadj = process.preprocess_dataset_w_DE(
        #                                                                 features, adjs, pretrain_method, \
        #                                                                 sparse, drop_percent, negative_samples_num)
        features = [torch.FloatTensor(pca_compression(feature,k=unify_dim)) for feature in features]
        
    aug_features, aug_adjs, lbls, negetive_samples, combinedadj = process.preprocess_dataset_w_DE_pyg(
                                                                        features, edge_indexs, pretrain_method, \
                                                                        drop_percent, negative_samples_num)
    


    print('Aug feat[0]: ', aug_features[0].size())
    


    if torch.cuda.is_available():
        print('Using CUDA')

        features = [tensors.cuda() for tensors in features]
        
        
        if not args.ep_aug: 
            edge_indexs = [tensors.cuda() for tensors in edge_indexs]
            adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).cuda() if sparse else torch.FloatTensor(adj.todense()).cuda() 
                for adj in adjs]
        
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()

        # aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_adjs = [
            [edge_index.cuda() for edge_index in ei_list]
            for ei_list in aug_adjs
        ]
        
        aug_features = [tensors.cuda() for tensors in aug_features]
        lbls = [tensors.cuda() for tensors in lbls]
    

            
    if model_type in ['samgpt', 'norm_mdgpt', 'sharedFUG', 'filterFUG', 'filterbank', 'barycenter']:
        if pretrain_method == 'GRAPHCL':
            
            if args.ep_aug: 
                train.train_fug_graphcl_ep_aug(model=model, lr=lr, weight_decay=l2_coef, 
                        start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                        features=features, edge_indexs=edge_indexs, drop_percent=drop_percent,
                         sparse=sparse, save_name=save_name, patience=patience, gamma=args.gamma, model_type=model_type)
            else: 
                train.train_fug_graphcl(model=model, lr=lr, weight_decay=l2_coef, 
                        start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                        aug_features=aug_features, aug_adjs=aug_adjs, 
                        lbls=lbls, sparse=sparse, save_name=save_name, 
                        patience=patience, gamma=args.gamma, model_type=model_type, 
                        barycenter=args.barycenter,  sample_size=sample_size, unify_dim=unify_dim)
        
    elif model_type == 'GraphACL':
        train.train_graphacl(model, features, edge_indexs, save_name, \
                        lr=lr, weight_decay=l2_coef, \
                        start_epoch=args.restart_epoch, num_epoch=nb_epochs, patience=patience, gamma=args.gamma)


write('#'*50)
write(f'Downastream dataset is 🌼⭐ {downstream} ⭐🌼')

downstream_dataset = load_dataset(downstream)
downstream_loader = DataLoader(downstream_dataset)

for data in downstream_loader:
    print(data)
    t_feature, adj= process.process_tu(data,data.x.shape[1])
    print('process done')
    edge_index = data.edge_index.cuda()
    # pca_t_feature = torch.FloatTensor(pca_compression(t_feature,k=unify_dim)).cuda()
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

pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
    
features, adjs, edge_indexs = process.get_features_adjs(pretrain_loaders,  \
                       cache_dir, pretrain_dataset_names, target_graph_id)
aug_features, aug_adjs, lbls, negetive_samples, combinedadj = process.preprocess_dataset_w_DE_pyg(
                                                                    features, edge_indexs, pretrain_method, \
                                                                    drop_percent, negative_samples_num)


### barycenter - 각 소스 그래프들의 W1 distance 측정 
# features = [torch.FloatTensor(pca_compression(feature,k=unify_dim)) for feature in features]

# aug_adjs = [tensors.cuda() for tensors in aug_adjs]
aug_adjs = [
            [edge_index.cuda() for edge_index in ei_list]
            for ei_list in aug_adjs
        ]
features = [tensors.cuda() for tensors in features]
edge_indexs = [tensors.cuda() for tensors in edge_indexs]
aug_features = [tensors.cuda() for tensors in aug_features]
lbls = [tensors.cuda() for tensors in lbls]


# xt_list = [model.samplers[i](features[i], edge_indexs[i]) for i in range(len(features))]

# bary_Y = torch.randn(sample_size, unify_dim, device=device)
# b = torch.ones(sample_size, device=device) / sample_size 

# # 각 도메인별 support point에 대한 가중치
# as_ = [torch.ones(sample_size, device=device) / sample_size for _ in range(len(aug_features))]
# # 각 도메인에 대한 가중치 
# weights = torch.ones(len(aug_features), device=device) / len(aug_features)


# bary_Y = train.wasserstein_barycenter(xt_list, as_, bary_Y, b, weights, n_iter=30) # "매 epoch마다 새로 뽑는 anchor" 역할 -> detach()!
# W1loss = torch.tensor(0.0, dtype=torch.float32).to(aug_features[0].device)
# i = 0 
# for xt in xt_list:
#     print(f'{pretrain_dataset_names[i]} - W1 dist: {train.wasserstein_distance(xt, bary_Y, reg=0.1):.4f}')
#     i += 1

# # t_feature = torch.FloatTensor(pca_compression(t_feature,k=unify_dim)).cuda()
# i = 0 
# for xt in xt_list:
#     print(f'{pretrain_dataset_names[i]} - {downstream}: {train.wasserstein_distance(xt, pca_t_feature[torch.randperm(t_feature.shape[0])[:args.sample_size], :], reg=0.1):.4f}')
#     i += 1

# print(f'{downstream} - Barycenter: {train.wasserstein_distance(pca_t_feature[torch.randperm(t_feature.shape[0])[:args.sample_size], :], bary_Y, reg=0.1):.4f}')

# # # for idx, feat in enumerate(aug_features): 
# # #     print(f'{[pretrain_dataset_names[idx]]}')
# # #     print(f'{feat[0][0][:20]}\n')

# from train import aggregate_features
# aggregate_feat = [aggregate_features(features[i], edge_indexs[i], gamma=args.gamma) for i in range(len(features))]
# if model_type == 'GraphACL': 
#     loss = model(features, edge_indexs, aggregate_feat) 
# else: 
#     loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, aggregate_feat) 

# basis_matrix = [] 
# for idx, layer in enumerate(model.dimension_encoder_layers): 
#     basis_matrix.append(layer.basis_matrix().detach())
basis_matrix = None 

adapta_epoch = args.adapt_ep


if model_type == 'samgpt': 
    features = torch.FloatTensor(pca_compression(t_feature.cpu(),k=unify_dim)).cuda()
    if args.downstream_task == 'node': 
        adapation.adaptation_FUG_node(model, features, edge_index, labels, \
                                    sparse, idx_test, shot_num, dataset, 
                                    args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                    args.combinetype, ablation_down, patience, save_path, \
                                    epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand,\
                                    gamma=args.gamma, basis_matrix=basis_matrix, n_mlp_layer=n_mlp_layer, sampling=args.sampling, model_type=model_type, shared=args.shared)
    else: 
        adapation.adaptation_FUG_graph(model, features, edge_index, labels, test_index, test_batch,\
                                    sparse, idx_test, shot_num, dataset, 
                                    args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                    args.combinetype, ablation_down, patience, save_path, \
                                    epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand,\
                                    gamma=args.gamma, basis_matrix=basis_matrix, n_mlp_layer=n_mlp_layer)
            
elif model_type in ['norm_mdgpt']: 
    if ablation_down == 'None': 
        features = torch.FloatTensor(pca_compression(t_feature.cpu(),k=unify_dim)).cuda()
        if args.downstream_task == 'node': 
            adapation.adaptation_FUG_node(model, features, edge_index, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, \
                                     epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand,\
                                     gamma=args.gamma, basis_matrix=basis_matrix, n_mlp_layer=n_mlp_layer, sampling=args.sampling, model_type=model_type, shared=args.shared)
        else: 
            adapation.adaptation_FUG_graph(model, features, edge_index, labels, test_index, test_batch,\
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, \
                                     epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand,\
                                     gamma=args.gamma, basis_matrix=basis_matrix, n_mlp_layer=n_mlp_layer)

            
    elif ablation_down == 'DEt_finetune':
        adapation.finetune_node(model, t_feature, edge_index, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, \
                                     epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand, gamma=args.gamma, n_mlp_layer=n_mlp_layer, sampling=args.sampling, model_type=model_type)

    elif ablation_down == 'pca_finetune':
        if args.downstream_task == 'node': 
            features = torch.FloatTensor(pca_compression(t_feature.cpu(),k=unify_dim)).cuda()
            adapation.finetune_node(model, features, edge_index, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, \
                                     epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand, gamma=args.gamma, n_mlp_layer=n_mlp_layer, sampling=args.sampling, model_type=model_type)
    else: 
        if args.downstream_task == 'node': 
            adapation.adaptation_FUG_node(model, t_feature, edge_index, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand, a=args.a, 
                                     gamma=args.gamma, basis_matrix=basis_matrix, n_mlp_layer=n_mlp_layer, sampling=args.sampling, model_type=model_type, shared=args.shared)
        else: 
            adapation.adaptation_FUG_graph(model, t_feature, edge_index, labels, test_index, test_batch,\
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, \
                                     epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand,\
                                     gamma=args.gamma, basis_matrix=basis_matrix, n_mlp_layer=n_mlp_layer)

elif model_type == 'sharedFUG' or model_type == 'filterFUG' or model_type == 'filterbank': 
    adapation.adaptation_sharedFUG_node(model_type, model, t_feature, edge_index, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                                     args.combinetype, ablation_down, patience, save_path, epoch=adapta_epoch, sample_size = sample_size, if_rand=args.if_rand, a=args.a, 
                                     gamma=args.gamma, basis_matrix=basis_matrix, n_mlp_layer=n_mlp_layer, sampling=args.sampling, shared=args.shared)

elif model_type == 'barycenter': 
    adapation.adaptation_barycenter_node(model, t_feature, edge_index, labels, \
                            sparse, idx_test, shot_num, dataset, \
                            args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                            args.combinetype, ablation_down, patience, save_path, epoch=adapta_epoch, \
                            sample_size=sample_size, if_rand=args.if_rand, a=args.a, gamma=args.gamma, de_input=args.de_input, 
                            n_mlp_layer=args.n_mlp_layer, sampling=args.sampling, shared=args.shared)
    

