from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg, FeatureMLP, DimensionNN_V2, GCN_encoder, FUG
from preprompt import PrePrompt, pca_compression, PrePromptNorm, PrePromptFUG,  sliced_wasserstein_torch
import preprompt
import pdb
import os
import sys
import tqdm
import argparse
from downprompt import downpromptFUG, downprompt, downprompt_graph, TargetAlignedModel, PrototypeClassifier
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
from utils.figure import * 
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
parser.add_argument('--de_loss', type=float, default=1.0, help='dim encoder loss 얼마나 반영할지')
parser.add_argument('--a', type=float, default=0.0, help='adatation 때 open dim encoder loss 얼마나 반영할지')

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
data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']
# data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'Texas', 'Cornell', 'Wisconsin', 'chameleon', 'squirrel']

# data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'Reddit']

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

def dimensional_sample_random(sample_size, x, edge_index, if_rand=False):
    with torch.no_grad():
        if if_rand != True:
            d_sample_matrix = x[:sample_size, :]
        else:
            d_sample_matrix = x[torch.randperm(x.shape[0]),:][:sample_size, :]
        return d_sample_matrix

sample_size = 183

if model_type == 'samgpt': 
    write(f"✅ SAMGPT - GraphCL")
    write(f'pretrain-dataset: {pretrain_dataset_names}')

    model = PrePrompt(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
            num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
            alpha = args.alpha, ablation = ablation_pre).cuda()
    
elif model_type == 'norm_mdgpt': 
    write(f"✅ Norm MDGPT - GraphCL")
    write(f'pretrain-dataset: {pretrain_dataset_names}')
    write(f'de loss: {args.de_loss}')
    model = PrePromptFUG(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
            num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone, #'norm_mdgpt',
            alpha = args.alpha, ablation = ablation_pre, scaling_factor=3, n_sample=sample_size, if_rand=False,  de_loss=args.de_loss).cuda()
      

perm = False 


try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:
    pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
    
    features, adjs, edge_indexs = process.get_features_adjs(pretrain_loaders,  \
                       cache_dir, pretrain_dataset_names, target_graph_id)
    
    

    aug_features, aug_adjs, lbls, negetive_samples, combinedadj = process.preprocess_dataset_w_DE(
                                                                    features, adjs, pretrain_method, \
                                                                    sparse, drop_percent, negative_samples_num)
    
    print('Aug feat[0]: ', aug_features[0].size())


    set_seed(seed)

    if torch.cuda.is_available():
        print('Using CUDA')

        features = [tensors.cuda() for tensors in features]
        edge_indexs = [tensors.cuda() for tensors in edge_indexs]

        adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).cuda()  if sparse else torch.FloatTensor(adj.todense()).cuda() 
            for adj in adjs]
        
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()

        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]
        lbls = [tensors.cuda() for tensors in lbls]
    
    if model_type == 'samgpt' or model_type == 'norm_mdgpt':
        if pretrain_method == 'GRAPHCL':
            train.train_fug_graphcl(model=model, lr=lr, weight_decay=l2_coef, 
                            start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                            aug_features=aug_features, aug_adjs=aug_adjs, 
                            lbls=lbls, sparse=sparse, save_name=save_name, patience=patience)

pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

features, adjs, edge_indexs = process.get_features_adjs(pretrain_loaders,  \
                    cache_dir, pretrain_dataset_names, target_graph_id)

labels = process.get_labels(pretrain_loaders, pretrain_dataset_names, target_graph_id)


aug_features, aug_adjs, lbls, negetive_samples, combinedadj = process.preprocess_dataset_w_DE(
                                                                features, adjs, pretrain_method, \
                                                                sparse, drop_percent, negative_samples_num)

set_seed(seed)

if torch.cuda.is_available():
    print('Using CUDA')

    features = [tensors.cuda() for tensors in features]
    edge_indexs = [tensors.cuda() for tensors in edge_indexs]

    adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).cuda()  if sparse else torch.FloatTensor(adj.todense()).cuda() 
        for adj in adjs]
    
    negetive_samples = [tensors.cuda() for tensors in negetive_samples]

    if len(negetive_samples) == 0:
        negetive_samples = negetive_sample.cuda()

    aug_adjs = [tensors.cuda() for tensors in aug_adjs]
    aug_features = [tensors.cuda() for tensors in aug_features]
    lbls = [tensors.cuda() for tensors in lbls]

    labels = [tensors.cuda() for tensors in labels]


model.eval()

arr = [] 
class_labels = []
for i in range(len(aug_features)):
    print(i)                        
    arr.append(model.get_forward(aug_features[i][0], aug_adjs[i][0], sparse, None, False, i))
    class_labels.append(labels[i])#TODO label )  # shape: [N_i]
tsne_per_domain_w_class(arr, class_labels, pretrain_dataset_names, args.backbone) 



# arr = [] 
# for i in range(len(aug_features)):
#     print(i)                   
#     arr.append(model.get_forward(aug_features[i][0], aug_adjs[i][0], sparse, None, False, i))

arr = model.get_token(features) # GCN 통과 전     
plot_similarity_heatmap(arr, pretrain_dataset_names)

# loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None) 

# for idx, layer in enumerate(model.dimension_encoder_layers): 
#     print(f'⭐⭐{[pretrain_dataset_names[idx]]}')
#     print(layer.basis_matrix().size())
#     # 저장
#     plot_heatmap(layer.basis_matrix().detach(), f"⭐ {pretrain_dataset_names[idx]} Basis", f"T_{downstream}_S_{pretrain_dataset_names[idx]}_heatmap.png")
#     # plot_tsne(layer.basis_matrix().detach(), f"⭐ {pretrain_dataset_names[idx]} Basis t-SNE", f"T_{downstream}_S_{pretrain_dataset_names[idx]}_tsne.png")
#     # write(f'{pretrain_dataset_names[idx]} => {layer.basis_matrix().mean(dim=0).pow(2)}')

#     print('\n')
        
# write('#'*50)
# write('PreTrain datasets are ')
# write(pretrain_dataset_names)
# write('Downastream dataset is ')
# write(f"✅ {downstream}")

# downstream_dataset = load_dataset(downstream)
# downstream_loader = DataLoader(downstream_dataset)

# for data in downstream_loader:
#     print(data)
#     t_feature, adj= process.process_tu(data,data.x.shape[1])
#     print('process done')
#     edge_index = data.edge_index.cuda()
#     # features = torch.FloatTensor(pca_compression(t_feature,k=unify_dim)).cuda()
#     adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
#     idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
#     labels = data.y
#     data=np.array(data.y)
#     np.unique(data)
#     nb_classes=len(np.unique(data))
#     print('nb_classes', nb_classes)
#     if args.downstream_task == 'graph':
#         test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
#         test_index = test_subgraph['idx'].cuda()
#         test_batch = test_subgraph['batch'].cuda()
#     if sparse:
#         adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
#     else:
#         adj = torch.FloatTensor(adj.todense()).cuda()
#     t_feature = t_feature.cuda()

# print(f'loading model from {save_name}')
# model.load_state_dict(torch.load(save_name))
# model = model.cuda()
 
# save_path = save_name + "_apapt"

# if model_type == 'samgpt' or model_type == 'norm_mdgpt': 
#     if ablation_down == 'None': 
#         adapation.adaptation_FUG_samgpt_node(model, t_feature, adj, labels, \
#                                      sparse, idx_test, shot_num, dataset, 
#                                      args.beta, hid_units, nb_classes, unify_dim, num_layers_num,\
#                                      args.combinetype, ablation_down, 1, save_path, epoch=100, sample_size = sample_size, if_rand=args.if_rand)
#     else: 
#         adapation.adaptation_FUG_samgpt_node(model, t_feature, adj, labels, \
#                                      sparse, idx_test, shot_num, dataset, 
#                                      args.beta, hid_units, nb_classes, unify_dim, num_layers_num,\
#                                      args.combinetype, ablation_down, patience, save_path, epoch=100, sample_size = sample_size, if_rand=args.if_rand, a=args.a)


# dim_pretext_weights, fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
# combines.append(args.beta)

# log = downpromptFUG(hid_units, nb_classes, unify_dim, num_layers_num, dim_pretext_weights,
#                     fea_pretext_weights, str_pretext_weights, combines, args.combinetype,
#                     ablation_down, sample_size, args.if_rand).cuda()
# log.load_state_dict(torch.load(save_path))

# log = log.cuda()
# shotnum = 1

# log.eval()
# logits = log(t_feature, adj, sparse, model.gcn, idx_test)

# print(f'log - oepn prompt: ')
# print(log.downstreamPrompt.open_prompt_dim.basis_matrix().size())
# # plot_heatmap(log.downstreamPrompt.open_prompt_dim.basis_matrix().detach(), f"⭐ {downstream} Open basis", f"T_{downstream}_heatmap.png")
# # plot_tsne(log.downstreamPrompt.open_prompt_dim.basis_matrix().detach(), f"⭐ {downstream} Open Basis t-SNE", f"T_{downstream}_tsne.png")
# print()

# print(f'log - composed prompt')
# write(f'⭐ {downstream} composed prompt')
# write(f'{pretrain_dataset_names}')
# write(log.downstreamPrompt.composedprompt_dim.prompt.weight)


# print()
