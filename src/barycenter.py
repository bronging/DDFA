import numpy as np
import scipy.sparse as sp
import random

from preprompt import PrePrompt, pca_compression, PrePromptFUG, PrePromptACL, PrePromptSharedFUG, PrePromptFilterFUG, FilterbankFUG
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

from models.sinkhorn import * 
from preprompt import * 


write(f"✅ PrePromptBarycenter")
write(f'   pretrain-dataset: {pretrain_dataset_names}')
write(f'   Backbone        : {args.backbone}')
write(f'   de loss         : {args.de_loss}')
write(f'   gamma           : {args.gamma}')
write(f'   shared          : {args.shared}')

model = PrePromptBarycenter(unify_dim, hid_units, nonlinearity, pretrain_dataset_num, 
        layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone, #'norm_mdgpt',
        alpha = args.alpha, ablation = ablation_pre, scaling_factor=3, n_sample=sample_size, 
        if_rand=False,  de_loss=args.de_loss, de_weight=args.de_weight, sampling=args.sampling,
        n_mlp_layer=args.n_mlp_layer, de_input=args.de_input, shared=args.shared).cuda()


try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:
    pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
    
    features, adjs, edge_indexs = process.get_features_adjs(pretrain_loaders,  \
                    cache_dir, pretrain_dataset_names, target_graph_id)

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
        
        aug_adjs = [
            [edge_index.cuda() for edge_index in ei_list]
            for ei_list in aug_adjs
        ]
        
        aug_features = [tensors.cuda() for tensors in aug_features]
        lbls = [tensors.cuda() for tensors in lbls]

    from train import train_fug_barycenter
    bary_Y = train_fug_barycenter(model=model, aug_features=aug_features, aug_adjs=aug_adjs,  
                        lbls=lbls, sparse=sparse, save_name=save_name, \
                        lr=lr, weight_decay=l2_coef, \
                        start_epoch=args.restart_epoch, num_epoch=nb_epochs,  
                        patience=patience, gamma=args.gamma, model_type=model_type, unify_dim=unify_dim, sample_size=sample_size)

write('#'*50)
write(f'Downastream dataset is 🌼⭐ {downstream} ⭐🌼')

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
    nb_classes=len(np.unique(data))
    print('nb_classes', nb_classes)

    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()
    t_feature = t_feature.cuda()


print(f'loading model from {save_name}')
model.load_state_dict(torch.load(save_name))
model = model.cuda()

save_path = save_name + "_apapt"


adapta_epoch = args.adapt_ep

adapation.adaptation_barycenter_node(model, t_feature, edge_index, labels, \
                            sparse, idx_test, shot_num, dataset, \
                            args.beta, hid_units, nb_classes, unify_dim, layers_num,\
                            args.combinetype, ablation_down, patience, save_path, epoch=adapta_epoch, \
                            sample_size=sample_size, if_rand=args.if_rand, a=args.a, gamma=args.gamma, de_input=args.de_input, 
                            n_mlp_layer=args.n_mlp_layer, sampling=args.sampling, shared=args.shared)