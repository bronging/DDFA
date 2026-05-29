import argparse
import logging
import os
import random
import warnings

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops

import adaptation
import train
from preprompt import *
from utils import process
from utils.dataset import *
from utils.logging_ import *


def set_seed(seed=42):
    random.seed(seed)                       
    np.random.seed(seed)                    
    torch.manual_seed(seed)                

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        
        torch.cuda.manual_seed_all(seed)   
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu(gpu): 
    print("CUDA Available:", torch.cuda.is_available())
    print('gpu:', str(gpu))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.cuda.set_device(gpu)
    device = torch.device("cuda")
    return device

parser = argparse.ArgumentParser("DDFA")

# Experiment
parser.add_argument('--gpu',                type=int,      default=0,       help='GPU device id')
parser.add_argument('--seed',               type=int,      default=39,      help='random seed')
parser.add_argument('--experiment',         type=str,      default='DDFA',  help='experiment name used for file naming')
parser.add_argument('--target_id',          type=int,      default=0,       help='target domain index: [0]Cora [1]Citeseer [2]Pubmed [3]Photo [4]Computers [5]FacebookPagePage [6]LastFMAsia')
parser.add_argument('--skip_pretrain',      type=int,      default=1,       help='load pretrained checkpoint if available (1: skip training)')
parser.add_argument('--shot_num',           type=int,      default=1,       help='number of labeled nodes per class (K-shot)')
parser.add_argument('--test_idx_num',       type=int,      default=1000,    help='number of query nodes for evaluation')
parser.add_argument('--downstream_task',    type=str,      default='node',  help='downstream task type: node or graph')
parser.add_argument('--pretrain_method',    type=str,      default='LP',    help='pre-training method: GRAPHCL or LP or splitLP')
parser.add_argument('--dataset',            type=int,      default=0,       help='dataset combination index')
parser.add_argument('--graphId',            nargs='+',     type=int,        default=[1], help='graph id within a dataset')

# Model Architecture
parser.add_argument('--backbone',           type=str,      default='gcn',   help='GNN backbone type')
parser.add_argument('--hid_units',          type=int,      default=256,     help='hidden dimension of GCN')
parser.add_argument('--num_layers',         type=int,      default=3,       help='number of GCN layers')
parser.add_argument('--unify_dim',          type=int,      default=50,      help='unified feature dimension (k)')
parser.add_argument('--num_de_layers',      type=int,      default=2,       help='number of layers in dimension encoder')
parser.add_argument('--combinetype',        type=str,      default='add',   help='domain token combination type: add or mul')
parser.add_argument('--alpha',              type=float,    default=3.0,     help='weight of Wasserstein barycenter loss')
parser.add_argument('--beta',               type=float,    default=100.0,   help='weight of diversity loss (intra + inter)')

# Pre-training
parser.add_argument('--pre_epochs',         type=int,      default=1000,    help='number of pre-training epochs')
parser.add_argument('--lr',                 type=float,    default=0.0001,  help='pre-training learning rate')
parser.add_argument('--negative_samples_num', type=int,    default=50,      help='number of negative samples for link prediction')
parser.add_argument('--drop_percent',       type=float,    default=0.1,     help='edge drop ratio for graph augmentation')
parser.add_argument('--batch_size',         type=int,      default=1,       help='batch size')

# Adaptation
parser.add_argument('--adapt_epochs',       type=int,      default=400,     help='number of adaptation epochs per episode')
parser.add_argument('--downlr',             type=float,    default=0.0,     help='adaptation learning rate')
parser.add_argument('--l2_coef',            type=float,    default=0.0001,  help='weight decay for adaptation optimizer')
parser.add_argument('--gamma',              type=float,    default=20.0,    help='weight of dimension encoder uniformity loss during adaptation')
parser.add_argument('--temp',               type=float,    default=0.2,     help='softmax temperature for prototype-based classification')

# Dimension Encoder (DE) / Sampling
parser.add_argument('--sample_size',        type=int,      default=256,     help='number of nodes sampled as DE input')
parser.add_argument('--sampling',           type=str,      default='random', help='node sampling strategy: random / degree / feat_norm / front')
parser.add_argument('--if_rand',            type=str2bool, default=False,   help='re-sample randomly at every forward call')

args = parser.parse_args()

warnings.filterwarnings("ignore")

set_seed(args.seed)
torch.cuda.empty_cache()
device = set_gpu(args.gpu)

shot_num = args.shot_num
negative_samples_num = args.negative_samples_num
drop_percent = args.drop_percent
hid_units = args.hid_units
num_layers = args.num_layers
unify_dim = args.unify_dim
target_graph_id = args.graphId
lr = args.lr

experiment = args.experiment 
pre_epochs = args.pre_epochs
num_de_layers = args.num_de_layers 

target_id = args.target_id
sample_size = args.sample_size

test_idx_num = args.test_idx_num

l2_coef = 0.0
sparse = True

        # 0         1          2         3         4                5                6        
data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']

dataset = data[args.target_id]
downstream = data[args.target_id]

pretrain_dataset_names = get_pretrain_dataset_names(data, target_id)
print(pretrain_dataset_names)

logfile, save_dir, result_dir, cache_dir = make_dir(experiment)
save_name, csv_name = get_save_name(args, pretrain_dataset_names, save_dir, result_dir)

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=logfile,
                    filemode='a', 
                    encoding="utf-8",)

log_args_table(args, max_per_line=5, col_width=30)

torch.autograd.set_detect_anomaly(True)

num_pretrain_dataset = len(pretrain_dataset_names)

write(f"✅ PrePromptBarycenter")
write(f'   pretrain-dataset: {pretrain_dataset_names}')
write(f'   Backbone        : {args.backbone}')
write(f'   alpha           : {args.alpha}')
write(f'   beta            : {args.beta}')


model = PrePromptBaryBasis(
    unify_dim=unify_dim,
    hid_units=hid_units,
    num_pretrain_dataset=num_pretrain_dataset,
    num_layers=num_layers,
    dropout=0.1,
    type_=args.combinetype,
    alpha=args.alpha,
    beta=args.beta,
    n_sample=sample_size,
    if_rand=False,
    sampling=args.sampling,
    num_de_layers=args.num_de_layers,
).cuda()


if args.skip_pretrain and os.path.exists(save_name):
    print(args.skip_pretrain)
    print(f'loading model from {save_name}')

    checkpoint = torch.load(save_name)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
   
else:
    pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
    
    features, adjs, edge_indexs = process.get_features_adjs(
        pretrain_loaders, cache_dir, pretrain_dataset_names, target_graph_id
    )
    
    _, _, _, negetive_samples, _ = process.preprocess_dataset_w_DE_pyg(
        features, edge_indexs, drop_percent, negative_samples_num
    )

    if torch.cuda.is_available():
        print('Using CUDA')

        features = [tensors.cuda() for tensors in features]
        edge_indexs = [tensors.cuda() for tensors in edge_indexs]

    train.train_fug_lp(
        model=model,
        features=features,
        edge_indexs=edge_indexs,
        negetive_samples=negetive_samples,
        sparse=sparse,
        save_name=save_name,
        lr=lr,
        weight_decay=l2_coef,
        num_epoch=pre_epochs,
        sample_size=sample_size,
        unify_dim=unify_dim,
        alpha=args.alpha,
    )


write('#'*50)
write(f'Downstream dataset is 🌼⭐ {downstream} ⭐🌼')

downstream_dataset = load_dataset(downstream)
downstream_loader = DataLoader(downstream_dataset)

data = next(iter(downstream_loader))

t_feature, adj = process.process_tu(data, data.x.shape[1])
edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
edge_index = edge_index.cuda()

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
labels = data.y
nb_classes = len(labels.unique())

test_labels = labels[idx_test]
write(f'  nb_classes : {nb_classes}')

if sparse:
    adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
else:
    adj = torch.FloatTensor(adj.todense()).cuda()
t_feature = t_feature.cuda()

print(f'loading model from {save_name}')
checkpoint = torch.load(save_name)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.cuda()

adaptation.adaptation_node(
    model=model,
    features=t_feature,
    edge_index=edge_index,
    labels=labels,
    args=args,
    sparse=sparse,
    idx_test=idx_test,
    nb_classes=nb_classes,
    dataset=dataset,
    downstream=downstream,
    csv_name=csv_name,
)
