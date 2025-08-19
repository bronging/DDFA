from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg, FeatureMLP
from preprompt import PrePrompt, PrePromptNorm, PrePrompt2, pca_compression, PrePromptwithMLP, MDGPTwithPerm, sliced_wasserstein_torch
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
data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']
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
                    level=logging.DEBUG,
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
            alpha = args.alpha, ablation = ablation_pre).cuda()
    

elif model_type == 'anchor_mlp':
    write(f"Separate learning: {args.separate_learning}")
    write(f"> alpha: {args.w1alpha}") if not args.separate_learning else None 
    write(f"MLP init as identity: {args.mlp_init}")
    write(f"# of MLP layers: {n_mlp_layer}")
    write(f"MLP bias: {mlp_bias}")
    write(f"source_domain: [{source_id}], ✅{pretrain_dataset_names[0]}")

    model = PrePromptwithMLP(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
        num_layers_num, 0.1, alpha = args.alpha, ablation = ablation_pre, 
        n_mlp_layer=n_mlp_layer, init_identity=mlp_init, mlp_bias=mlp_bias).cuda()
    
elif model_type == 'vec2mlp':
    # ablation_pre, ablation_down = 'ft', 'fo' 
    # fo: open 만 
    # fc: mixing 만
    # ft: open, mixing 짬뽕 
    # None: token, mlp X 
    model = PrePrompt2(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
            num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
            alpha = args.alpha, ablation = ablation_pre,
            n_mlp_layer=n_mlp_layer, init_identity=mlp_init, mlp_bias=mlp_bias).cuda() 

elif model_type == 'permMDGPT': 
    perm_hid_dim=128
    perm_n_layers=2

    model = MDGPTwithPerm(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
            num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
            alpha = args.alpha, ablation = ablation_pre, 
            perm_hid_dim=perm_hid_dim, perm_n_layers=perm_n_layers, mlp_init=mlp_init).cuda()
    
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:
    pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
    features, adjs, aug_features, aug_adjs, \
        lbls, negetive_samples, combinedadj = process.preprocess_dataset(
                                                pretrain_loaders, pretrain_method, 
                                                cache_dir, pretrain_dataset_names, target_graph_id, 
                                                sparse, drop_percent, negative_samples_num, unify_dim)
    # 이어서 학습 
    if args.restart_epoch != 0: 
        model.load_state_dict(torch.load(save_name))

    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()
        
        features = [tensors.cuda() for tensors in features]
        adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).cuda()  if sparse else torch.FloatTensor(adj.todense()).cuda() 
            for adj in adjs]
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]

        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()

        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]
        lbls = [tensors.cuda() for tensors in lbls]


    if model_type == 'samgpt' or model_type == 'norm_mdgpt' :
        if pretrain_method == 'GRAPHCL':
            train.train_samgpt_graphcl(model=model, lr=lr, weight_decay=l2_coef, 
                            start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                            aug_features=aug_features, aug_adjs=aug_adjs, 
                            lbls=lbls, sparse=sparse, save_name=save_name, patience=patience)
            
        elif pretrain_method == 'LP' or pretrain_method == 'splitLP':
            train.train_samgpt_lp(model=model, lr=lr, weight_decay=l2_coef, 
                            start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                            features=features, adjs=adjs, negetive_samples=negetive_samples, 
                            sparse=sparse)
    
    elif model_type == 'permMDGPT': 
            if args.separate_learning == True: # GCN 튜닝 -> MLP 튜닝  GraphCL loss + MLP loss 
                train.train_samgpt_graphcl_separate(model=model, lr=lr, weight_decay=l2_coef, 
                            start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                            aug_features=aug_features, aug_adjs=aug_adjs, LP=LP, 
                            lbls=lbls, sparse=sparse, save_name=save_name, patience=patience, w1alpha=args.w1alpha)
            else: # GraphCL + W1 loss 
                if args.mlpalpha == 0: # GraphCL loss 만 
                    train.train_permMDGPT_graphcl(model=model, lr=lr, weight_decay=l2_coef, 
                            start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                            aug_features=aug_features, aug_adjs=aug_adjs, 
                            lbls=lbls, sparse=sparse, save_name=save_name, patience=patience)
                else: # GraphCL loss + (a*W1 loss) + MLP loss 
                    train.train_permMDGPT_graphcl_mlp_w1(model=model, lr=lr, weight_decay=l2_coef, 
                                start_epoch=args.restart_epoch, num_epoch=nb_epochs, 
                                aug_features=aug_features, aug_adjs=aug_adjs, LP=LP,
                                lbls=lbls, sparse=sparse, save_name=save_name, patience=patience, w1alpha=args.w1alpha, mlpalpha=args.mlpalpha)
                
    

        
write('#'*50)
write('PreTrain datasets are ')
write(pretrain_dataset_names)
write('Downastream dataset is ')
write(f"✅ {downstream}")

downstream_dataset = load_dataset(downstream)
downstream_loader = DataLoader(downstream_dataset)

for data in downstream_loader:
    print(data)
    features,adj= process.process_tu(data,data.x.shape[1])
    print('process done')
    features = torch.FloatTensor(pca_compression(features,k=unify_dim)).cuda()
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

print(f'loading model from {save_name}')
model.load_state_dict(torch.load(save_name))
model = model.cuda()

save_path = save_name + "_apapt"

if model_type == 'samgpt': 
    # target random permutation
    # Best 는 저장 
    
    num_permutations = 50
    acc_list = [] 
    before_dist_list = []
    after_dist_list = []  
    perm_list = [] 
    random_seed_list = random.sample(range(1, 10**6), num_permutations)
    print(random_seed_list)
    for n in range(num_permutations): 
        # 1. permutation 
        set_seed(random_seed_list[n])
        idx_perm = np.random.permutation(features.shape[1])
        permuted_features = features[:, idx_perm]
        perm_list.append(idx_perm)
        print(f'[{n}]: {idx_perm}')


        # 2. Adaptation + evaluation 
        log, acc = adapation.adaptation_samgpt_node_w_acc(model, permuted_features, adj, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                                     args.combinetype, ablation_down, patience, n)
        
        acc_list.append(acc.item())

        # # 3. 분포 차이 계산
        # w = [] 
        # w2 = [] 

        # emb = model.embed(features, adj, sparse, None, LP)[0].squeeze(0)
        # token_emb = log.get_emb(features, model.gcn, adj, sparse).squeeze(0)

        # for i in range(6): 
        #     w1dist = sliced_wasserstein_torch(emb, token_emb[i])
        #     w2dist = sliced_wasserstein_torch(token_emb, token_emb[i])
        #     w.append(w1dist.item())
        #     w2.append(w2dist.item())
        
        # write(f'Before mean W1: {np.mean(w):.6f}')
        # write(f'After  mean W1: {np.mean(w2):.6f}\n')
        
        # before_dist_list.append(w)
        # after_dist_list.append(w2)

    write('=========================================================')
    write(f'Mean ACC: {np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}')
    write(f'Best of {num_permutations}: {max(acc_list):.2f}')
    write('=========================================================')

    # torch.save(before_dist_list, f'{save_path}_PERM_before_unifying.ptr')
    # torch.save(after_dist_list, f'{save_path}_PERM_after_unifying.ptr')
    # torch.save(acc_list, f'{save_path}_PERM_acc.ptr')
    # torch.save(perm_list, f'{save_path}_PERM_idx_list.ptr')

elif model_type == 'anchor_mlp':
    domain0_feat, domain0_adj = get_anchor_dataset()

    feature_mlp = FeatureMLP(in_dim=unify_dim, hidden_dim=unify_dim, out_dim=unify_dim, 
                            num_layer=n_mlp_layer, init_identity=mlp_init, mlp_bias=mlp_bias).cuda()

    w1_step = 200
    
    adapation.adaptation_anchor_mlp_w1(w1_step, features, adj, sparse, LP, \
                             domain0_feat, domain0_adj, \
                             preprompt, model, feature_mlp)

    frozen_model = TargetAlignedModel(pretrained_model=model, featuremlp=feature_mlp)
    frozen_model.eval()
    with torch.no_grad():
        target_embeds = frozen_model(features, adj, sparse, LP).squeeze(0)

    shotnum = 1
    accs, macrof, microf = [], [], []
    downstream_lower = downstream.lower()

    for i in range(100):
        if  args.downstream_task == 'graph':
            idx_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/idx.pt".
                format(downstream_lower,shotnum,downstream_lower,i)).type(torch.long).cuda()
            
            batch_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/batch.pt".
                format(downstream_lower,shotnum,downstream_lower,i)).type(torch.long).cuda()

            lbls_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/labels.pt".
                format(downstream_lower,shotnum,downstream_lower,i)).type(torch.long).squeeze().cuda()
                    
        else:
            idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".
                format(downstream_lower,shotnum,downstream_lower,i)).type(torch.long).cuda()

            lbls_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".
                format(downstream_lower,shotnum,downstream_lower,i)).type(torch.long).squeeze().cuda()
        
        test_lbls = labels[idx_test].cuda()

        classifier = PrototypeClassifier(nb_classes=nb_classes, hidden_dim=hid_units).cuda()
        classifier.update_prototypes(target_embeds, idx_train, lbls_train)

        logits = classifier(target_embeds, idx_test)
        preds = torch.argmax(logits, dim=1)

        acc = (preds == test_lbls).float().mean().item() * 100
        micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro') * 100
        macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro') * 100

        accs.append(acc)
        microf.append(micro)
        macrof.append(macro)

    write_rst(accs, microf, macrof)

elif model_type == 'vec2mlp':
    adapation.adaptation_vec2mlp_node(model, features, adj, labels, \
                                      sparse, idx_test, shot_num, dataset, 
                                      args.beta, hid_units, nb_classes, unify_dim,\
                                     args.combinetype, ablation_down, patience,
                                      n_mlp_layer, mlp_init, mlp_bias, mixing_mlp, save_path)

elif model_type == 'permMDGPT':
    if args.down_mlp == False: 
        adapation.adaptation_samgpt_node(model, features, adj, labels, \
                                     sparse, idx_test, shot_num, dataset, 
                                     args.beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                                     args.combinetype, ablation_down, patience, save_path)
    else:
        adapation.adaptation_permMDGPT_node(model, features, adj, labels, \
                                            sparse, idx_test, shot_num, dataset, 
                                            args.beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                                            args.combinetype, ablation_down, patience, save_path,
                                            perm_hid_dim=perm_hid_dim, perm_n_layers=perm_n_layers, mlp_init=mlp_init)
