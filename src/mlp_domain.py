from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg, FeatureMLP
from preprompt import PrePrompt, pca_compression
from legacy_exp.preprompt_lagacy import PrePromptwithMLPToken
import preprompt
import pdb
import os
import sys
import tqdm
import argparse
from downprompt import downpromptMLP, downprompt_graph, TargetAlignedModel, PrototypeClassifier
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

def get_parent_curr_dir():
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(current_file_path))
    sys.path.append(parent_dir)
    current_dir = os.path.dirname(current_file_path)
    return parent_dir, current_dir

def make_dir(experiment): 
    parent_dir, current_dir = get_parent_curr_dir()
    logfile = os.path.join(current_dir, f'{experiment}_log.txt')
    save_dir = os.path.join(parent_dir, 'checkpoints', experiment)
    result_dir = os.path.join(parent_dir, 'result')
    cache_dir = os.path.join(parent_dir, 'cache')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    return logfile, save_dir, result_dir, cache_dir 

def get_save_name(args, pretrain_dataset_names, save_dir, result_dir): 
    pretrain_dataset_str = ''
    for strs in pretrain_dataset_names: 
        pretrain_dataset_str += '_'+strs
    set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
    save_name = os.path.join(save_dir, f'{set_name}.pkl')
    csv_name = os.path.join(result_dir, f'{set_name}.csv')

    return save_name, csv_name

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes', 'y'):
        return True
    elif v.lower() in ('false', '0', 'no', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def write_rst(accs, microf, macrof):
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
    micro_mean = np.mean(microf)
    macro_mean = np.mean(macrof)
    micro_std = np.std(microf)
    macro_std = np.std(macrof)

    print("-" * 100)
    print(f"[{shot_num}-shot]\nAcc: {acc_mean:.2f} ± {acc_std:.2f}")
    print(f"Macro F1: {macro_mean:.2f} ± {macro_std:.2f}, Micro F1: {micro_mean:.2f} ± {micro_std:.2f}")
    print("-" * 100)
    logging.info('-' * 100)
    logging.info(f"[{shot_num}-shot]")
    logging.info(f"Acc: {acc_mean:.2f} ± {acc_std:.2f}")
    logging.info(f"Macro F1: {macro_mean:.2f} ± {macro_std:.2f}, Micro F1: {micro_mean:.2f} ± {micro_std:.2f}")
    logging.info(f"{'-' * 100}\n")
    

torch.cuda.empty_cache()
parser = argparse.ArgumentParser("SAMGPT")
parser.add_argument('--target_id', type=int, default=5, help='[Cora, Citeseer, Pubmed, Photo, Computers, FacebookPagePage, LastFMAsia]')
parser.add_argument('--dataset', type=str, default='FacebookPagePage', help='target data')
parser.add_argument('--pretrain_datasets', nargs='+', type=str, 
    help='pretrain datasets', default=['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'LastFMAsia'])
parser.add_argument('--downstream_task', type=str, default='node', help='node or graph')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--pretrain_method', type=str, default="GRAPHCL", help='GRAPHCL or LP or splitLP')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')   
parser.add_argument('--graphId', nargs='+', type=int, default=[1], help='target graph\'s id in one dataset')
parser.add_argument('--alpha', type=float, default=1.0, help='alpha of combines')
parser.add_argument('--beta', type=float, default=1.0, help='beta of combines')
parser.add_argument('--negative_samples_num', type=int, default=40, help='negative_samples_num')
parser.add_argument('--skip_pretrain', type=int, default=0, help='try to use trained models')
parser.add_argument('--ablation_pre', type=str, default='all', help='ablation_pre')
parser.add_argument('--ablation_down', type=str, default='all', help='ablation_down')
parser.add_argument('--unify_dim', type=int, default=50, help='unify_dim')
parser.add_argument('--shot_num', type=int, default=1, help='shot_num')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hid_units', type=int, default=256, help='hid_units')
parser.add_argument('--layers_num', type=int, default=3, help='layers_num')
parser.add_argument('--backbone', type=str, default='gcn', help='backbone')
parser.add_argument('--batch_size', type=int, default=1, help='backbone')
parser.add_argument('--restart_epoch', type=int, default=0, help='이어서 학습 시작할 에포크')

parser.add_argument('--mlp_init', type=str2bool, default=False, help='mlp identity init')
parser.add_argument('--separate_learning', type=str2bool, default=True, help='mlp/gcn 학습 같이(f) or 따로(t)')
parser.add_argument('--w1alpha', type=float, default=1.0, help='graphcl loss + w1alpha*w1loss')

parser.add_argument('--experiment', type=str, default='MLP_source0_ep200', help='실험 종류')
parser.add_argument('--nb_epochs', type=int, default=200, help='pretraining epoch')
parser.add_argument('--n_mlp_layer', type=int, default=1, help='num of mlp layers')

args = parser.parse_args()
warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)

shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
aug_type = args.aug_type
drop_percent = args.drop_percent
hid_units = args.hid_units
num_layers_num = args.layers_num
dataset = args.dataset
unify_dim = args.unify_dim
target_graph_id = args.graphId
lr = args.lr

mlp_init = args.mlp_init
experiment = args.experiment 
nb_epochs = args.nb_epochs
n_mlp_layer = args.n_mlp_layer 

#         0         1          2         3         4                5                6        
data = ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia']
dataset = data[args.target_id]
downstream = data[args.target_id]
pretrain_dataset_names = [data[i] for i in range(len(data)) if i != args.target_id]
print(pretrain_dataset_names)

seed = args.seed

print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)
device = torch.device("cuda")

set_seed(seed)

LP = (args.pretrain_method == 'LP')

# nb_epochs = 10000
patience = 50
l2_coef = 0.0
drop_prob = 0.0
sparse = True

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
nonlinearity = 'prelu'  # special name to separate parameters

best = 1e9
best_t = 0
firstbest = 0
cnt_wait = 0
test_idx_num = 100
negetive_sample = torch.tensor(0.0)


print(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1
pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]

logfile, save_dir, result_dir, cache_dir = make_dir(experiment)
save_name, csv_name = get_save_name(args, pretrain_dataset_names, save_dir, result_dir)

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=logfile,
                    filemode='a', 
                    encoding="utf-8",)

logging.info(f"Separate learning: {args.separate_learning}")
if not args.separate_learning: 
    logging.info(f"> alpha: {args.w1alpha}")
logging.info(f"MLP init as identity: {args.mlp_init}")
logging.info(f"# of MLP layers: {n_mlp_layer}")

torch.autograd.set_detect_anomaly(True)

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

model = PrePromptwithMLPToken(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
        num_layers_num, 0.1, type_='add', alpha = args.alpha, ablation = args.ablation_pre, 
        n_mlp_layer=n_mlp_layer, init_identity=mlp_init).cuda()

try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:
    features = []
    adjs = []
    aug_adjs = []
    aug_features = []
    negetive_samples = []
    lbls = []

    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
        if (step+1) not in target_graph_id:
            print(datas)
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            if not(os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt') and \
                os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt') ):
                feature, adj = process.process_tu(data,data.x.shape[1])
                feature = torch.FloatTensor(pca_compression(feature,k=unify_dim))
                torch.save(feature, f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                torch.save(adj, f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            feature, adj = torch.load(f'{cache_dir}/{pretrain_dataset_name}_feature.pt'), \
                torch.load(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            
#GRAPHCL:
            if args.pretrain_method == 'GRAPHCL':
                if not(os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt') and \
                    os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt') and \
                    os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt') ):
                    aug_feature, aug_adj, lbl = aug.build_aug(adj, feature, sparse, drop_percent)
                    torch.save(aug_feature, f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt')
                    torch.save(aug_adj, f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt')
                    torch.save(lbl,  f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')
                aug_feature, aug_adj, lbl = torch.load(f'{cache_dir}/{pretrain_dataset_name}_aug_feature.pt'), \
                torch.load(f'{cache_dir}/{pretrain_dataset_name}_aug_adj.pt'),  \
                torch.load(f'{cache_dir}/{pretrain_dataset_name}_lbl.pt')
                aug_features.append(aug_feature)
                aug_adjs.append(aug_adj)
                lbls.append(lbl)

                del aug_feature, aug_adj, lbl
                gc.collect()
#split_LP:
            if args.pretrain_method == 'splitLP':
                if not os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt'):
                    negetive_sample = preprompt.prompt_pretrain_sample(adj, 50)
                    torch.save(negetive_sample, f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_sample = torch.load(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_samples.append(negetive_sample)

            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            features.append(feature)
            adjs.append(adj)

            del feature, adj
            gc.collect()
#LP:    
        if args.pretrain_method == 'LP':    
            combinedadj = process.combine_dataset_list_sp(adjs)
            print('combinedadj', combinedadj.shape)
            negetive_sample = preprompt.prompt_pretrain_sample(combinedadj, args.negative_samples_num)


    if args.separate_learning: 
        main_optim = torch.optim.Adam(
            [p for name, p in model.named_parameters() if not name.startswith('feature_MLP_layers')],
            lr=lr, weight_decay=l2_coef
        )
        mlp_optim = torch.optim.Adam(model.feature_MLP_layers.parameters(), lr=lr)
    else: 
        w1alpha = args.w1alpha
        full_optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

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
        
    if args.pretrain_method == 'GRAPHCL':
        dataset = PretrainDatasetAug(aug_features, aug_adjs, lbls)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        del aug_adjs, aug_features, lbls 
        gc.collect()

    for epoch in range(args.restart_epoch, nb_epochs):
        set_seed(seed)
        
        model.train()
        
        total_loss = 0
        total_mlp_loss = 0 
        
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch} ")
        for domain_idx, batch in enumerate(batch_bar): 
#GRAPHCL
            if args.pretrain_method == 'GRAPHCL':
                loss = model(batch['feature'], batch['adj'], sparse, None, domain_idx, None, None, batch['lbls'], None, )
#LP:    
            if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
                loss =  model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)
             
            if sparse: 
                adj = batch['adj'][0, 0]
            else:
                adj = torch.FloatTensor(batch['adj'][0, 0].todense())
 
            if domain_idx == 0: 
                domain0_feat = batch['feature'][0,0]
                domain0_adj = adj 
                mlp_loss = torch.tensor(0.0)

            if args.separate_learning: 
                main_optim.zero_grad()
                loss.backward()
                main_optim.step()
                total_loss += loss.item()

                if domain_idx != 0: 
                    prev_emb, _ = model.embed(domain0_feat, domain0_adj, sparse, None, domain_idx, LP)
                    prev_emb = prev_emb.detach()     
                    curr_emb, _ = model.embed(batch['feature'][0, 0], adj, sparse, None, domain_idx, LP)
                    mlp_loss = sliced_wasserstein_torch(prev_emb.squeeze(0), curr_emb.squeeze(0))

                    mlp_optim.zero_grad()
                    mlp_loss.backward()
                    total_mlp_loss += mlp_loss.item()

            else: 
                if domain_idx != 0: 
                    prev_emb, _ = model.embed(domain0_feat, domain0_adj, sparse, None, LP, domain_idx)
                    prev_emb = prev_emb.detach()     
                    curr_emb, _ = model.embed(batch['feature'][0, 0], adj, sparse, None, LP, domain_idx)
                    mlp_loss = sliced_wasserstein_torch(prev_emb.squeeze(0), curr_emb.squeeze(0))

                t_loss = loss + w1alpha * mlp_loss
                full_optim.zero_grad()
                t_loss.backward()
                full_optim.step()

                total_loss += t_loss.item()
                total_mlp_loss += mlp_loss.item()

            if args.separate_learning:     
                batch_bar.set_postfix(loss=loss.item(), mlp_loss=mlp_loss.item())
            else: 
                batch_bar.set_postfix(loss=t_loss.item(), mlp_loss=mlp_loss.item())

        if total_loss < best:
            firstbest = 1
            best = total_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
    logging.info(f"best epoch: {best_t}")

print('#'*50)
print('PreTrain datasets are ', pretrain_dataset_names)
print('Downastream dataset is ', args.dataset)
# logging.info('#'*50)
logging.info('PreTrain datasets are ')
logging.info(pretrain_dataset_names)
logging.info('Downastream dataset is ')
logging.info(f"✅ {downstream}")


domain0_feat = torch.load(f'{cache_dir}/{pretrain_dataset_names[0]}_aug_feature.pt')[0].cuda()
domain0_adj = torch.load(f'{cache_dir}/{pretrain_dataset_names[0]}_aug_adj.pt')[0]
if sparse: 
    domain0_adj = domain0_adj.cuda()
else: 
    domain0_adj = torch.FloatTensor(domain0_adj.todense())

downstream_dataset = load_dataset(downstream)
print(downstream_dataset)
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

feature_mlp = FeatureMLP(in_dim=unify_dim, hidden_dim=unify_dim, out_dim=unify_dim, num_layer=2, init_identity=mlp_init)
mlp_optim = torch.optim.Adam(feature_mlp.parameters(), lr=1e-1)
feature_mlp.cuda()


model.eval()# freeze GNN
with torch.no_grad():
    anchor_emb, _ = model.embed(domain0_feat, domain0_adj, sparse, None, LP, graphid=0)
    anchor_emb = anchor_emb.squeeze(0).detach()

w1_step = 200
feature_mlp.train()
for epoch in range(1, w1_step + 1):
    x_trans = feature_mlp(features)

    target_emb, _ = model.embed(x_trans, adj, sparse, None, LP, graphid=0)

    # sliced Wasserstein 거리 계산
    loss = sliced_wasserstein_torch(anchor_emb, target_emb, n_proj=100) #+ alpha * F.mse_loss(x_trans[mask], t_data.x[mask])
    
    mlp_optim.zero_grad()
    loss.backward()
    mlp_optim.step()

    if epoch % 50 == 0:
        print(f"[Epoch {epoch}] W1 loss: {-loss.item():.4f} (W1={loss:.4f})")


downstreamlrlist = [0.001]
test_embs = target_emb[0, idx_test]

shotnum = 1
accs, macrof, microf = [], [], []
downstream_lower = downstream.lower()
fea_domain_weights = model.get_weights()
combines = [args.alpha, args.beta]

feature_mlp.eval()
tot = torch.zeros(1)
tot = tot.cuda()
accs = []
macrof = []
microf = []
cnt_wait = 0
best = 1e9
best_t = 0

for i in tqdm(range(100)):
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

    log = downpromptMLP(ft_in=hid_units, nb_classes=nb_classes, feature_dim=unify_dim, 
                               target_mlp=feature_mlp, fea_domain_weights=fea_domain_weights,
                               combines=combines)
    log.train()

    pretrain_embs = target_emb[0, idx_train]
    opt = torch.optim.Adam(log.parameters(), lr=lr)
    log = log.cuda()
    best = 1e9
    best_acc = torch.zeros(1).cuda()

    for _ in tqdm(range(400)):
        opt.zero_grad()
        if  args.downstream_task == 'graph':
            logits = log(features,adj,sparse,model.gcn,idx_train,batch_train,lbls_train,1).float().cuda()
        else:
            logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1).float().cuda()
        loss = xent(logits, lbls_train)
        if loss < best:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            #print('Early stopping!')
            break

        loss.backward()
        opt.step()
        #tqdm.write(f"Epoch {i+1} | Loss: {loss.item():.4f}")

    if  args.downstream_task == 'graph':
        logits = log(features, adj, sparse, model.gcn, test_index, test_batch)
    else:
        logits = log(features, adj, sparse, model.gcn, idx_test)
    preds = torch.argmax(logits, dim=1).cuda()
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    preds_cpu = preds.cpu().numpy()
    test_lbls_cpu = test_lbls.cpu().numpy()
    micro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='micro')
    macro_f1 = f1_score(test_lbls_cpu, preds_cpu, average='macro')
    microf.append(micro_f1 * 100)
    macrof.append(macro_f1 * 100)
    accs.append(acc * 100)
    tot += acc
    tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
    if i %10  == 0: 
        accs_tensor = torch.stack(accs)
        logging.info(f'[{i}]{accs_tensor.mean().item():.2f} ± {accs_tensor.std().item():.2f}')
        print(f'[{i}]{accs_tensor.mean().item():.2f} ± {accs_tensor.std().item():.2f}')

print('-' * 100)
print('Average accuracy:[{:.4f}]'.format(tot.item() / 100))
accs_tensor = torch.stack(accs)
acc_mean = accs_tensor.mean().item()
acc_std = accs_tensor.std().item()
microf_mean = sum(microf) / len(microf)
macrof_mean = sum(macrof) / len(macrof)
microf_std = torch.std(torch.tensor(microf)).item()
macrof_std = torch.std(torch.tensor(macrof)).item() 
print('Mean:[{:.4f}]'.format(acc_mean))
print('Std :[{:.4f}]'.format(acc_std))
print('-' * 100)
logging.info('-' * 100)
logging.info(f'Acc:[{accs_tensor.mean().item():.2f} ± {accs_tensor.std().item():.2f}]')
logging.info('-' * 100)
