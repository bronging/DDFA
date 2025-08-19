from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg
from preprompt import PrePrompt, pca_compression
import preprompt
import pdb
import os
import sys
import tqdm
import argparse
from downprompt import downprompt, downprompt_graph, SharedPermutaLayer, PermutaLayer
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("SAMGPT")
import torch.nn.functional as F
import torch
import logging
from utils.dataloader import PretrainDatasetAug
from torch.utils.data import DataLoader

torch.cuda.empty_cache()
parser.add_argument('--dataset', type=str, default="Cora", help='data')
parser.add_argument('--pretrain_datasets', nargs='+', type=str, 
    help='pretrain datasets', default=['Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia'])
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
parser.add_argument('--experiment', type=str, default='PermConv', help='실험 종류')
parser.add_argument('--permute', type=bool, default=True, help='premute or not')
parser.add_argument('--restart_epoch', type=int, default=0, help='이어서 학습 시작할 에포크')


args = parser.parse_args()
import warnings
warnings.filterwarnings("ignore")
print('-' * 100)
print(args)
print('-' * 100)
shot_num = args.shot_num
pretrain_dataset_names = args.pretrain_datasets
aug_type = args.aug_type
drop_percent = args.drop_percent

pretrain_dataset_str = ''
for strs in pretrain_dataset_names:
    pretrain_dataset_str += '_'+strs

print("CUDA Available:", torch.cuda.is_available())
print('gpu:', str(args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(args.gpu)

seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)
current_dir = os.path.dirname(current_file_path)

from torch_geometric.loader import DataLoader
from utils.dataset import *
from utils import process
from utils import aug
import gc 

# nb_epochs = 10000
nb_epochs = 400
patience = 50
lr = args.lr
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.hid_units
sparse = True
LP = (args.pretrain_method == 'LP')
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
nonlinearity = 'prelu'  # special name to separate parameters
dataset = args.dataset
device = torch.device("cuda")
best = 1e9
best_t = 0
firstbest = 0
cnt_wait = 0
num_layers_num = args.layers_num
features = []
adjs = []
aug_adjs = []
aug_features = []
negetive_samples = []
lbls = []
negetive_sample = torch.tensor(0.0)

print(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names)
num_pretrain_dataset_num = len(pretrain_dataset_names) + len(args.graphId) - 1
pretrain_loaders = [DataLoader(load_dataset(dataset)) for dataset in pretrain_dataset_names]
unify_dim = args.unify_dim
logfile = os.path.join(current_dir, 'log.txt')
save_dir = os.path.join(parent_directory, 'checkpoints')
save_dir = os.path.join(save_dir, args.experiment)
result_dir = os.path.join(parent_directory, 'result')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'
set_name = f'model__{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=logfile,
                    filemode='a')
logging.info(save_dir)

model = PrePrompt(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
        num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
        alpha = args.alpha, ablation = args.ablation_pre, permute=args.permute).cuda()

test_idx_num = 100
target_graph_id = args.graphId
try:
    print(args.skip_pretrain)
    assert args.skip_pretrain == 1, 'try to use trained models'
    print(f'loading model from {save_name}')
    model.load_state_dict(torch.load(save_name))
except:
    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
        if (step+1) not in target_graph_id:
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            print(pretrain_dataset_name)
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
# #split_LP:
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


    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    # 이어서 학습 
    if args.restart_epoch != 0: 
        model.load_state_dict(torch.load(save_name))

    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()

        if args.pretrain_method != 'GRAPHCL':
            features = [tensors.cuda() for tensors in features]
            adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).cuda()  if sparse else torch.FloatTensor(adj.todense()).cuda() 
                for adj in adjs]
        # lbls = [tensors.cuda() for tensors in lbls]
        
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()
        # aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        # aug_features = [tensors.cuda() for tensors in aug_features]
    
    if args.pretrain_method == 'GRAPHCL':
        dataset = PretrainDatasetAug(aug_features, aug_adjs, lbls)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        del aug_adjs, aug_features, lbls 
        gc.collect()
    
    for epoch in range(args.restart_epoch, nb_epochs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model.train()
        
        total_loss =0 
        print(f"\n Epoch {epoch}/{nb_epochs}")

        for domain_idx, batch in enumerate(tqdm(dataloader, desc="Pretraining")): 
            feat = batch['feature'].cuda()
            adj = batch['adj'].cuda()
            lbls = batch['lbls'].cuda()
        
            #GRAPHCL
            if args.pretrain_method == 'GRAPHCL':
                loss = model(feat, adj, sparse, None, None, None, lbls, None, domain_id=domain_idx)
            #LP:    
            if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
                loss =  model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        print('Loss:[{:.6f}]'.format(total_loss))
        if total_loss < best:
            firstbest = 1
            best = total_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
            logging.info(best_t)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))



print('#'*50)
print('PreTrain datasets are ', pretrain_dataset_names)
print('Downastream dataset is ', args.dataset)
logging.info('#'*50)
logging.info('PreTrain datasets are ')
logging.info(pretrain_dataset_names)
logging.info('Downastream dataset is ')
logging.info(args.dataset)

downstream_dataset = load_dataset(args.dataset)
print(downstream_dataset)
downstream_loader = DataLoader(downstream_dataset)
for data in downstream_loader:
    print(data)
    features,adj= process.process_tu(data,data.x.shape[1])
    print('process done')
    features = torch.FloatTensor(pca_compression(features,k=unify_dim)).cuda() # pca로 차원 축소 
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    idx_test = range(int(data.y.shape[0] - test_idx_num), data.y.shape[0])
    labels = data.y
    data=np.array(data.y)
    np.unique(data)
    nb_classes=len(np.unique(data))
    print('nb_classes', nb_classes)
    if args.downstream_task == 'graph':
        from downprompt import downprompt_graph as downprompt
        test_subgraph = process.build_subgraph(adj.todense().A, torch.tensor(idx_test), False)
        test_index = test_subgraph['idx'].cuda()
        test_batch = test_subgraph['batch'].cuda()
    else:
        from downprompt import downprompt
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj).cuda()
    else:
        adj = torch.FloatTensor(adj.todense()).cuda()

print(f'loading model from {save_name}')
model.load_state_dict(torch.load(save_name))
model = model.cuda()
embeds, _ = model.embed(features, adj, sparse, None, LP)
downstreamlrlist = [0.001]
test_embs = embeds[0, idx_test]
exp_iter = 5

for downstreamlr in downstreamlrlist:

    test_lbls = labels[idx_test].cuda()
    accs = []
    macrof = []
    microf = []
    print('-' * 100)

    for shotnum in range(1,shot_num+1):
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        macrof = []
        microf = []
        
        cnt_wait = 0
        best = 1e9
        best_t = 0
        print("shotnum",shotnum)
        for i in tqdm(range(exp_iter)):
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()

            combines.append(args.beta)

            log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                            fea_pretext_weights, str_pretext_weights, combines, args.combinetype,
                            args.ablation_down).cuda()

            log.train()

            perm = PermutaLayer(unify_dim)
            perm.train()

            if  args.downstream_task == 'graph':
                idx_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/idx.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()
                
                batch_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/batch.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()

                lbls_train = torch.load("data/fewshot_{}_graph/{}-shot_{}/{}/labels.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()
                
            else:
                idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()

                lbls_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".
                    format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()

            pretrain_embs = embeds[0, idx_train]
            # opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)

            params = list(perm.parameters()) + list(log.parameters())
            opt = torch.optim.Adam(params, lr=downstreamlr)
            log = log.cuda()
            best = 1e9
            best_acc = torch.zeros(1).cuda()

            perm = perm.cuda()


            for _ in range(400):
                # perm layer 학습 

                # prompting 
                opt.zero_grad()

                #features, perm_mat = perm(features) # perm layer 학습 

                if  args.downstream_task == 'graph':
                    logits, perm_mat = log(features,adj,sparse,model.gcn,idx_train,batch_train,lbls_train,1).float().cuda()
                else:
                    logits, perm_mat = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1)
                    logits = logits.float().cuda()
                    perm_mat = perm_mat.cuda()
                
                loss = xent(logits, lbls_train)
                loss2 = compute_kl_loss(perm_mat, prior="identity")

                total_loss = loss + 1 * loss2

                if total_loss < best:
                    best = total_loss
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    #print('Early stopping!')
                    break

                total_loss.backward()
                opt.step()

            if  args.downstream_task == 'graph':
                logits, perm_mat = log(features, adj, sparse, model.gcn, test_index, test_batch)
            else:
                logits, perm_mat = log(features, adj, sparse, model.gcn, idx_test)
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

        print('-' * 100)
        print('Average accuracy:[{:.4f}]'.format(tot.item() / exp_iter))
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
        logging.info('Mean:[{:.4f}]'.format(accs_tensor.mean().item()))
        logging.info('Std :[{:.4f}]'.format(accs_tensor.std().item()))
        logging.info('-' * 100)

        with open(f'{csv_name}', mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file, dialect="excel")
            
            acc_mean_formatted = f"{acc_mean:.3f}"
            acc_std_formatted = f"{acc_std:.3f}"
            microf_mean_formatted = f"{microf_mean:.3f}"
            macrof_mean_formatted = f"{macrof_mean:.3f}"
            microf_std_formatted = f"{microf_std:.3f}"
            macrof_std_formatted = f"{macrof_std:.3f}"
            
            writer.writerow([pretrain_dataset_str, downstream_dataset, acc_mean_formatted, acc_std_formatted, 
                microf_mean_formatted, microf_std_formatted, 
                macrof_mean_formatted, macrof_std_formatted])
