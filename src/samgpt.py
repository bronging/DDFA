from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg
# from preprompt import PrePrompt, pca_compression
import pdb
import os
import sys

import argparse
# from downprompt import downprompt, downprompt_graph
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("SAMGPT")
import torch.nn.functional as F
import torch
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from layers import GCN, AvgReadout
import torch_scatter

import torch
import torch.nn as nn

class textprompt(nn.Module):
    def __init__(self, hid_units, type_='mul'):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.prompttype = type_
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph_embedding):
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0],1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding=self.weight * graph_embedding

        return graph_embedding
    


class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        # graph_embedding=torch.mm(self.weight, graph_embedding)
        assert len(graph_embedding) == self.weight.shape[1], 'length must equal'
        ans = torch.zeros_like(graph_embedding[0])
        for i in range(len(graph_embedding)):
            ans += self.weight[0][i] * graph_embedding[i]
        return ans

class combineprompt(nn.Module):
    def __init__(self):
        super(combineprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, 2), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding1, graph_embedding2):

        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    
class composedtoken(nn.Module):
    def __init__(self, texttokens, type_='mul'):
        super(composedtoken, self).__init__()
        # print(texttoken1.shape)
        self.texttoken = torch.cat(texttokens,dim=0)
        # print(self.texttoken.shape)
        self.prompt = weighted_prompt( len(texttokens) )
        self.type = type_

    def forward(self, seq):
        # print(seq.shape)
        
        texttoken = self.prompt(self.texttoken)
        
        # print(texttoken.shape)
        if self.type == 'add':
            texttoken = texttoken.repeat(seq.shape[0],1)
            rets = texttoken + seq
        if self.type == 'mul':
            rets = texttoken * seq
        return rets
    
class composedNet(nn.Module):
    def __init__(self, length):
        super(composedNet, self).__init__()
        #self.texttoken = torch.cat(texttokens,dim=0)
        self.length = length
        self.prompt = weighted_prompt( length ).cuda()

    def forward(self, paras):
        # print(seq.shape)
        assert self.length == len(paras), 'number of paras must equal to self.length'
        target = {}
        for key, value in paras[0].items():
            target[key] = torch.zeros_like(value)
        for key in paras[0].keys():
            para_key = [para[key] for para in paras]
            target[key] = self.prompt(para_key)

        return target
 

class downstreamprompt(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers_num, fea_pretext_weights, str_pretext_weights, 
                combines, type_ = 'mul', ablation = 'all'):
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
            seq_fea == open_seq_fea
        elif self.ablation_choice[-2:] == 'fc':
            seq_fea = composed_seq_fea
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
                  combines, type_='mul', ablation = 'all'):
        super(downprompt, self).__init__()

        self.num_pretrain_datasets = len(fea_pretext_weights)
        
        self.downstreamPrompt = downstreamprompt(feature_dim, ft_in, num_layers_num, 
            fea_pretext_weights, str_pretext_weights, combines, type_, ablation)
        
        self.nb_classes = nb_classes
        self.leakyrelu = nn.ELU()
        self.one = torch.ones(1, ft_in)
        self.ave = torch.FloatTensor(nb_classes, ft_in)


    def forward(self,features,adj,sparse,gcn,idx,labels=None,train=0):

        embeds = self.downstreamPrompt(features, gcn, adj, sparse).squeeze(0)   
        rawret = embeds[idx]
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers, MLP, GatLayers
from layers import AvgReadout 
import numpy as np
from sklearn.decomposition import PCA
import copy

class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation, num_pretrain_dataset_num, num_layers_num, 
        dropout, type_, backbone = 'gcn', alpha=1.0, ablation='all'):
        super(PrePrompt, self).__init__()
        self.lp = Lp(n_in, n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.read = AvgReadout()
        self.prompttype = type_
        
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
         
    def compute_prelogits_LP(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list, 
        sparse = False):
        for fea_pretext, str_layers, seq, adj in \
            zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.lp(self.gcn, seq, adj, sparse)
            else:
                fea_prelogits = self.lp(self.gcn, fea_pretext(seq) , adj, sparse) 
                str_prelogits = self.lp(self.gcn, seq, adj, sparse, str_layers)
                yield self.ablation(fea_prelogits, str_prelogits)
        
    def compute_prelogits_GRAPHCL(self, feature_prompt_layers, structure_prompt_layers, seq_list, adj_list,
        sparse = False, msk = None, samp_bias1 = None, samp_bias2 = None):
        for fea_pretext, str_layers, seq, adj in \
            zip(feature_prompt_layers, structure_prompt_layers, seq_list, adj_list):
            if self.ablation_choice == 'None':
                yield self.graphcledge(self.gcn, 
                seq[0], seq[1], seq[2], seq[3], 
                adj[0], adj[1], adj[2], sparse, msk,
                samp_bias1, samp_bias2, 'edge')
            else:
                preseq_list = [fea_pretext(seq[i]) for i in range(len(seq))] 
                fea_prelogits = self.graphcledge(self.gcn, 
                    preseq_list[0], preseq_list[1], preseq_list[2], preseq_list[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, aug_type='edge')

                str_prelogits = self.graphcledge(self.gcn, 
                    seq[0], seq[1], seq[2], seq[3], 
                    adj[0], adj[1], adj[2], sparse, msk,
                    samp_bias1, samp_bias2, 'edge', str_layers, args.st_validation)
                
                yield self.ablation(fea_prelogits, str_prelogits)

    def embed(self, seq, adj, sparse, msk, LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

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
            logits = list(self.compute_prelogits_GRAPHCL(
                self.feature_prompt_layers, 
                self.structure_prompt_layers,
                seq_list, 
                adj_list,  
                sparse, msk, samp_bias1, samp_bias2))
            for i in range(len(logits)):
                loss = self.loss(logits[i], lbl[i])
                total_loss += loss
        else:
            logits = list(self.compute_prelogits_LP(
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

        return total_loss



def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    print(pca.explained_variance_ratio_.sum())
    return seq

def svd_compression(seq, k):
    res = np.zeros_like(seq)
    U, Sigma, VT = np.linalg.svd(seq)
    print(U[:,:k].shape)
    print(VT[:k,:].shape)
    res = U[:,:k].dot(np.diag(Sigma[:k]))
 
    return res

def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))

def compareloss(feature,tuples,temperature):
    h_tuples=mygather(feature,tuples)
    temp = torch.arange(0, len(tuples)).to(feature.device)
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    h_i = mygather(feature, temp)

    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim) / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))

    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return torch.tensor(res.astype(int) )

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

parser.add_argument('--st_validation', type=str, default='origin', help='backbone')
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

nb_epochs = 10000
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
result_dir = os.path.join(parent_directory, 'result')
cache_dir = os.path.join(parent_directory, 'cache')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
graphids = ''
for id in args.graphId:
    graphids += str(id) + '_'
set_name = f'model_{args.downstream_task}_{args.pretrain_method}_{pretrain_dataset_str}_{args.alpha}_{args.beta}_{args.ablation_pre}_{args.ablation_down}_{args.unify_dim}_{args.hid_units}_{args.lr}_{args.backbone}'
save_name = os.path.join(save_dir, f'{set_name}.pkl')
csv_name = os.path.join(result_dir, f'{set_name}.csv')
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=logfile,
                    filemode='a')

logging.info('#'*50)
logging.info('PreTrain datasets are ')
logging.info(pretrain_dataset_names)
logging.info('Downastream dataset is ')
logging.info(args.dataset)
logging.info(args.ablation_down)
logging.info(args.st_validation)

model = PrePrompt(unify_dim, hid_units, nonlinearity, num_pretrain_dataset_num, 
        num_layers_num, 0.1, type_ = args.combinetype, backbone = args.backbone,
        alpha = args.alpha, ablation = args.ablation_pre).cuda()

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
#split_LP:
            if args.pretrain_method == 'splitLP':
                if not os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt'):
                    negetive_sample = prompt_pretrain_sample(adj, 50)
                    torch.save(negetive_sample, f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_sample = torch.load(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_samples.append(negetive_sample)

            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            features.append(feature)
            adjs.append(adj)
#LP:    
        if args.pretrain_method == 'LP':    
            combinedadj = process.combine_dataset_list_sp(adjs)
            print('combinedadj', combinedadj.shape)
            negetive_sample = prompt_pretrain_sample(combinedadj, args.negative_samples_num)


    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        print('Using CUDA')
        model = model.cuda()
        
        features = [tensors.cuda() for tensors in features]
        adjs = [process.sparse_mx_to_torch_sparse_tensor(adj).cuda()  if sparse else torch.FloatTensor(adj.todense()).cuda() 
            for adj in adjs]
        lbls = [tensors.cuda() for tensors in lbls]
        negetive_samples = [tensors.cuda() for tensors in negetive_samples]
        if len(negetive_samples) == 0:
            negetive_samples = negetive_sample.cuda()
        aug_adjs = [tensors.cuda() for tensors in aug_adjs]
        aug_features = [tensors.cuda() for tensors in aug_features]
        

    for epoch in range(nb_epochs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        loss = 0
        model.train()
        optimiser.zero_grad()
#GRAPHCL
        if args.pretrain_method == 'GRAPHCL':
            loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None)
#LP:    
        if args.pretrain_method == 'LP' or args.pretrain_method == 'splitLP':
            loss =  model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)
        loss.backward()
        optimiser.step()
        print('Loss:[{:.6f}]'.format(loss))
        if loss < best:
            firstbest = 1
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
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
    else:
        # from downprompt import downprompt
        a = 1
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
        for i in tqdm(range(100)):
            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()

            combines.append(args.beta)

            log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                            fea_pretext_weights, str_pretext_weights, combines, args.combinetype,
                            args.ablation_down).cuda()

            log.train()

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
            opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.cuda()
            best = 1e9
            best_acc = torch.zeros(1).cuda()

            for _ in range(400):
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