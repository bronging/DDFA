import gc
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch_geometric.utils import add_self_loops, degree

import utils.aug as aug

def get_features_adjs(pretrain_loaders,  \
                       cache_dir, pretrain_dataset_names, target_graph_id):
    features = []
    adjs = []
    edge_indexs = [] 

    histories = []
    total_nodes_list = []

    histories_2hop = []
    total_nodes_list_2hop = []

    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
        if (step+1) not in target_graph_id:
            print(datas)
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            feature, adj = process_tu(data, data.x.shape[1])
            features.append(feature)
            adjs.append(adj)
            edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
            edge_indexs.append(edge_index)
           
    
    return features, adjs, edge_indexs


def preprocess_dataset_w_DE_pyg(features, edge_indices,
                                 drop_percent, negative_samples_num):

    aug_edge_indices = []
    aug_features = []
    combined_edge_index = None
    negative_samples = []
    lbls = []

    for i in range(len(features)):
        x = features[i]
        edge_index = edge_indices[i]

        num_nodes = x.shape[0]
        negative_sample = prompt_pretrain_sample_edgeindex(edge_index, num_nodes, negative_samples_num)
        negative_samples.append(negative_sample)
        print(negative_sample.shape)

    return aug_features, aug_edge_indices, lbls, negative_samples, combined_edge_index

def prompt_pretrain_sample_edgeindex(edge_index, num_nodes, n):
    """
    edge_index : [2, E] tensor (PyG 스타일)
    num_nodes  : 전체 노드 수
    n          : negative sample 개수
    return     : [num_nodes, 1+n] (각 노드별로 [pos, neg1, ..., negn])
    """
    # 1. adjacency list 생성
    edge_index = edge_index.cpu().numpy()
    neighbors = [[] for _ in range(num_nodes)]
    for u, v in edge_index.T:   # [2,E] → (u,v) 튜플 반복
        neighbors[u].append(v)
        neighbors[v].append(u)  # 무방향 그래프라면 필요, 방향 그래프라면 제거

    # 2. 샘플링
    whole = np.arange(num_nodes)
    res = np.zeros((num_nodes, 1+n), dtype=int) # pos 1개 + neg n개 

    for i in range(num_nodes): # 각 anchor 노드 i 
        neighs = neighbors[i]  # i의 이웃 노드들 
        if len(neighs) > 0:
            pos = np.random.choice(neighs)  # positive = 이웃 중 하나
        else:
            pos = i                         # 이웃 없으면 자기 자신이 positive 
        res[i, 0] = pos

        # negatives = 비이웃 중 n개
        non_neighs = np.setdiff1d(whole, np.array(neighs + [i]))
        if len(non_neighs) >= n:
            negs = np.random.choice(non_neighs, n, replace=False)
        else:
            negs = np.random.choice(non_neighs, n, replace=True)  # 부족하면 중복 허용
        res[i, 1:1+n] = negs # [num nodes, 1 + negative num] 

    return torch.tensor(res, dtype=torch.long)



def pca_compression(seq,k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)
    
    print(pca.explained_variance_ratio_.sum())
    return seq


def find_2hop_neighbors_sp(adj, node):
    # print(adj.getrow(node))
    # print(adj.getrow(node).todense().A)
    nodeadj = adj.getrow(node).todense().A[0]
    neighbors = []
    # print(type(adj))
    for i in range(len(nodeadj)):
        if len(neighbors) >= 4:
            break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if nodeadj[i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        nodeadj = adj.getrow(i).todense().A[0]
        for j in range(len(nodeadj)):
            if cnt >= 2:
                break
            if nodeadj[j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop



def find_2hop_neighbors(adj, node):
    neighbors = []
    # print(type(adj))
    for i in range(len(adj[node])):
        if len(neighbors) >= 10:
            break
        # print('i',i)
        # print('node',node)
        # print('adj[node][i]',adj[node,i])
        if adj[node][i] != 0 and node != i:
            neighbors.append(i)
    neighbors_2hop = []
    for i in neighbors:
        cnt = 0
        for j in range(len(adj[i])):
            if cnt >= 4:
                break
            if adj[i][j] != 0 and j != i:
                neighbors_2hop.append(j)
                cnt += 1
    return neighbors, neighbors_2hop

def build_subgraph(adj, idx_train, sparse = True):
    neighborslist = [[] for x in range(idx_train.shape[0])]
    neighbors_2hoplist = [[] for x in range(idx_train.shape[0])]
    mainindex = [[] for x in range(idx_train.shape[0])]
    mainlist = [[] for x in range(idx_train.shape[0])]
    idx_train_list = idx_train.tolist()
    for x in range(idx_train.shape[0]):        
        if sparse:
            neighborslist[x], neighbors_2hoplist[x] = find_2hop_neighbors_sp(adj, idx_train[x])
        else:
            neighborslist[x], neighbors_2hoplist[x] = find_2hop_neighbors(adj, idx_train[x])
        mainlist[x] = [idx_train_list[x]] + neighborslist[x] + neighbors_2hoplist[x]
        mainindex[x] = [x] * len(mainlist[x])
    neighborslist = sum(neighborslist,[])
    neighbors_2hoplist = sum(neighbors_2hoplist,[])
    mainlist = sum(mainlist,[])
    mainindex = sum(mainindex,[])
    return {
        'idx':torch.tensor(mainlist),
        'batch':torch.tensor(mainindex),        
    }



# Process a (subset of) a TU dataset into standard form
def process_tu(data, class_num):
    # print("len",nb_graphs)
    ft_size = data.num_features

    num = range(class_num)

    labelnum=range(class_num,ft_size)

    features = data.x[:, num]

    rawlabels = data.x[:, labelnum]
    # masks[g, :sizes[g]] = 1.0
    e_ind = data.edge_index
    # print("e_ind",e_ind)
    coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                        shape=(features.shape[0], features.shape[0]))
    # print("coo",coo)
    adjacency = coo


    adj = sp.csr_matrix(adjacency)

    # graphlabels = labels

    return features, adj

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    current_path = os.path.dirname(__file__)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




