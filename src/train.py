import inspect
import os

import numpy as np
import ot
import torch
from tqdm import tqdm

from utils.logging_ import write


def cost_matrix(X, Y):
    # X: [n, d], Y: [N, d] (torch.Tensor)
    M = torch.cdist(X, Y, p=2)  # Euclidean distance
    M = M / (M.max() + 1e-9)    # normalization
    return M

# Compute OT plan between each domain distribution and barycenter support Y, then update barycenter
def wasserstein_barycenter(Xs, as_, Y, b, weights, n_iter=20, reg=1e-1, lambda_div=0.001):
    device = Y.device
    for it in range(n_iter):
        gammas = []
        Y_new = torch.zeros_like(Y, device=device)
        for X, a, w in zip(Xs, as_, weights):
            M = cost_matrix(X, Y)
            
            # Compute transport plan via Sinkhorn approximation; gamma[i,j] = fraction of mass at x_i transported to y_j            
            gamma = ot.sinkhorn(a, b, M, reg, numItermax=50, method='sinkhorn', backend='torch')
            # Column-wise sum: total source mass received by each y_j
            mass = gamma.sum(0)[:, None] + 1e-9  
            # Update barycenter position using primal solution gamma
            Y_new += w * (gamma.T @ X) / mass 
            gammas.append(gamma)

        Y = Y_new

    Y = Y_new.detach()
    return Y, gammas

def normalize_adjacency(edge_index, num_nodes):
    
    # 1. build sparse adjacency matrix
    adj = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.size(1), device=edge_index.device), 
        (num_nodes, num_nodes)
    )

    # 2. compute degree and D^{-1/2}
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    # 3. compute normalized adjacency: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = deg_inv_sqrt.view(-1, 1)
    row, col = edge_index
    norm_vals = D_inv_sqrt[row] * D_inv_sqrt[col]
    
    norm_adj = torch.sparse_coo_tensor(
        edge_index, norm_vals.squeeze(), (num_nodes, num_nodes)
    )

    return norm_adj  

def aggregate_features(X, edge_index):
    num_nodes = X.size(0)
    # AX 
    A_hat = normalize_adjacency(edge_index, num_nodes)    

    return  (A_hat @ X)  # aggregation: A_hat X


def train_fug_lp(model, features, edge_indexs, negetive_samples, 
                        sparse=False, save_name='',\
                        lr=0.001, weight_decay=0.0, \
                        num_epoch=10000,\
                        sample_size=256, \
                        unify_dim=50, alpha=3.0):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'sparse', 'save_name', 'features', 'edge_indexs', 'negetive_samples'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")
    
    device = features[0].device 

    best = float('inf')
    cnt_wait = 0
    patience=50
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_t = 0 
    bary_Y = None 
    num_datasets = len(features) 

    best_barycenter = None 

    # Initialize barycenter support
    bary_Y = torch.randn(sample_size, unify_dim, device=device)
    b = torch.ones(sample_size, device=device) / sample_size 
    as_ = [torch.ones(sample_size, device=device) / sample_size for _ in range(num_datasets)]  # Uniform weights over support points per domain
    weights = torch.ones(num_datasets, device=device) / num_datasets  # Uniform weights over domains

    aggregate_feat = [aggregate_features(features[i], edge_indexs[i]) for i in range(num_datasets)]
    
    loss_log = {"total": [], "lp": [], "de": [], "basis": [], "w1": []}
    for epoch in range(num_epoch):
        model.train()
        optimiser.zero_grad()

        # Dimension-reduced features via XT projection
        xt_list = model.get_reduction(features, edge_indexs, aggregate_feat)
            
        with torch.no_grad():
            gammas = []
            for X, a, w in zip(xt_list, as_, weights):
                M = cost_matrix(X, bary_Y)
                gamma = ot.sinkhorn(a, b, M, reg=1e-1, numItermax=50, method='sinkhorn', backend='torch')
                gammas.append(gamma)

        wb_loss = torch.tensor(0.0, dtype=torch.float32).to(features[0].device)
        wb_list = []
        for xt, gamma in zip(xt_list, gammas):
            C = torch.cdist(xt, bary_Y, p=2)
            C = C / (C.max() + 1e-9)
            wb_i = torch.sum(gamma * C)  # gamma * cost = W_1 distance (p=1 Wasserstein)
            wb_loss += wb_i
            wb_list.append(wb_i)

        lp_diversity_loss = model(features, edge_indexs, sparse, aggregate_feat, samples=negetive_samples) 

        if epoch % 10 == 0: 
            with torch.no_grad():
                bary_Y, _ = wasserstein_barycenter(xt_list, as_, bary_Y, b, weights, n_iter=10)  # Updated every 10 epochs; detach() prevents gradient flow

        print(f'lp_diversity_loss: {lp_diversity_loss.item()}, wb_loss: {wb_loss.item()}')
        
        total_loss = lp_diversity_loss + alpha * wb_loss

        total_loss.backward()
        optimiser.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, total_loss.item())) 
        
        if total_loss < best:
            best = total_loss
            best_t = epoch
            best_barycenter = bary_Y
            cnt_wait = 0 
            torch.save({'model_state_dict': model.state_dict()}, save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
        
        torch.cuda.empty_cache()
        del total_loss