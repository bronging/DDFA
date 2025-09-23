import torch 
from tqdm import tqdm
from utils.logging_ import write 
import inspect
from utils import process
import utils.aug as aug 

import numpy as np
import ot

def wgan_gp(critic, real, fake, lam_gp=10.0):
    # real, fake: [B, d]
    B = min(real.size(0), fake.size(0))
    real = real[:B]; fake = fake[:B]
    eps = torch.rand(B, 1, device=real.device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)
    d_hat = critic(x_hat)              # [B]
    grads = torch.autograd.grad(
        outputs=d_hat.sum(),
        inputs=x_hat,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]                                # [B, d]
    gp = ((grads.norm(2, dim=1) - 1.0)**2).mean()
    return lam_gp * gp


def build_fixed_support_Y(fixed_samples, k=None):
    # fixed_samples: list of [n_s, d]
    allX = torch.cat(fixed_samples, dim=0)  # [D*n_s, d]
    # allX = fixed_samples

    if (k is None) or (k >= allX.size(0)):
        # k를 생략하거나 너무 크게 잡으면 그냥 모든 포인트를 지지점으로
        Y = allX.clone()
    else:
        # (선택) k-means로 요약
        from sklearn.cluster import KMeans
        Y_np = allX.detach().cpu().numpy()
        km = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(Y_np)
        Y = torch.from_numpy(km.cluster_centers_).to(allX.device).float()  # [k, d]
    return Y  # [K, d], 고정!


def sinkhorn_barycenter_weights(Y_torch, fixed_samples, eps=0.05, domain_weights=None,
                                method='sinkhorn_stabilized', numIter=1000, stopThr=1e-7):
    """
    고정 지지점 Y 위에 얹히는 무게 a \in Δ^K만 계산.
    """
    device = Y_torch.device
    Y = Y_torch.detach().cpu().numpy()     # [K, d]
    X_list = [xs.detach().cpu().numpy() for xs in fixed_samples]  # 각 [n_s, d]

    # 각 도메인 히스토그램(균등)
    b_list = [np.ones(x.shape[0]) / x.shape[0] for x in X_list]
    if domain_weights is None:
        domain_weights = np.ones(len(X_list)) / len(X_list)

    # 비용행렬 C_k = dist(Y, X_k)  (W1: euclidean)
    C_list = [ot.dist(Y, X, metric='euclidean') for X in X_list]  # 각 [K, n_s]

    a = ot.bregman.barycenter(
        C_list, b_list, reg=eps, weights=domain_weights,
        numItermax=numIter, stopThr=stopThr, method=method
    )  # [K,]
    a = torch.from_numpy(a).to(device).float()
    a = a / a.sum()
    return a 

def sample_from_barycenter(Y, a, m):
    idx = torch.multinomial(a, num_samples=m, replacement=True)  # [m]
    return Y[idx]  # [m, d]


def cost_matrix(X, Y):
    # X: [n, d], Y: [N, d] (torch.Tensor)
    M = torch.cdist(X, Y, p=2)  # Euclidean distance
    M = M / (M.max() + 1e-9)    # normalization
    return M

# barycenter collapse 방지용 diversity loss 
def diversity_loss(Y):
    dist = torch.cdist(Y, Y, p=2)
    return -dist.mean() 


def sinkhorn_barycenter_iteration(Xs, as_, Y, b, weights, n_iter=20, reg=1e-1):
    """
    Algorithm 1: Fast Computation of Wasserstein Barycenters (Cuturi & Doucet 2014)
    
    Xs: list of [n_i, d] torch.Tensor (각 도메인 support points)
    as_: list of [n_i] torch.Tensor (각 도메인 확률 벡터, 보통 uniform)
    Y: [N, d] torch.Tensor (barycenter support 초기값)
    b: [N] torch.Tensor (barycenter 확률, 보통 uniform)
    weights: [m] torch.Tensor (도메인별 가중치, 합=1)
    n_iter: 반복 횟수
    reg: Sinkhorn 정규화 파라미터
    """
    device = Y.device
    for it in range(n_iter):
        Y_new = torch.zeros_like(Y, device=device)
        
        for X, a, w in zip(Xs, as_, weights):
            # 1. Cost matrix
            M = cost_matrix(X, Y)
            
            # 2. Optimal transport plan (torch backend)
            gamma = ot.sinkhorn(a, b, M, reg, numItermax=100, method='sinkhorn', backend='torch')
            
            # 3. Barycenter update
            mass = gamma.sum(0)[:, None] + 1e-9  # [N, 1]
            Y_new += w * (gamma.T @ X) / mass
        
        # 4. Update Y
        Y = Y_new.detach()
        
    return Y

# 도메인별 분포와 barycenter support Y 사이 OT plan 구하고 barycenter 업데이트
def wasserstein_barycenter(Xs, as_, Y, b, weights, n_iter=20, reg=1e-1, lambda_div=0.001):
    device = Y.device
    for it in range(n_iter):
        # print(f'=========={it}===============')
        gammas = []
        Y_new = torch.zeros_like(Y, device=device)
        for X, a, w in zip(Xs, as_, weights):
            # print('X: ', X[:5, :7])
            # print('Y: ', Y[:5, :7])
            M = cost_matrix(X, Y)
            
            # torch backend sinkhorn
            gamma = ot.sinkhorn(a, b, M, reg, numItermax=50, method='sinkhorn', backend='torch') # transport plan gamma, sinkhorn으로 근사해서 구함. OT plan. [i,j]=xi의 질량 중 yj로 옮겨간 비율 
            # barycenter 업데이트
            mass = gamma.sum(0)[:, None] + 1e-9 # 열 방향 합 - 각 Y가 얼마나 많은 source 질량을 받았는지 
            Y_new += w * (gamma.T @ X) / mass # primal 의 해 gamma를 이용해서 source 분포의 support를 barycenter 위치로 옮기는 과정 
            gammas.append(gamma)

            # print('M:', M.shape) # [256, 256]
            # print('gamma: ', gamma.shape)  # [256, 256]
            # print('mass: ', mass.shape) # [256, 1]
            # print('Y_new: ', Y_new.shape) # [256, 50]
        # # diversity regularizer 적용 (detach 하지 않고)
        # div_loss = diversity_loss(Y_new)
        # Y_new = Y_new + lambda_div * torch.randn_like(Y_new) * div_loss.item()  
        Y = Y_new

    Y = Y_new.detach()
    # lr = 0.1
    # Y = (1 - lr) * Y + lr * Y_new
    # Y = Y.detach()
        
    # print('gama', gamma.shape)
    # print('y: ', Y.shape)

    # print('X[0]: ', Xs[0][:5])
    # print('Y: ', Y[:5, :10])
    return Y, gammas

def wasserstein_distance(Z, bary_Y, reg=1e-1):
    """
    Z: [n_i, d] (source domain representation)
    bary_Y: [N, d] (barycenter support)
    """

    # cost matrix (Euclidean distances)
    # C = torch.cdist(Z, bary_Y, p=2)  # [n, N]
    # C = C / (C.max() + 1e-9)

    # W1 = torch.sum(gamma * C)

    # return W1

    n, d = Z.shape
    N = bary_Y.shape[0]

    # uniform weights
    a = torch.ones(n, device=Z.device) / n
    b = torch.ones(N, device=Z.device) / N

    # cost matrix (Euclidean distances)
    C = torch.cdist(Z, bary_Y, p=2)  # [n, N]
    C = C / (C.max() + 1e-9)

    # # convert to numpy for POT
    # a_np, b_np, C_np = a.cpu().numpy(), b.cpu().numpy(), C.detach().cpu().numpy()

    # Sinkhorn distance
    # G = ot.sinkhorn(a_np, b_np, C_np, reg)  # transport plan
    G = ot.sinkhorn(a, b, C, reg, numItermax=50, method="sinkhorn", backend="torch")  # transport plan
    W1 = torch.sum(G * C)

    return W1

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

def aggregate_features(X, edge_index, gamma=0):
    num_nodes = X.size(0)
    # AX 
    A_hat = normalize_adjacency(edge_index, num_nodes)    

    # I = torch.eye(num_nodes, device=X.device)
    # return (I-A_hat) @ X 
    # return A_hat @ A_hat @ X # A^2X
    return (A_hat @ X) # aggregation: A_hat X

def train_graphacl(model, feat_list, edge_list, save_name, \
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50, 
                        gamma=0):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'feat_list', 'edge_list', 'lbls', 'sparse', 'save_name'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    # print(f"Feature size: {aug_sfeatures.shape()}")
    #print(f"Adj size: {aug_adjs.shape()}")
    best = float('inf')
    cnt_wait = 0

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    torch.save(model.state_dict(), save_name)
    best_t = 0 

    aggregate_feat = [aggregate_features(feat_list[i], edge_list[i], gamma=gamma) for i in range(len(feat_list))]
    # aggregate_feat = None
    for epoch in range(start_epoch, num_epoch):
        model.train()
        
        loss = model(feat_list, edge_list, aggregate_feat) 
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, loss.item())) 
        
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
    write(f'best epoch: {best_t}')

from geomloss import SamplesLoss

def train_fug_graphcl(model, aug_features=None, aug_adjs=None, features=None, edge_indexs=None, 
                        lbls=None, sparse=False, save_name='', drop_percent=0.1,\
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50, \
                        gamma=0, model_type=None, barycenter=False, sample_size=183, \
                        unify_dim=50, ep_aug=False ):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'aug_features', 'aug_adjs', 'lbls', 'sparse', 'save_name', 'features', 'edge_indexs'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    # print(f"Feature size: {aug_sfeatures.shape()}")
    #print(f"Adj size: {aug_adjs.shape()}")
    best = float('inf')
    cnt_wait = 0

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # torch.save(model.state_dict(), save_name)
    best_t = 0 
    bary_Y = None 
    num_datasets = len(features) if ep_aug else len(aug_features)

    if barycenter: 
        if model_type == 'filterFUG': 
            unify_dim = unify_dim // 2 

        sinkhorn_loss = SamplesLoss("sinkhorn", p=1, blur=0.05)

        best_barycenter = None 
        device = features[0].device if ep_aug else aug_features[0][0].device 
        # barycenter support 초기화
        bary_Y = torch.randn(sample_size, unify_dim, device=device)
        b = torch.ones(sample_size, device=device) / sample_size 

        # 각 도메인별 support point에 대한 가중치
        as_ = [torch.ones(sample_size, device=device) / sample_size for _ in range(num_datasets)]
        # 각 도메인에 대한 가중치 
        weights = torch.ones(num_datasets, device=device) / num_datasets

    if model_type == 'samgpt': 
        aggregate_feat = None 
    else: 
        if ep_aug: 
            aggregate_feat = [aggregate_features(features[i], edge_indexs[i], gamma=gamma) for i in range(num_datasets)]
        else: 
            aggregate_feat = [aggregate_features(aug_features[i][0], aug_adjs[i][0], gamma=gamma) for i in range(num_datasets)]
    
    for epoch in range(start_epoch, num_epoch):
        model.train()

        if ep_aug: 
            aug_features, aug_adjs, lbls, _, _ = process.preprocess_dataset_w_DE_pyg(
                                                                        features, edge_indexs, 'GRAPHCL', \
                                                                        drop_percent, 0)
            aug_adjs = [
                [edge_index.cuda() for edge_index in ei_list]
                for ei_list in aug_adjs
            ]
            lbls = [tensors.cuda() for tensors in lbls]
       
        if barycenter: 
            # XT로 축소된 피처들 
            if model_type == 'samgpt': 
                xt_list = [model.samplers[i](aug_features[i][0], aug_adjs[i][0]) for i in range(len(aug_features))]
            else: 
                xt_list = model.get_reduction(aug_features, aug_adjs, aggregate_feat)
            bary_Y, gammas = wasserstein_barycenter(xt_list, as_, bary_Y, b, weights, n_iter=10) # "매 epoch마다 새로 뽑는 anchor" 역할 -> detach()!
            W1loss = torch.tensor(0.0, dtype=torch.float32).to(aug_features[0].device)
            for i in range(1, len(xt_list)): 
                W1loss += wasserstein_distance(xt_list[i], xt_list[0])
            for xt, gamma in zip(xt_list, gammas):
            # for xt in xt_list:
                W1loss += sinkhorn_loss(xt, bary_Y)
                C = torch.cdist(xt, bary_Y, p=2)
                C = C / (C.max() + 1e-9)
                W1loss += torch.sum(gamma * C) # gamma^p * cost = W의 p 제곱. p=1, W1 거리.  
                
                # W1loss += wasserstein_distance(xt, bary_Y, reg=0.1)

        if model_type == 'samgpt':
            loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None) 

        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, aggregate_feat) 
        # if barycenter: 
        #     # XT로 축소된 피처들 
        #     # emb_list = model.get_sample_embed(aug_features, aug_adjs, aggregate_feat,  sparse, None, None, None)
        #     emb_list = [model.samplers[i](logits[i]) for i in range(len(aug_features))]
        #     bary_Y = wasserstein_barycenter(emb_list, as_, bary_Y, b, weights, n_iter=30) # "매 epoch마다 새로 뽑는 anchor" 역할 -> detach()!
        #     W1loss = torch.tensor(0.0, dtype=torch.float32).to(aug_features[0].device)
        #     for xt in emb_list:
        #         W1loss += wasserstein_distance(xt, bary_Y, reg=0.1)


        if barycenter: 
            print(f'cl loss: {loss}, W1: {W1loss}')
            total_loss = loss + W1loss
        else: 
            # print(f'cl loss: {loss}')
            total_loss = loss 

        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, total_loss.item())) 
        
        if total_loss < best:
            best = total_loss
            best_t = epoch
            best_barycenter = bary_Y
            # best_barycenter = xt_list[0]
            cnt_wait = 0
            if epoch > 5: 
                torch.save({    
                    'model_state_dict': model.state_dict(),
                    'best_barycenter': best_barycenter, 
                    'epoch': epoch, 
                    'loss': total_loss
                    }, save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
    write(f'best epoch: {best_t}')


def train_fug_lp(model, features, edge_indexs, negetive_samples, 
                        sparse=False, save_name='',\
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50, \
                        gamma=0, model_type=None, barycenter=False, sample_size=183, \
                        unify_dim=50, w1loss=1.0):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'sparse', 'save_name', 'features', 'edge_indexs', 'negetive_samples'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    # print(f"Feature size: {aug_sfeatures.shape()}")
    #print(f"Adj size: {aug_adjs.shape()}")
    best = float('inf')
    cnt_wait = 0

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.save(model.state_dict(), save_name)
    best_t = 0 
    bary_Y = None 
    num_datasets = len(features) 

    if barycenter: 
        if model_type == 'filterFUG': 
            unify_dim = unify_dim // 2 

        sinkhorn_loss = SamplesLoss("sinkhorn", p=1, blur=0.05)

        best_barycenter = None 
        device = features[0].device 
        # barycenter support 초기화
        bary_Y = torch.randn(sample_size, unify_dim, device=device)
        # bary_Y = torch.randn(sample_size, 256, device=device)
        b = torch.ones(sample_size, device=device) / sample_size 

        # 각 도메인별 support point에 대한 가중치
        as_ = [torch.ones(sample_size, device=device) / sample_size for _ in range(num_datasets)]
        # 각 도메인에 대한 가중치 
        weights = torch.ones(num_datasets, device=device) / num_datasets

    if model_type == 'samgpt': 
        aggregate_feat = None 
    else: 
        aggregate_feat = [aggregate_features(features[i], edge_indexs[i], gamma=gamma) for i in range(num_datasets)]
    
    for epoch in range(start_epoch, num_epoch):
        model.train()
        optimiser.zero_grad()

        if barycenter: 
            # XT로 축소된 피처들 
            xt_list = model.get_reduction(features, edge_indexs, aggregate_feat)
            with torch.no_grad():
                bary_Y, gammas = wasserstein_barycenter(xt_list, as_, bary_Y, b, weights, n_iter=10) # "매 epoch마다 새로 뽑는 anchor" 역할 -> detach()!
            W1loss = torch.tensor(0.0, dtype=torch.float32).to(features[0].device)
            for i in range(1, len(xt_list)): 
                W1loss += wasserstein_distance(xt_list[i], xt_list[0])
            for xt, gamma in zip(xt_list, gammas):
            # for xt in xt_list:
                W1loss += sinkhorn_loss(xt, bary_Y)
                C = torch.cdist(xt, bary_Y, p=2)
                C = C / (C.max() + 1e-9)
                W1loss += torch.sum(gamma * C) # gamma^p * cost = W의 p 제곱. p=1, W1 거리.  
                
                # W1loss += wasserstein_distance(xt, bary_Y, reg=0.1)
        loss = model(features, edge_indexs, sparse, None, None, None, None, aggregate_feat, samples=negetive_samples) 

        # if barycenter: 
        #     # XT로 축소된 피처들 
        #     xt_list = model.get_emb(features, edge_indexs, aggregate_feat)
        #     with torch.no_grad():
        #         bary_Y, gammas = wasserstein_barycenter(xt_list, as_, bary_Y, b, weights, n_iter=1) # "매 epoch마다 새로 뽑는 anchor" 역할 -> detach()!
        #     W1loss = torch.tensor(0.0, dtype=torch.float32).to(features[0].device)
        #     for i in range(1, len(xt_list)): 
        #         W1loss += wasserstein_distance(xt_list[i], xt_list[0])
        #     for xt, gamma in zip(xt_list, gammas):
        #     # for xt in xt_list:
        #         W1loss += sinkhorn_loss(xt, bary_Y)
        #         C = torch.cdist(xt, bary_Y, p=2)
        #         C = C / (C.max() + 1e-9)
        #         W1loss += torch.sum(gamma * C) # gamma^p * cost = W의 p 제곱. p=1, W1 거리.  

        if barycenter: 
            print(f'cl loss: {loss.item()}, W1: {W1loss.item()}')
            total_loss = loss + w1loss * W1loss
        else: 
            # print(f'cl loss: {loss}')
            total_loss = loss 

        
        total_loss.backward()
        optimiser.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, total_loss.item())) 
        


        if total_loss < best:
            best = total_loss
            best_t = epoch
            best_barycenter = bary_Y
            # for i, xt in enumerate(xt_list): 
            #     print(f'{i}: {xt.mean(dim=0)[:10]}, {xt.std(dim=0)[:10]}')
            # print(best_barycenter[:10, :7])
            # print(f'std: {best_barycenter.std(dim=0)}')
            # best_barycenter = xt_list[0]
            cnt_wait = 0
            if epoch > 5: 
                torch.save({    
                    'model_state_dict': model.state_dict(),
                    'best_barycenter': best_barycenter, 
                    'epoch': epoch, 
                    'loss': total_loss, 
                    'fixed_idx': [sampler.fixed_indices for sampler in model.samplers]
                    }, save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
        
        torch.cuda.empty_cache()
        del total_loss
        
    write(f'best epoch: {best_t}')



def train_fug_first_graphcl(model, aug_features, aug_adjs, 
                        lbls, sparse, save_name, \
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50, gamma=0):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'aug_features', 'aug_adjs', 'lbls', 'sparse', 'save_name'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    # print(f"Feature size: {aug_sfeatures.shape()}")
    #print(f"Adj size: {aug_adjs.shape()}")
    best = float('inf')
    cnt_wait = 0

    # optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 1) 먼저 encoder 파라미터만 모으기
    enc_params = list(model.dimension_encoder_layers.parameters())
    enc_param_ids = {id(p) for p in enc_params}

    # 2) 나머지(메인) 파라미터 분리
    base_params = [p for p in model.parameters() if id(p) not in enc_param_ids]

    # 3) Optimizer 두 개 생성 (학습률/WD 분리 가능)
    optim_main = torch.optim.Adam(base_params, lr=lr, weight_decay=weight_decay)

    enc_lr = 5e-4 if 'enc_lr' not in globals() else enc_lr      # 예시
    enc_wd = 1e-4 if 'enc_wd' not in globals() else enc_wd
    optim_enc  = torch.optim.Adam(enc_params, lr=enc_lr, weight_decay=enc_wd)

    torch.save(model.state_dict(), save_name)
    best_t = 0 

    
    aggregate_feat = [aggregate_features(aug_features[i][0], aug_adjs[i][0], gamma=gamma) for i in range(len(aug_features))]
    # aggregate_feat = None
    for epoch in range(start_epoch, num_epoch):
        model.train()
                
        optim_enc.zero_grad()

        de = torch.tensor(0.0, dtype=torch.float32).to(aug_features[0].device)
        basis_mean = []

        for idx, dim_pretext in enumerate(model.dimension_encoder_layers): 
            sample = model.samplers[idx](aug_features[idx][0], aug_adjs[idx][0])   
            _ = dim_pretext(sample) 
            de += dim_pretext.dimensional_loss()
            basis_mean.append(dim_pretext.mean_basis_vector())
        basis_mean_loss = torch.stack(basis_mean).mean(dim=0).pow(2).mean()
        de_loss = de + basis_mean_loss

        
        de_loss.backward()
        optim_enc.step()

        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None, aggregate_feat) 

        optim_main.zero_grad()
        loss.backward()
        optim_main.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, loss.item())) 
        
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
    write(f'best epoch: {best_t}')

def train_fug_graphcl_sampling(model, aug_features, aug_adjs, 
                        lbls, sparse, save_name, \
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'aug_features', 'aug_adjs', 'lbls', 'sparse', 'save_name'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    # print(f"Feature size: {aug_sfeatures.shape()}")
    #print(f"Adj size: {aug_adjs.shape()}")
    best = float('inf')
    cnt_wait = 0

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.save(model.state_dict(), save_name)
    best_t = 0 
    for epoch in range(start_epoch, num_epoch):
        
        if epoch % 50 == 0:
            for sampler in model.samplers:
                sampler.reset_indices()  # ✅ 이 줄로 sampler가 다음 호출 시 새롭게 샘플링함

        model.train()
        
        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None) 
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, loss.item())) 
        
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
    write(f'best epoch: {best_t}')
 
def train_samgpt_graphcl(model, aug_features, aug_adjs, 
                        lbls, sparse, save_name, \
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50, pretrain_dataset_names=None):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'aug_features', 'aug_adjs', 'lbls', 'sparse', 'save_name'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    # print(f"Feature size: {aug_sfeatures.shape()}")
    #print(f"Adj size: {aug_adjs.shape()}")
    best = float('inf')
    cnt_wait = 0

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.save(model.state_dict(), save_name)
    best_t = 0 
    for epoch in range(start_epoch, num_epoch):
        model.train()
        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, None) 
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, loss.item())) 
        
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
    write(f'best epoch: {best_t}')
    arr = [] 
    for i in range(len(aug_features)):
        print(i)
        arr.append(model.get_forward(aug_features[i][0], aug_adjs[i][0], sparse, None, False, i))
    print('emb[0] > > > ', arr[0].size())
    



    # # (2) 임베딩 합치기 및 라벨 생성
    # all_embeddings = torch.cat(arr, dim=0).cpu().numpy()
    # labels = np.concatenate([[i] * len(arr[i]) for i in range(len(arr))])

    # # (3) t-SNE 수행
    # tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    # tsne_result = tsne.fit_transform(all_embeddings)

    # # (4) 시각화 및 저장
    # plt.figure(figsize=(10, 6))
    # colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    # for i in range(6):
    #     idx = labels == i
    #     plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=f'Graph {i}', color=colors[i], alpha=0.6, s=10)

    # plt.title("GNAE_t-SNE of Node Embeddings from 6 Graphs (Domain Token Applied)")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("MDGPT_GNAE_tsne_graph_domains.png", dpi=300)

        
    # plt.figure(figsize=(8, 5))

    # for i, z in enumerate(arr):
    #     norms = torch.norm(z, dim=1).cpu().numpy()
    #     plt.hist(norms, bins=50, alpha=0.4, color=colors[i], label=f'Graph {i}', density=True)

    # plt.xlabel("L2 Norm of Node Embedding")
    # plt.ylabel("Density")
    # plt.title("GNAE_Embedding Norm Distribution per Graph")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("MDGPT_GNAE_norm_graph_domains.png", dpi=300)

def train_samgpt_lp(model, features, adjs, 
                        negetive_samples, sparse, save_name, \
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adjs', 'lbls', 'sparse', 'save_name', 'negetive_samples'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    # print(f"Feature size: {aug_sfeatures.shape()}")
    #print(f"Adj size: {aug_adjs.shape()}")
    best = float('inf')
    cnt_wait = 0

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.save(model.state_dict(), save_name)
    best_t = 0 
    for epoch in range(start_epoch, num_epoch):
        model.train()
        loss =  model(features, adjs, sparse, None, None, None, None, samples=negetive_samples)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        print('Epoch {}: Loss = {:.6f}'.format(epoch, loss.item())) 
        
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_name)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            tqdm.write('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
    write(f'best epoch: {best_t}')

