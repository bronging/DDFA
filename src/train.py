import torch 
import random
from tqdm import tqdm
# from preprompt import sliced_wasserstein_torch
from utils.logging_ import write 
import inspect
from utils import process
import utils.aug as aug 

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import torch.nn.functional as F


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

    # RWPE 
    # T = compute_transition_matrix(edge_index.cpu().numpy(), num_nodes)
    # s = compute_structural_encoding(T, max_order=8)  # sᵢ ∈ ℝ⁴
    # return s 

    # AX 
    A_hat = normalize_adjacency(edge_index, num_nodes)    

    # I = torch.eye(num_nodes, device=X.device)
    # return (I-A_hat) @ X 
    return (A_hat @ X) + gamma*X  # aggregation: A_hat X

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

def train_fug_graphcl(model, aug_features, aug_adjs, 
                        lbls, sparse, save_name, \
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50, gamma=0, model_type=None):
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
    
    if model_type == 'samgpt': 
        aggregate_feat = None 
    else: 
        aggregate_feat = [aggregate_features(aug_features[i][0], aug_adjs[i][0], gamma=gamma) for i in range(len(aug_features))]
    for epoch in range(start_epoch, num_epoch):
        model.train()
        
        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, aggregate_feat) 
        
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

def train_fug_graphcl_ep_aug(model, features, edge_indexs, 
                        sparse, save_name, drop_percent,\
                        lr=0.001, weight_decay=0.0, \
                        start_epoch=0, num_epoch=200, patience=50, gamma=0, model_type=None):
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'edge_indexs', 'sparse', 'save_name'}
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

        # 매 에포크마다 aug 뷰 생성 
        # aug_features, aug_adjs, lbls = process.graphcl_ep_build_aug(features, adjs, sparse, drop_percent)
        
        aug_features, aug_adjs, lbls, negetive_samples, combinedadj = process.preprocess_dataset_w_DE_pyg(
                                                                        features, edge_indexs, 'GRAPHCL', \
                                                                        drop_percent, 0)
        
        
        aug_adjs = [
            [edge_index.cuda() for edge_index in ei_list]
            for ei_list in aug_adjs
        ]
        aug_features = [tensors.cuda() for tensors in aug_features]
        lbls = [tensors.cuda() for tensors in lbls]
        
        if model_type == 'samgpt': 
            aggregate_feat = None 
        else: 
            aggregate_feat = [aggregate_features(aug_features[i][0], aug_adjs[i][0], gamma=gamma) for i in range(len(aug_features))]

        loss = model(aug_features, aug_adjs, sparse, None, None, None, lbls, aggregate_feat) 
        
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
    write(f'    best epoch: {best_t}')
    # arr = [] 
    # for i in range(len(aug_features)):
    #     print(i)
    #     arr.append(model.get_forward(aug_features[i][0], aug_adjs[i][0], sparse, None, False, i))
    # # print('emb[0] > > > ', arr[0].size())
    
    # import numpy as np
    # from sklearn.manifold import TSNE
    # import matplotlib.pyplot as plt 

    # # (1) 임베딩 리스트 arr을 불러옵니다.
    # # arr = [tensor of shape [N1, 256], tensor of shape [N2, 256], ..., tensor of shape [N6, 256]]

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
    # plt.savefig("GNAE_tsne_graph_domains.png", dpi=300)

        
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
    # plt.savefig("GNAE_norm_graph_domains.png", dpi=300)
    
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

