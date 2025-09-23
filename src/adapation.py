from utils import process 
import inspect
import torch 
import torch.nn as nn
import numpy as np 
import scipy.sparse as sp
import copy
from tqdm import tqdm 
from models.dimension import DimensionNN_FUG, FUG
from preprompt import Sampler
from downprompt import *
from utils.logging_ import write, write_rst, write_rst2
from sklearn.metrics import f1_score
from utils.figure import * 
from train import *

def adaptation_GraphACL_node(model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, basis_matrix=None, n_mlp_layer=1, sampling='random'):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    downstreamlrlist = [0.001]
    write(f'    a = {a} (loss = xcent + a*de_loss!)')
    emb_test = []
    emb_train = [] 

    for downstreamlr in downstreamlrlist:

        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

        for shotnum in range(1,shot_num+1):
   
            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            dim_pretext_weights, fea_pretext_weights, combines = model.get_weights()
            agg_feat = aggregate_features(features, adj, gamma)

            for i in tqdm(range(100)):
                    
                combines.append(beta)

                log = downpromptFUG(hid_units, nb_classes, unify_dim, num_layers_num, dim_pretext_weights,
                                fea_pretext_weights, None, combines, combinetype,
                                ablation_down, sample_size, if_rand, gamma, basis_matrix, n_mlp_layer, sampling, agg_feat).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                best = 1e9
               
                for ep in tqdm(range(epoch)):
                    opt.zero_grad()

                    # if ep % 10 == 0:
                    #     log.downstreamPrompt.sampler.reset_indices() 

                    logits = log(features,adj,sparse,model.encoder,idx_train,lbls_train,1).float()
                    
                    loss_p = xent(logits, lbls_train)
                    if ablation_down != 'None': 
                        de_loss = log.de_loss()
                        loss = loss_p + a * de_loss
                    else: 
                        loss = loss_p
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

                logits = log(features, adj, sparse, model.encoder, idx_test)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    if 'de_loss' in locals():
                        print(f'loss_p: {loss_p}, de loss: {de_loss}')
                    else: 
                        print(f'logg_p: {loss}')
                    # emb_test.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_test])
                    # emb_train.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_train])

                    
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)
    return emb_test, emb_train , idx_train
    #torch.save(log.state_dict(), save_path)


def f1_score_torch(preds, labels, average='micro'):
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    num_classes = torch.max(labels).item() + 1
    eps = 1e-8

    if average == 'micro':
        true_positive = torch.sum((preds == labels) & (labels >= 0)).float()
        total_predicted = preds.numel()
        total_true = labels.numel()
        precision = true_positive / (total_predicted + eps)
        recall = true_positive / (total_true + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return f1

    elif average == 'macro':
        f1s = []
        for c in range(num_classes):
            tp = ((preds == c) & (labels == c)).sum().float()
            fp = ((preds == c) & (labels != c)).sum().float()
            fn = ((preds != c) & (labels == c)).sum().float()
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1s.append(f1)
        return torch.stack(f1s).mean()
    
def adaptation_samgpt_node(model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, 
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    downstreamlrlist = [0.001]

    for downstreamlr in downstreamlrlist:

        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

        for shotnum in range(1,shot_num+1):
   
            accs = []
            macrof = []
            microf = []
            
            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
            combines.append(beta)

            for i in tqdm(range(100)):
                log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                                fea_pretext_weights, str_pretext_weights, combines, combinetype,
                                ablation_down).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                best = 1e9

                for _ in tqdm(range(epoch)):
                    opt.zero_grad()

                    logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1).float()
                    
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

                logits = log(features, adj, sparse, model.gcn, idx_test)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    print(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    if i % 10 == 0:
                        write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)
    torch.save(log.state_dict(), save_path)

def GraphACL_finetune_node(model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, n_mlp_layer=1, sampling='random'):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    downstreamlrlist = [0.001]
    write(f'    a = {a} (loss = xcent + a*de_loss!)')

    for downstreamlr in downstreamlrlist:

        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

         
        for shotnum in range(1,shot_num+1):
   
            accs = []
            macrof = []
            microf = []
            
            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            dim_pretext_weights, fea_pretext_weights, combines = model.get_weights()
            num_source_domains = len(fea_pretext_weights)

            for i in tqdm(range(100)):
                gcn_copy = copy.deepcopy(model.encoder)
                    
                combines.append(beta)
                
                log = finetune(gcn_copy, hid_units, nb_classes, unify_dim, \
                                num_source_domains, ablation=ablation_down, sample_size=sample_size, \
                                if_rand=if_rand, n_mlp_layer=n_mlp_layer, sampling=sampling).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                best = 1e9
               
                for _ in tqdm(range(epoch)):
                    opt.zero_grad()

                    logits = log(features,adj,sparse,idx_train,lbls_train,1).float()
                    
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

                logits = log(features, adj, sparse, idx_test)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    print(f'logg_p: {loss}')
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)

    return idx_train

def finetune_node(model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, n_mlp_layer=1, sampling='random', model_type='norm_mdgpt'):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    downstreamlrlist = [0.001]
    write(f'    a = {a} (loss = xcent + a*de_loss!)')

    for downstreamlr in downstreamlrlist:

        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

         
        for shotnum in range(1,shot_num+1):
   
            accs = []
            macrof = []
            microf = []
            
            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            if model_type == 'samgpt': 
                fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
            else: 
                dim_pretext_weights, fea_pretext_weights, str_pretext_weights, combines = model.get_weights()
            num_source_domains = len(fea_pretext_weights)

            for i in tqdm(range(100)):
                gcn_copy = copy.deepcopy(model.gcn)
                    
                combines.append(beta)
                
                log = finetune(gcn_copy, hid_units, nb_classes, unify_dim, \
                                num_source_domains, ablation=ablation_down, sample_size=sample_size, \
                                if_rand=if_rand, n_mlp_layer=n_mlp_layer, sampling=sampling).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                best = 1e9
               
                for _ in tqdm(range(epoch)):
                    opt.zero_grad()

                    logits = log(features,adj,sparse,idx_train,lbls_train,1).float()
                    
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

                logits = log(features, adj, sparse, idx_test)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    print(f'logg_p: {loss}')
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)

    return idx_train

def adaptation_FUG_node(model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, basis_matrix=None, \
                            n_mlp_layer=1, sampling='random', model_type=None, shared=False, lr=0.001):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    # downstreamlrlist = [0.001]
    # downstreamlrlist = [0.0003]
    downstreamlrlist = [lr]
    write(f'    a = {a} (loss = xcent + a*de_loss!)')
    emb_test = []
    emb_train = [] 

    for downstreamlr in downstreamlrlist:
        write(f'learning rate: {downstreamlr}')
        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

        for shotnum in range(1,shot_num+1):

            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            model.eval()
            if model_type == 'samgpt': 
                fea_pretext_weights, str_pretext_weights, combines, shared_token = model.get_weights()
                combines.append(beta)
            else: 
                dim_pretext_weights, fea_pretext_weights, balance_weights, combines, shared_token = model.get_weights()
                agg_feat = aggregate_features(features, adj, gamma)


            for i in tqdm(range(100)):
                    
                combines.append(beta)

                if model_type == 'samgpt': 
                    log = downprompt(hid_units, nb_classes, unify_dim, num_layers_num,
                                fea_pretext_weights, str_pretext_weights, combines, combinetype,
                                ablation_down, shared, shared_token).cuda()
                else: 
                    log = downpromptFUG(hid_units, nb_classes, unify_dim, num_layers_num, dim_pretext_weights,
                                fea_pretext_weights, balance_weights, combines, combinetype,
                                ablation_down, sample_size, if_rand, gamma, basis_matrix, n_mlp_layer, sampling, agg_feat, shared, shared_token).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                best = 1e9
               
                for ep in tqdm(range(epoch)):
                    opt.zero_grad()

                    # if ep % 10 == 0:
                    #     log.downstreamPrompt.sampler.reset_indices() 

                    logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1).float()
                    
                    loss_p = xent(logits, lbls_train)
                    if ablation_down in ['None', 'ft']: 
                        loss = loss_p
                    elif ablation_down != 'None': 
                        de_loss = log.de_loss()
                        loss = loss_p + a * de_loss
                    
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

                log.eval()
                logits = log(features, adj, sparse, model.gcn, idx_test)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                # print(log.downstreamPrompt.open_balance_token.get_normalized_weight())
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    if 'de_loss' in locals():
                        print(f'loss_p: {loss_p}, de loss: {de_loss}')
                    else: 
                        print(f'logg_p: {loss}')
                    # emb_test.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_test])
                    # emb_train.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_train])

                    
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)
    return emb_test, emb_train , idx_train

def get_topk_AX_neighbors_weighted(agg_feat, idx_train, lbls_train, k, idx_exclude=None):
    """
    agg_feat: [N, d] AX embedding
    idx_train: Tensor [n_train]
    lbls_train: Tensor [n_train]
    k: top-k
    idx_exclude: indices to exclude (e.g., test set)
    
    Returns:
        new_idx: 확장된 idx (원래 train + pseudo)
        new_lbls: 각 idx의 label
        weights: 각 샘플의 weight (support=1, pseudo=similarity 값)
    """
    N = agg_feat.size(0)
    device = agg_feat.device
    feat_norm = agg_feat / (agg_feat.norm(dim=1, keepdim=True) + 1e-9)

    all_idx = []
    all_lbls = []
    all_weights = []

    for i, v in enumerate(idx_train):
        sim = (feat_norm[v] @ feat_norm.T).clone()  # [N]
        sim[v] = -1e9  # 자기 자신 제외
        if idx_exclude is not None:
            sim[idx_exclude] = -1e9
        
        vals, nbrs = torch.topk(sim, k)
        
        # support node
        all_idx.append(v.unsqueeze(0))
        all_lbls.append(lbls_train[i].unsqueeze(0))
        all_weights.append(torch.tensor([1.0], device=device))  # support weight=1
        
        # pseudo neighbors
        all_idx.append(nbrs)
        all_lbls.append(lbls_train[i].repeat(k))
        # all_weights.append(torch.full((k,), 1.0 / (k), device=device))  
        all_weights.append(vals) # similarity 값이 weight

    new_idx = torch.cat(all_idx)
    new_lbls = torch.cat(all_lbls)
    weights = torch.cat(all_weights)

    return new_idx, new_lbls, weights

def adaptation_barycenter_node(model_type, model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, de_input='x', \
                            n_mlp_layer=1, sampling='random', shared=False, barycenter=None, lr=0.001,
                            basis_matrix=None,  csv_name='', seed=39):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    # downstreamlrlist = [0.001]
    downstreamlrlist = [lr]
    # downstreamlrlist = [0.0003, 0.001]
    # write(f'    a = {a} (loss = xcent + a*de_loss!)')
    write(f'    a = {a} (Tok-k pseudo nodes)')
    write(f'    gamma = {gamma}')
    emb_test = []
    emb_train = [] 

    for downstreamlr in downstreamlrlist:
        write(f'learning rate: {downstreamlr}')
        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

        for shotnum in [shot_num]:
            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            model.eval()

            dimension_encoder_layers,  combines, shared_token = model.get_weights()

            agg_feat = aggregate_features(features, adj, gamma)
            
            for i in tqdm(range(100)):
                    
                combines.append(beta)

                if model_type == 'barycenter': 
                    log = downpromptBarycenter(hid_units, nb_classes, unify_dim, num_layers_num, dimension_encoder_layers,
                        combines, combinetype,
                        ablation_down, sample_size, if_rand, gamma,n_mlp_layer, sampling, agg_feat, shared, shared_token, 
                        de_input, basis_matrix).cuda()
                elif model_type == 'w1mlp': 
                    log = downpromptW1MLP(hid_units, nb_classes, unify_dim, num_layers_num, dimension_encoder_layers,
                        combines, combinetype,
                        ablation_down, sample_size, if_rand, gamma,n_mlp_layer, sampling, agg_feat, shared, shared_token, 
                        de_input).cuda()

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]
                
                # if a != 0.0: 
                #     idx_aug, labels_aug, weights = get_topk_AX_neighbors_weighted(
                #         agg_feat, idx_train, lbls_train, k=int(a), idx_exclude=idx_test
                #     )
                
                if de_input == 'x': 
                    sample = log.downstreamPrompt.sampler(features, adj)
                elif de_input == 'ax': 
                    sample = log.downstreamPrompt.sampler(agg_feat, adj)

                seq = [features, sample]

                best = 1e9

                log.train()

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                # for ep in tqdm(range(100)):
                #     opt.zero_grad()
                #     # logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1, None).float()
                #     # de_loss = log.de_loss()
                #     # loss = de_loss
                #     emb = log.downstreamPrompt.get_emb(features)
                #     loss = wasserstein_distance(emb, barycenter, reg=0.1)
                #     loss.backward()
                #     opt.step()

                # opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                
                for ep in tqdm(range(epoch)):
                    opt.zero_grad()
                    # logits = log(seq,adj,sparse,model.gcn,idx_train, labels_aug,1, None, idx_aug, weights).float()
                    logits = log(seq,adj,sparse,model.gcn,idx_train,lbls_train,1, None).float()
                    
                    # emb = log.downstreamPrompt.get_emb(features)
                    # W1loss = wasserstein_distance(emb, barycenter, reg=0.1)

                    loss = xent(logits, lbls_train)
                    # loss_p = xent(logits, labels_aug)

                    # loss = loss_p 
                    # print(f'loss: {loss:.4f}, W1loss: {W1loss:.4f}')
                    # loss += (W1loss)

                    # loss_p = loss_p + 0.05*log.ave.mean(dim=0).pow(2).mean()
                    # if gamma != 0.0: 
                    #     emb = log.downstreamPrompt.get_composed()
                    #     # emb = log.downstreamPrompt.get_emb_2(features, model.gcn, adj, sparse)
                    #     w1loss = wasserstein_distance(emb, barycenter, reg=0.1)
                    #     loss = loss_p + (gamma*w1loss)
                    # else: 
                    #     loss = loss_p

                    # if a != 0.0: 
                    #     de_loss = log.de_loss()
                    #     loss = loss + a * de_loss
                    #     # emb = log.downstreamPrompt.get_emb(features)
                    #     # w1loss = wasserstein_distance(emb, barycenter, reg=0.1)
                    #     # loss += (0.5*w1loss)

                    # else: 
                    #     loss = loss_p
                    # loss = loss_p

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

                # max_probs, preds = torch.max(logits, dim=1)
                # print(f'train logit: {max_probs.detach()}\ntrain label: {preds.detach()}')

                log.eval()
                logits = log(seq, adj, sparse, model.gcn, idx_test, None, 0, None)
                # preds = torch.argmax(logits, dim=1)
                max_probs, preds = torch.max(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                # tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f} | ✅: {logits[preds==test_lbls].std(dim=1).mean(dim=0):.4f}, ❌: {logits[~(preds==test_lbls)].std(dim=1).mean(dim=0):.4f}")
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f} | ✅: {max_probs[preds==test_lbls].mean().item():.4f}, ❌: {max_probs[~(preds==test_lbls)].mean().item():.4f}")
                # print(log.downstreamPrompt.open_balance_token.get_normalized_weight())
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    if 'de_loss' in locals():
                        print(f'loss_p: {loss_p}, de loss: {de_loss}')#, w1:{w1loss}')
                    elif 'w1loss' in locals():
                        print(f'loss: {loss}, loss_p: {loss_p}, W1 loss: {w1loss}')#, w1:{w1loss}')
                    elif 'entropy' in locals(): 
                        print(f'loss: {loss}, loss_p: {loss_p}, entropy: {entropy}')
                    else: 
                        print(f'logg_p: {loss}')
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor, csv_name, seed, 'node')

def adaptation_barycenter_graph(model_type, model, features, adj, labels, test_index, test_batch, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, de_input='x', \
                                n_mlp_layer=1, sampling='random', shared=False, barycenter=None, lr=0.001, basis_matrix=None, csv_name='', seed=39):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix', 'test_batch', 'test_index'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    # downstreamlrlist = [0.001]
    downstreamlrlist = [lr]
    # downstreamlrlist = [0.0003, 0.001]
    # write(f'    a = {a} (loss = xcent + a*de_loss!)')
    write(f'    a = {a} (Tok-k pseudo nodes)')
    write(f'    gamma = {gamma}')


    for downstreamlr in downstreamlrlist:
        write(f'learning rate: {downstreamlr}')
        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

        for shotnum in [shot_num]:
            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            train_batch_list = []

            for i in range(100):
                idx_train = torch.load(f"data/fewshot_{dataset.lower()}_graph/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                batch_train = torch.load(f"data/fewshot_{dataset.lower()}_graph/{shotnum}-shot_{dataset.lower()}/{i}/batch.pt").long().cuda()
                lbls_train = torch.load(f"data/fewshot_{dataset.lower()}_graph/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().cuda()

                train_indices_list.append(idx_train)
                train_batch_list.append(batch_train)
                train_labels_list.append(lbls_train)

            model.eval()

            dimension_encoder_layers,  combines, shared_token = model.get_weights()

            agg_feat = aggregate_features(features, adj, gamma)
            
            # eigvecs, eigvals = spectral_energy_distribution(features, adj, num_nodes=None, k=None)
            # sample = spectral_energy_histogram(features, adj, n_bins=sample_size)
            # sample = sample.T # [k, d]
            # features = [features, sample]

            for i in tqdm(range(100)):
                    
                combines.append(beta)

                if model_type == 'barycenter': 
                    log = downpromptBarycenter(hid_units, nb_classes, unify_dim, num_layers_num, dimension_encoder_layers,
                        combines, combinetype,
                        ablation_down, sample_size, if_rand, gamma,n_mlp_layer, sampling, agg_feat, shared, shared_token, 
                        de_input, basis_matrix).cuda()
                elif model_type == 'w1mlp': 
                    log = downpromptW1MLP(hid_units, nb_classes, unify_dim, num_layers_num, dimension_encoder_layers,
                        combines, combinetype,
                        ablation_down, sample_size, if_rand, gamma,n_mlp_layer, sampling, agg_feat, shared, shared_token, 
                        de_input).cuda()

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]
                batch_train = train_batch_list[i]

                if a != 0.0: 
                    idx_aug, labels_aug, weights = get_topk_AX_neighbors_weighted(
                        agg_feat, idx_train, lbls_train, k=int(a), idx_exclude=idx_test
                    )
                
                if de_input == 'x': 
                    sample = log.downstreamPrompt.sampler(features, adj)
                elif de_input == 'ax': 
                    sample = log.downstreamPrompt.sampler(agg_feat, adj)

                seq = [features, sample]

                best = 1e9

                log.train()

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                # for ep in tqdm(range(100)):
                #     opt.zero_grad()
                #     # logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1, None).float()
                #     # de_loss = log.de_loss()
                #     # loss = de_loss
                #     emb = log.downstreamPrompt.get_emb(features)
                #     loss = wasserstein_distance(emb, barycenter, reg=0.1)
                #     loss.backward()
                #     opt.step()

                # opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                
                for ep in tqdm(range(epoch)):
                    opt.zero_grad()
                    if a != 0.0: 
                        logits = log(seq,adj,sparse,model.gcn,idx_train, labels_aug,1, batch_train, idx_aug, weights).float()
                    else: 
                        logits = log(seq,adj,sparse,model.gcn,idx_train,lbls_train,1, batch_train).float()
                    
                    # emb = log.downstreamPrompt.get_emb(features)
                    # W1loss = wasserstein_distance(emb, barycenter, reg=0.1)

                    loss_p = xent(logits, lbls_train)
                    # loss_p = xent(logits, labels_aug)

                    loss = loss_p 
                    # print(f'loss: {loss:.4f}, W1loss: {W1loss:.4f}')
                    # loss += (W1loss)

                    # loss_p = loss_p + 0.05*log.ave.mean(dim=0).pow(2).mean()
                    # if gamma != 0.0: 
                    #     emb = log.downstreamPrompt.get_composed()
                    #     # emb = log.downstreamPrompt.get_emb_2(features, model.gcn, adj, sparse)
                    #     w1loss = wasserstein_distance(emb, barycenter, reg=0.1)
                    #     loss = loss_p + (gamma*w1loss)
                    # else: 
                    #     loss = loss_p

                    # if a != 0.0: 
                    #     de_loss = log.de_loss()
                    #     loss = loss + a * de_loss
                    #     # emb = log.downstreamPrompt.get_emb(features)
                    #     # w1loss = wasserstein_distance(emb, barycenter, reg=0.1)
                    #     # loss += (0.5*w1loss)

                    # else: 
                    #     loss = loss_p
                    # loss = loss_p

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

                # max_probs, preds = torch.max(logits, dim=1)
                # print(f'train logit: {max_probs.detach()}\ntrain label: {preds.detach()}')

                log.eval()
                logits = log(seq, adj, sparse, model.gcn, test_index, None, 0, batch=test_batch)
                # preds = torch.argmax(logits, dim=1)
                max_probs, preds = torch.max(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                # tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f} | ✅: {logits[preds==test_lbls].std(dim=1).mean(dim=0):.4f}, ❌: {logits[~(preds==test_lbls)].std(dim=1).mean(dim=0):.4f}")
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f} | ✅: {max_probs[preds==test_lbls].mean().item():.4f}, ❌: {max_probs[~(preds==test_lbls)].mean().item():.4f}")
                # print(log.downstreamPrompt.open_balance_token.get_normalized_weight())
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    if 'de_loss' in locals():
                        print(f'loss_p: {loss_p}, de loss: {de_loss}')#, w1:{w1loss}')
                    elif 'w1loss' in locals():
                        print(f'loss: {loss}, loss_p: {loss_p}, W1 loss: {w1loss}')#, w1:{w1loss}')
                    elif 'entropy' in locals(): 
                        print(f'loss: {loss}, loss_p: {loss_p}, entropy: {entropy}')
                    else: 
                        print(f'logg_p: {loss}')
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor, csv_name, seed, 'graph')

def adaptation_sharedFUG_node(model_type, model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, basis_matrix=None, \
                                n_mlp_layer=1, sampling='random', shared=False, barycenter=None, lr=0.001):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    # downstreamlrlist = [0.001]
    downstreamlrlist = [lr]
    # downstreamlrlist = [0.0003, 0.001]
    write(f'    a = {a} (loss = xcent + a*de_loss!)')
    emb_test = []
    emb_train = [] 

    for downstreamlr in downstreamlrlist:
        write(f'learning rate: {downstreamlr}')
        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

        for shotnum in range(1,shot_num+1):

            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            model.eval()

            if model_type == 'sharedFUG': 
                dim_pretext_weights, shared_dimension_encoder, balance_weights, combines = model.get_weights()
            elif model_type == 'filterFUG':
                #high_dimension_encoder, low_dimension_encoder, high domain token, combines, shared domain token = model.get_weights()
                dim_pretext_weights, shared_dimension_encoder, balance_weights, combines, shared_token = model.get_weights()
            elif model_type == 'filterbank': 
                # high_dimension_encoder, low_dimension_encoder, identity_dimension_encoder, combines = model.get_weights()
                dim_pretext_weights, shared_dimension_encoder, balance_weights, combines, shared_token = model.get_weights()
            
            agg_feat = aggregate_features(features, adj, gamma)
            
            

            for i in tqdm(range(100)):
                    
                combines.append(beta)

                log = downpromptSharedFUG(model_type, hid_units, nb_classes, unify_dim, num_layers_num, dim_pretext_weights,
                            shared_dimension_encoder, balance_weights, combines, combinetype,
                            ablation_down, sample_size, if_rand, gamma, basis_matrix, n_mlp_layer, sampling, agg_feat, shared, shared_token).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=0.001)
                best = 1e9
                
                # if barycenter != None: 
                for ep in tqdm(range(50)):
                    opt.zero_grad()

                    emb = log.downstreamPrompt.get_emb(features, adj)
                    loss = wasserstein_distance(emb, barycenter, reg=0.1)

                    loss.backward()
                    opt.step()

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)

                for ep in tqdm(range(epoch)):
                    opt.zero_grad()

                    # if ep % 10 == 0:
                    #     log.downstreamPrompt.sampler.reset_indices() 

                    logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1).float()
                    
                    loss_p = xent(logits, lbls_train)
                    if ablation_down != 'None': 
                        de_loss = log.de_loss()
                        loss = loss_p + a * de_loss
                    else: 
                        loss = loss_p
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
                
                log.eval()
                logits = log(features, adj, sparse, model.gcn, idx_test)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                # print(log.downstreamPrompt.open_balance_token.get_normalized_weight())
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    if 'de_loss' in locals():
                        print(f'loss_p: {loss_p}, de loss: {de_loss}')
                    else: 
                        print(f'logg_p: {loss}')
                    # emb_test.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_test])
                    # emb_train.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_train])

                    
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)
    return emb_test, emb_train , idx_train
    #torch.save(log.state_dict(), save_path)



def adaptation_FUG_graph(model, features, adj, labels, test_index, test_batch, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, basis_matrix=None, n_mlp_layer=1, \
                            ):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    downstreamlrlist = [0.001]
    write(f'    a = {a} (loss = xcent + a*de_loss!)')

    for downstreamlr in downstreamlrlist:

        test_lbls = labels[idx_test].cuda()
        print('-' * 100)
         
        for shotnum in range(1,shot_num+1):
   
            accs = []
            macrof = []
            microf = []
            
            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            train_batch_list = []
            for i in range(100):
                idx_train = torch.load(f"data/fewshot_{dataset.lower()}_graph/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                batch_train = torch.load(f"data/fewshot_{dataset.lower()}_graph/{shotnum}-shot_{dataset.lower()}/{i}/batch.pt").long().cuda()
                lbls_train = torch.load(f"data/fewshot_{dataset.lower()}_graph/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().cuda()

                train_indices_list.append(idx_train)
                train_batch_list.append(batch_train)
                train_labels_list.append(lbls_train)

            dim_pretext_weights, fea_pretext_weights, str_pretext_weights, combines = model.get_weights()

            for i in tqdm(range(100)):
                    
                combines.append(beta)

                log = downpromptFUG(hid_units, nb_classes, unify_dim, num_layers_num, dim_pretext_weights,
                                fea_pretext_weights, str_pretext_weights, combines, combinetype,
                                ablation_down, sample_size, if_rand, gamma, basis_matrix, n_mlp_layer).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]
                batch_train = train_batch_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                best = 1e9
               
                for _ in tqdm(range(epoch)):
                    opt.zero_grad()

                    logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1,batch_train).float()

                    loss_p = xent(logits, lbls_train)
                    if ablation_down != 'None': 
                        de_loss = log.de_loss()
                        loss = loss_p + a * de_loss
                    else: 
                        loss = loss_p
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

                logits = log(features, adj, sparse, model.gcn, test_index, batch=test_batch)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    if 'de_loss' in locals():
                        print(f'loss_p: {loss_p}, de loss: {de_loss}')
                    else: 
                        print(f'logg_p: {loss}')

                    
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)

def adaptation_FUG_first_node(model, features, adj, labels, \
                           sparse, idx_test, shot_num, dataset, \
                           beta, hid_units, nb_classes, unify_dim, num_layers_num,\
                            combinetype, ablation_down, patience, save_path, epoch=400, \
                            sample_size = 182, if_rand=False, a=0.0, gamma=0.5, basis_matrix=None, n_mlp_layer=1, sampling='random'):
    
    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    excluded = {'model', 'features', 'adj', 'labels', 'idx_test', 'save_name', 'basis_matrix'}
    for arg in args:
        if arg not in excluded:
            print(f"   └ {arg} = {values[arg]}")

    xent = nn.CrossEntropyLoss()
    downstreamlrlist = [0.001]
    write(f'    a = {a} (loss = xcent + a*de_loss!)')
    emb_test = []
    emb_train = [] 

    for downstreamlr in downstreamlrlist:

        test_lbls = labels[idx_test].cuda()
        accs = []
        macrof = []
        microf = []
        print('-' * 100)

        for shotnum in range(1,shot_num+1):

            cnt_wait = 0
            best = 1e9
            print("shotnum",shotnum)

            train_indices_list = []
            train_labels_list = []
            for i in range(100):
                idx = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
                lbl = torch.load(f"data/fewshot_{dataset.lower()}/{shotnum}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
                train_indices_list.append(idx)
                train_labels_list.append(lbl)

            model.eval()
            dim_pretext_weights, fea_pretext_weights, balance_weights, combines, shared_token = model.get_weights()
            agg_feat = aggregate_features(features, adj, gamma)
            
            from downprompt import compute_x_ax_weight,compute_degree_weight, compute_variance_weight
            a, b = compute_x_ax_weight(features, agg_feat)
            # a,b = compute_degree_weight(adj, features.shape[0])
            # a, b = compute_variance_weight(features, agg_feat)
            print(a, b)

            for i in tqdm(range(100)):
                    
                combines.append(beta)

                log = downpromptFUG(hid_units, nb_classes, unify_dim, num_layers_num, dim_pretext_weights,
                                fea_pretext_weights, balance_weights, combines, combinetype,
                                ablation_down, sample_size, if_rand, gamma, basis_matrix, n_mlp_layer, sampling, agg_feat, shared_token).cuda()
                log.train()
 

                idx_train = train_indices_list[i]
                lbls_train = train_labels_list[i]

                opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
                best = 1e9
               
                # DE 먼저 
                for ep in tqdm(range(epoch)):
                    opt.zero_grad()

                    # if ep % 10 == 0:
                    #     log.downstreamPrompt.sampler.reset_indices() 
                
                    logits = log(features,adj,sparse,model.gcn,idx_train,lbls_train,1).float()
                    
                    loss_p = xent(logits, lbls_train)
                    if ablation_down != 'None': 
                        de_loss = log.de_loss()
                        loss = loss_p + a * de_loss
                    else: 
                        loss = loss_p
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

                logits = log(features, adj, sparse, model.gcn, idx_test)

                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
                micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
                macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
                microf.append(micro_f1)
                macrof.append(macro_f1)
                accs.append(acc.item() * 100)
                tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")
                # print(log.downstreamPrompt.open_balance_token.get_normalized_weight())
                if i % 10  == 0: 
                    acc_arr = np.array(accs)
                    write(f'[{i}]{acc_arr.mean():.2f} ± {acc_arr.std():.2f}')
                    print(len(accs))
                    if 'de_loss' in locals():
                        print(f'loss_p: {loss_p}, de loss: {de_loss}')
                    else: 
                        print(f'logg_p: {loss}')
                    # emb_test.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_test])
                    # emb_train.append(log.get_emb(features, model.gcn, adj, sparse).detach()[idx_train])

                    
        
        microf_tensor = torch.stack(microf).cpu().numpy()  # shape: (N,)
        macrof_tensor = torch.stack(macrof).cpu().numpy()
        write_rst(accs, shotnum, microf_tensor, macrof_tensor)
    return emb_test, emb_train , idx_train
    #torch.save(log.state_dict(), save_path)

                  