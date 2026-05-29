import csv
import inspect

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from downprompt import *
from preprompt import Sampler
from train import *
from utils import process
from utils.logging_ import write, write_rst


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
    

def adaptation_node(model, features, edge_index, labels, args,
                    sparse, idx_test, nb_classes, dataset, downstream, csv_name):

    write(f"🟩 Function: {inspect.currentframe().f_code.co_name}")
    write(f'   lr     = {args.downlr}')
    write(f'   gamma  = {args.gamma}')
    write(f'   temp   = {args.temp}')

    xent = nn.CrossEntropyLoss()
    patience = 50
    test_lbls = labels[idx_test].cuda()

    train_indices_list = []
    train_labels_list = []
    for i in range(100):
        idx = torch.load(f"data/fewshot_{dataset.lower()}/{args.shot_num}-shot_{dataset.lower()}/{i}/idx.pt").long().cuda()
        lbl = torch.load(f"data/fewshot_{dataset.lower()}/{args.shot_num}-shot_{dataset.lower()}/{i}/labels.pt").long().squeeze().cuda()
        train_indices_list.append(idx)
        train_labels_list.append(lbl)

    model.eval()
    domain_tokens = model.get_weights()
    agg_feat = aggregate_features(features, edge_index)

    accs, macrof, microf = [], [], []

    for i in tqdm(range(100)):
        idx_train = train_indices_list[i]
        lbls_train = train_labels_list[i]

        log = downprompt(
            args.hid_units, nb_classes, args.unify_dim, args.combinetype,
            args.sample_size, args.temp, agg_feat, domain_tokens, args.num_de_layers,
        ).cuda()

        sampler = Sampler(sample_size=args.sample_size, if_rand=args.if_rand, sampling=args.sampling)
        sample = sampler(agg_feat, edge_index, include_idx=idx_train)
        seq = [features, sample]

        log.train()
        opt = torch.optim.Adam(log.parameters(), lr=args.downlr, weight_decay=args.l2_coef)
        cnt_wait, best = 0, 1e9

        for ep in tqdm(range(args.adapt_epochs)):
            opt.zero_grad()
            logits, uniform = log(seq, edge_index, sparse, model.gcn, idx_train, lbls_train, 1)
            loss = xent(logits, lbls_train) + args.gamma * uniform

            if loss < best:
                best = loss
                cnt_wait = 0
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                break

            loss.backward()
            opt.step()

        log.eval()
        logits, _ = log(seq, edge_index, sparse, model.gcn, idx_test, None, 0)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls) / test_lbls.shape[0]
        micro_f1 = f1_score_torch(preds, test_lbls, average='micro') * 100
        macro_f1 = f1_score_torch(preds, test_lbls, average='macro') * 100
        accs.append(acc.item() * 100)
        microf.append(micro_f1)
        macrof.append(macro_f1)
        tqdm.write(f"Iter {i+1} | Acc: {acc.item():.4f}")

        if i % 10 == 0:
            acc_arr = np.array(accs)
            write(f'[{i}] {acc_arr.mean():.2f} ± {acc_arr.std():.2f}')

        with open(f"data/{args.experiment}_{downstream.lower()}_fewshot_node_detailed.csv", "a", newline="") as f:
            csv.writer(f, dialect="excel").writerow([args.seed, args.shot_num, i, acc.item() * 100])

    microf_tensor = torch.stack(microf).cpu().numpy()
    macrof_tensor = torch.stack(macrof).cpu().numpy()
    write_rst(accs, args.shot_num, microf_tensor, macrof_tensor, csv_name, args.seed, 'node')
