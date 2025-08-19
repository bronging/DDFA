
import os

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
import torch.nn as nn
import pandas as pd

import numpy as np
# import matplotlib.pyplot as plt
from sklearn import manifold
from preprompt import pca_compression, prompt_pretrain_sample
import utils.aug as aug 
import gc 

import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_undirected

# 그래프별 Homophily Ratio 계산 함수
def compute_homophily_distribution(data, N_bins=10):
    edge_index = to_undirected(data.edge_index)
    labels = data.y
    num_nodes = data.num_nodes
    homophily_ratios = torch.zeros(num_nodes)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]
        if len(neighbors) == 0:
            homophily_ratios[node] = 0.0
            continue
        same_label_count = (labels[neighbors] == labels[node]).sum().item()
        homophily_ratios[node] = same_label_count / len(neighbors)

    bins = np.linspace(0, 1, N_bins + 1)
    counts, _ = np.histogram(homophily_ratios.numpy(), bins=bins)
    return counts, bins, num_nodes

# 그래프별 homo distribution 시각화 함수
def plot_distribution(counts, bins, total_nodes, dataset_name):
    bin_labels = [f'{round(bins[i], 1)}–{round(bins[i+1], 1)}' for i in range(len(bins)-1)]
    percentages = (counts / total_nodes) * 100

    plt.figure(figsize=(8, 5))
    bars = plt.bar(bin_labels, counts, width=0.8)

    # 막대 위에 비율 텍스트 추가
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{pct:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45)
    plt.xlabel('Homophily Ratio Range')
    plt.ylabel('Number of Nodes')
    plt.title(f'Homophily Ratio Distribution ({dataset_name})')
    plt.tight_layout()
    plt.savefig(f"homophily_distribution_{dataset_name}.png", dpi=300)

# 각 그래프의 homophily ratio 분포를 저장
def compute_homophily_histogram(data, N_bins=10):
    edge_index = to_undirected(data.edge_index)
    labels = data.y
    num_nodes = data.num_nodes
    homophily_ratios = torch.zeros(num_nodes)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]
        if len(neighbors) == 0:
            homophily_ratios[node] = 0.0
            continue
        same_label_count = (labels[neighbors] == labels[node]).sum().item()
        homophily_ratios[node] = same_label_count / len(neighbors)

    bins = np.linspace(0, 1, N_bins + 1)
    counts, _ = np.histogram(homophily_ratios.numpy(), bins=bins)
    return counts, bins, num_nodes

# 여러 그래프의 homo distributions 누적 막대 그래프 그리기
def plot_stacked_distribution(histories, bins, dataset_names, total_nodes_list, extra_str='_1hop'):
    bin_labels = [f'{round(bins[i], 1)}–{round(bins[i+1], 1)}' for i in range(len(bins)-1)]
    N_bins = len(bin_labels)

    counts_array = np.array(histories)  # shape: [num_graphs, N_bins]
    total_sum = counts_array.sum(axis=0)
    percentages = (total_sum / total_sum.sum()) * 100

    # 누적 막대
    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(N_bins)

    for i, (counts, name) in enumerate(zip(counts_array, dataset_names)):
        ax.bar(bin_labels, counts, bottom=bottoms, label=name)
        bottoms += counts

    # 각 bin 위에 전체 비율 표시
    for i in range(N_bins):
        ax.text(i, bottoms[i] + 1, f'{percentages[i]:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Homophily Ratio Range")
    ax.set_ylabel("Number of Nodes (Stacked)")
    ax.set_title(f"Stacked Homophily Ratio Distribution Across Graphs {extra_str}")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"homophily_distribution_{dataset_names}_{extra_str}.png", dpi=300)

# 디그리별  homo distribution 
def compute_homophily_ratios_deg(data):
    edge_index = to_undirected(data.edge_index)
    labels = data.y
    num_nodes = data.num_nodes
    homophily_ratios = torch.zeros(num_nodes)
    degrees = torch.zeros(num_nodes, dtype=torch.long)

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node]
        degrees[node] = len(neighbors)
        if len(neighbors) == 0:
            homophily_ratios[node] = 0.0
        else:
            same_label_count = (labels[neighbors] == labels[node]).sum().item()
            homophily_ratios[node] = same_label_count / len(neighbors)

    return homophily_ratios, degrees

# 디그리별 homo distribution 그래프 그리기. - 각 디그리별로 정규화해서 그림 
def group_and_plot_by_degree_normalized(homophily_ratios, degrees, dataset_name, N_bins=10):
    # Degree 구간 정의
    degree_bins = [1, 5, 10, 20, 50, 100, 99999]
    group_labels = [f'{degree_bins[i]}–{degree_bins[i+1]}' for i in range(len(degree_bins) - 1)]

    hist_per_group = []
    node_counts = []

    bins = np.linspace(0, 1, N_bins + 1)

    for i in range(len(degree_bins) - 1):
        mask = (degrees >= degree_bins[i]) & (degrees < degree_bins[i+1])
        selected_ratios = homophily_ratios[mask]
        count = selected_ratios.shape[0]
        node_counts.append(count)

        if count > 0:
            hist, _ = np.histogram(selected_ratios.numpy(), bins=bins)
            hist = hist / hist.sum()  # ✅ 정규화: 그룹 내 비율
        else:
            hist = np.zeros(N_bins, dtype=float)
        hist_per_group.append(hist)

    # ✅ 시각화
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})

    # (1) 정규화된 Homophily Ratio 분포
    bin_labels = [f'{round(bins[i], 1)}–{round(bins[i+1], 1)}' for i in range(N_bins)]
    x = np.arange(len(bin_labels))

    for hist, label in zip(hist_per_group, group_labels):
        axs[0].plot(x, hist * 100, label=f'Degree {label}')  # ✅ y축을 %로 보기 좋게

    axs[0].set_xticks(x)
    axs[0].set_xticklabels(bin_labels, rotation=45)
    axs[0].set_ylabel('Group-Internal % of Nodes')
    axs[0].set_title(f'Homophily Ratio Distribution (Normalized) by Degree Group – {dataset_name}')
    axs[0].legend()

    # (2) Degree 그룹별 노드 수
    axs[1].bar(group_labels, node_counts, color='gray')
    axs[1].set_ylabel("Node Count")
    axs[1].set_xlabel("Degree Group")
    axs[1].set_title("Number of Nodes per Degree Group")

    plt.tight_layout()
    plt.savefig(f"{dataset_name}_homophily_distribution_degree.png", dpi=300)

# 2hop homophily distribution 
from collections import defaultdict
def compute_2hop_homophily_distribution(data, N_bins=10):
    edge_index = to_undirected(data.edge_index)
    labels = data.y
    num_nodes = data.num_nodes

    # 1-hop neighbor 저장
    neighbors_dict = defaultdict(set)
    for src, dst in edge_index.t().tolist():
        neighbors_dict[src].add(dst)

    homophily_ratios = torch.zeros(num_nodes)

    for node in range(num_nodes):
        one_hop = neighbors_dict[node]
        two_hop = set()

        for n in one_hop:
            two_hop.update(neighbors_dict[n])

        # 자신과 1-hop 제거
        two_hop.discard(node)
        two_hop.difference_update(one_hop)

        two_hop = list(two_hop)

        if len(two_hop) == 0:
            homophily_ratios[node] = 0.0
        else:
            same_label_count = (labels[two_hop] == labels[node]).sum().item()
            homophily_ratios[node] = same_label_count / len(two_hop)

    # Histogram bin 계산
    bins = np.linspace(0, 1, N_bins + 1)
    counts, _ = np.histogram(homophily_ratios.numpy(), bins=bins)
    return counts, bins, num_nodes

def get_labels(pretrain_loaders, pretrain_dataset_names, target_graph_id): 
 
    labels = [] 

    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
        if (step+1) not in target_graph_id:
            print(datas)
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            labels.append(data.y)
            print(f'{pretrain_dataset_name}: {data.y.shape}, {data.y[:10]}')

    return labels

from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import degree

def calculate_similarity_proportions(x, edge_index, labels, k, device='cpu'):
    """
    이 함수는 X, AX, (I-A)X 및 무작위 기준선에 대해 상위 k개 유사 이웃 중 같은 클래스 노드의 비율을 계산하고 출력합니다.

    Args:
        x (torch.Tensor): 노드 피처.
        edge_index (torch.Tensor): 그래프의 엣지 인덱스.
        labels (torch.Tensor): 노드 레이블.
        k (int): 고려할 상위 유사 이웃의 수.
        device (str): 계산을 실행할 장치 ('cpu' 또는 'cuda').
    """

    # 텐서를 지정된 장치로 이동
    x = x.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)

    # --- 1단계: 인접 행렬 및 필터링된 피처 준비 ---
    num_nodes = x.size(0)

    # 자체 루프(self-loops)를 추가하여 A_hat을 얻기
    edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    # 자체 루프가 포함된 밀집 인접 행렬 생성
    adj_with_loops = to_dense_adj(edge_index_with_loops, max_num_nodes=num_nodes).squeeze(0)

    # 정규화를 위한 차수 행렬 계산
    deg = degree(edge_index_with_loops[0], num_nodes=num_nodes, dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    deg_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)

    # 대칭 정규화 인접 행렬(A_hat) 계산
    a_hat = deg_inv_sqrt_matrix @ adj_with_loops @ deg_inv_sqrt_matrix

    # 필터링된 피처 계산
    ax_hat = torch.matmul(a_hat, x)
    i_minus_a_hat = torch.eye(num_nodes, device=device) - a_hat
    i_minus_a_hat_x = torch.matmul(i_minus_a_hat, x)

    # 분석할 피처 세트를 딕셔너리에 저장
    feature_sets = {
        'X': x,
        'AX_hat': ax_hat,
        'I_minus_A_hat_X': i_minus_a_hat_x
    }
    
    # --- 2단계: 통계 계산을 위한 헬퍼 함수 정의 ---
    def compute_stats(feature_matrix, name, largest_bool=True):
        # 코사인 유사도 행렬 계산
        # 0-norm 벡터의 경우 0으로 나누는 것을 피하기 위해 작은 엡실론 추가
        norms = torch.norm(feature_matrix, p=2, dim=1, keepdim=True) + 1e-8
        normalized_features = feature_matrix / norms
        similarity_matrix = torch.matmul(normalized_features, normalized_features.T)

        same_class_proportions = torch.zeros(num_nodes, device=device)

        # 각 노드에 대해, 상위 k개 유사 이웃을 찾고 같은 클래스 비율 계산
        for i in range(num_nodes):
            # 유사도 점수에서 자기 자신 제외
            similarity_matrix[i, i] = -1.0
            
            # 상위 k개 가장 유사한 이웃의 인덱스 가져오기
            top_k_indices = torch.topk(similarity_matrix[i], k=k, largest=largest_bool).indices
            
            # 이웃이 같은 클래스인지 확인하고 비율 계산
            same_class_count = (labels[top_k_indices] == labels[i]).sum().item()
            same_class_proportions[i] = same_class_count / k

        # 클래스별로 결과를 그룹화하고 평균 비율 계산
        class_proportions = {}
        unique_classes = torch.unique(labels)
        for c in unique_classes:
            class_indices = (labels == c)
            avg_proportion = same_class_proportions[class_indices].mean().item()
            num_nodes_in_class = class_indices.sum().item()
            class_proportions[int(c.item())] = avg_proportion

        if largest_bool:
            print(f"\n--- {name} Similarity (Top {k} Most SIMILAR) ---")
        else:
            print(f"\n--- {name} Similarity (Top {k} Most DISSIMILAR) ---")
        for cls, avg_prop in class_proportions.items():
            print(f"Class {cls} (Nodes: {num_nodes_in_class:5d}): Same Class Proportion = {avg_prop:.4f}")
        # 전체 노드에 대한 평균 같은 클래스 비율 출력
        overall_avg_proportion = same_class_proportions.mean().item()
        print(f"Overall Average: {overall_avg_proportion:.4f}")

    # --- 3단계: 각 피처 세트에 대해 분석 실행 ---
    for name, features in feature_sets.items():
        compute_stats(features, name)
    compute_stats(i_minus_a_hat_x, 'I_minus_A_hat_X', largest_bool=False)

    # --- 4단계: 무작위 기준선 계산 ---
    random_proportions = torch.zeros(num_nodes, device=device)
    for i in range(num_nodes):
        # 무작위로 k개의 다른 노드 선택
        random_indices = torch.randperm(num_nodes - 1, device=device)[:k]
        
        # 이웃이 같은 클래스인지 확인하고 비율 계산
        random_proportions[i] = (labels[random_indices] == labels[i]).sum().item() / k

    # 무작위 기준선 결과를 클래스별로 그룹화하고 평균 계산
    random_class_proportions = {}
    unique_classes = torch.unique(labels)
    for c in unique_classes:
        class_indices = (labels == c)
        avg_proportion = random_proportions[class_indices].mean().item()
        random_class_proportions[int(c.item())] = avg_proportion

    print(f"\n--- Random Baseline (Top {k} Neighbors) ---")
    for cls, avg_prop in random_class_proportions.items():
        print(f"Class {cls}: Same Class Proportion = {avg_prop:.4f}")
    overall_avg_proportion = random_proportions.mean().item()
    print(f"Overall Average: {overall_avg_proportion:.4f}")
        
def get_features_adjs(pretrain_loaders,  \
                       cache_dir, pretrain_dataset_names, target_graph_id, ):
    features = []
    adjs = []
    edge_indexs = [] 

    histories = []
    total_nodes_list = []

    histories_2hop = []
    total_nodes_list_2hop = []

    fig = False 
    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
        if (step+1) not in target_graph_id:
            print(datas)
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            feature, adj = process_tu(data, data.x.shape[1])
            features.append(feature)
            adjs.append(adj)
            edge_indexs.append(data.edge_index)
            # print(f'====={pretrain_dataset_name}=====')
            # calculate_similarity_proportions(data.x, data.edge_index, data.y, k=10)
            # print()
            # print(f"전체 노드 개수: {data.num_nodes}")
            # unique_labels, counts = torch.unique(data.y, return_counts=True)
            # # 클래스별 노드 개수 출력
            # for label, count in zip(unique_labels, counts):
            #     print(f"Class {label.item()}: {count.item()} nodes")
        

            if fig:
                # 그래프 별 homophily distribution 
                counts, bins, total_nodes = compute_homophily_distribution(data, N_bins=10)
                plot_distribution(counts, bins, total_nodes, pretrain_dataset_name)

                histories.append(counts)
                total_nodes_list.append(total_nodes)

                # 그래프 별 2hop homophily distribution 
                counts, bins, total_nodes = compute_2hop_homophily_distribution(data, N_bins=10)
                plot_distribution(counts, bins, total_nodes, pretrain_dataset_name+"_2hop")
                
                histories_2hop.append(counts)
                total_nodes_list_2hop.append(total_nodes)
                
                # 그래프 별 디그리별 homo distribution 
                homophily_ratios, degrees = compute_homophily_ratios_deg(data)
                group_and_plot_by_degree_normalized(homophily_ratios, degrees, dataset_name=pretrain_dataset_name, N_bins=10)
    if fig:
        # 누적 homopily distribution 
        plot_stacked_distribution(histories, bins, pretrain_dataset_names, total_nodes_list)
        plot_stacked_distribution(histories_2hop, bins, pretrain_dataset_names, total_nodes_list, '_2hop')
    return features, adjs, edge_indexs


def graphcl_ep_build_aug(features, adjs, sparse, drop_percent):
    aug_adjs = []
    aug_features = []
    lbls = []

    for i in range(len(features)):
        feature = features[i]
        adj = adjs[i]

        aug_feature, aug_adj, lbl = aug.build_aug(adj, feature, sparse, drop_percent)
        
        aug_features.append(aug_feature)
        aug_adjs.append(aug_adj)
        lbls.append(lbl)
        
    return aug_features, aug_adjs, lbls 

def preprocess_dataset_w_DE_pyg(features, edge_indices, pretrain_method,
                                 drop_percent, negative_samples_num):

    aug_edge_indices = []
    aug_features = []
    combined_edge_index = None
    negative_samples = []
    lbls = []

    for i in range(len(features)):
        x = features[i]
        edge_index = edge_indices[i]

        if pretrain_method == 'GRAPHCL':
            aug_feature, aug_edge_index_list, lbl = aug.build_aug_pyg(x, edge_index, drop_percent)

            aug_features.append(aug_feature)
            aug_edge_indices.append(aug_edge_index_list)  # shape: [3, 2, num_edges]
            lbls.append(lbl)

            del aug_feature, aug_edge_index_list, lbl
            gc.collect()

        if pretrain_method == 'splitLP':
            negative_sample = prompt_pretrain_sample(edge_index, 50)  # NOTE: assumes edge_index input
            negative_samples.append(negative_sample)

        # edge_index is already normalized in PyG, no need for explicit normalization
        del x, edge_index
        gc.collect()

    return aug_features, aug_edge_indices, lbls, negative_samples, combined_edge_index

def preprocess_dataset_w_DE(features, adjs, pretrain_method, \
                       sparse, drop_percent, negative_samples_num):

    aug_adjs = []
    aug_features = []
    combinedadj, negetive_samples = [], []
    lbls = []


    for i in range(len(features)):
        feature = features[i]
        adj = adjs[i]
            
        if pretrain_method == 'GRAPHCL':
            aug_feature, aug_adj, lbl = aug.build_aug(adj, feature, sparse, drop_percent)
            
            aug_features.append(aug_feature)
            aug_adjs.append(aug_adj)
            lbls.append(lbl)

            del aug_feature, aug_adj, lbl
            gc.collect()

        if pretrain_method == 'splitLP':                
            negetive_sample = prompt_pretrain_sample(adj, 50)
            negetive_samples.append(negetive_sample)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        # features.append(feature)
        # adjs.append(adj)

        del feature, adj
        gc.collect()

    if pretrain_method == 'LP':    
        combinedadj = combine_dataset_list_sp(adjs)
        print('combinedadj', combinedadj.shape)
        negetive_samples = prompt_pretrain_sample(combinedadj, negative_samples_num)

    return aug_features, aug_adjs, lbls, negetive_samples, combinedadj

def preprocess_dataset(pretrain_loaders, pretrain_method, \
                       cache_dir, pretrain_dataset_names, target_graph_id, \
                       sparse, drop_percent, negative_samples_num, unify_dim=50):
    features = []
    adjs = []
    aug_adjs = []
    aug_features = []
    combinedadj, negetive_samples = [], []
    lbls = []


    for step, datas in enumerate(zip(*pretrain_loaders)):
        print('step', step)
        if (step+1) not in target_graph_id:
            print(datas)
            continue
        for pretrain_dataset_name, data in zip(pretrain_dataset_names, datas):
            if not(os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_feature.pt') and \
                os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_adj.pt') ):
                feature, adj = process_tu(data,data.x.shape[1])
                feature = torch.FloatTensor(pca_compression(feature,k=unify_dim))
                torch.save(feature, f'{cache_dir}/{pretrain_dataset_name}_feature.pt')
                torch.save(adj, f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            feature, adj = torch.load(f'{cache_dir}/{pretrain_dataset_name}_feature.pt'), \
                torch.load(f'{cache_dir}/{pretrain_dataset_name}_adj.pt')
            
#GRAPHCL:
            if pretrain_method == 'GRAPHCL':
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
            if pretrain_method == 'splitLP':
                if not os.path.exists(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt'):
                    negetive_sample = prompt_pretrain_sample(adj, 50)
                    torch.save(negetive_sample, f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_sample = torch.load(f'{cache_dir}/{pretrain_dataset_name}_negetive_sample.pt')
                negetive_samples.append(negetive_sample)

            adj = normalize_adj(adj + sp.eye(adj.shape[0]))
            features.append(feature)
            adjs.append(adj)

            del feature, adj
            gc.collect()

        if pretrain_method == 'LP':    
            combinedadj = combine_dataset_list_sp(adjs)
            print('combinedadj', combinedadj.shape)
            negetive_samples = prompt_pretrain_sample(combinedadj, negative_samples_num)

    return features, adjs, aug_features, aug_adjs, lbls, negetive_samples, combinedadj

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


def sp_adj(adj,node1,node2):
    begin = 0
    for i in range(adj.row.shape):
        if adj.row[i] == node1:
            begin = i
            break


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

def plotlabels(feature, Trure_labels, name):
 # maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']

    S_lowDWeights = visual(feature)
    colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab','hotpink']
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels)) 
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape) # [num, 3]
    for index in range(4): 
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
    #     plt.scatter(X, Y, cmap='brg', s=20, marker='.', c=colors[index], edgecolors=colors[index])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.title(name, fontsize=32, fontweight='normal', pad=20)
    
    # plt.savefig('plt_graph/exceptcomputers/{}.png'.format(name),dpi=500)
    # plt.show()
    # plt.clf()

def visual(feat):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape) # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    return x_final

def combine_dataset(*args):
    # print(feature1.shape)
    # print(feature2.shape)
    for step,adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    
    return adj

def combine_dataset_list(args):
    # print(feature1.shape)
    # print(feature2.shape)
    for step,adj in enumerate(args):
        if step == 0:
            adj1 = adj.todense()
        else:
            adj2 = adj.todense()
            zeroadj = np.zeros((adj1.shape[0], adj2.shape[0]))
            tmpadj1 = np.column_stack((adj1, zeroadj))
            tmpadj2 = np.column_stack((zeroadj.T, adj2))
            adj1 = np.row_stack((tmpadj1, tmpadj2))
            
    adj = sp.csr_matrix(adj1)
    
    return adj

def combine_dataset_list_sp(args):

    adj1 = None
    
    for step, adj in enumerate(args):
        if step == 0:
            adj1 = adj 
        else:
            num_rows1, num_cols1 = adj1.shape
            num_rows2, num_cols2 = adj.shape
            zeroadj1 = sp.csr_matrix((num_rows1, num_cols2))  
            zeroadj2 = sp.csr_matrix((num_rows2, num_cols1)) 
            
            top = sp.hstack([adj1, zeroadj1])
            bottom = sp.hstack([zeroadj2, adj])
            adj1 = sp.vstack([top, bottom])
    
    return adj1.tocsr()

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

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




