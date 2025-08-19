import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch.nn.functional as F
from preprompt import sliced_wasserstein_torch
import torch 
import numpy as np


def plot_heatmap(matrix: torch.Tensor, title: str, filename: str, save_dir: str = "./heatmaps"):
    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
    filepath = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Dimension')
    plt.ylabel('New Dimension(50)')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)  # 고해상도 저장
    plt.close()  # 리소스 해제

    print(f"✅ Saved heatmap: {filepath}")


def plot_tsne(out, title, filename, save_dir: str = "./tsne"):
    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
    filepath = os.path.join(save_dir, filename)
    # out: (num_features, reduced_dim)  예: (700, 50)
    # 예: out = model.out.detach().cpu().numpy()
    out_np = out.cpu().numpy()  # torch.Tensor → numpy

    # t-SNE 임베딩
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    out_tsne = tsne.fit_transform(out_np)

    # 시각화
    plt.figure(figsize=(8, 6))
    plt.scatter(out_tsne[:, 0], out_tsne[:, 1], s=10, alpha=0.6)
    plt.title("t-SNE Visualization of Feature Basis Vectors ($t_i$)")
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)  # 고해상도 저장
    print(f"✅ Saved heatmap: {filepath}")

def tsne_test_train_w_class(n_test, n_train, all_embs, all_labels, pretrain_dataset_names, backbone): 


    # 4. 도메인 마스크 만들기 (test는 0, train은 1)
    is_train = torch.cat([
        torch.zeros(n_test),
        torch.ones(n_train)
    ], dim=0)

    tsne = TSNE(n_components=2, perplexity=min(30, (all_embs.shape[0]-1)//2))
    tsne_result = tsne.fit_transform(all_embs.cpu().numpy())

    tsne_x = tsne_result[:, 0]
    tsne_y = tsne_result[:, 1]

    plt.figure(figsize=(10, 8))

    num_classes = all_labels.max().item() + 1
    colors = plt.cm.tab20.colors  # 20가지 색상

    for cls in range(num_classes):
        cls_mask = (all_labels == cls)
        
        # test와 train 각각 시각화
        for train_flag, label in zip([0, 1], ['Test', 'Train']):
            mask = cls_mask & (is_train == train_flag)
            size = 20 if train_flag == 0 else 80  # test는 작게, train은 크게
            
            plt.scatter(
                tsne_x[mask], tsne_y[mask],
                c=[colors[cls % len(colors)]],
                label=f'Class {cls} ({label})',
                s=size, alpha=0.7, edgecolors='k' if train_flag == 1 else 'none'
            )

    plt.title("t-SNE: Train vs Test Embeddings")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{backbone}_tsne_domain_class_{pretrain_dataset_names}.png", dpi=300) 


def tsne_per_domain_w_class(embedding, class_labels, pretrain_dataset_names, backbone): 

    # (2) 임베딩 합치기 및 라벨 생성
    all_embeddings = torch.cat(embedding, dim=0).cpu().numpy()
    domain_labels = np.concatenate([[i] * len(embedding[i]) for i in range(len(embedding))])
    class_labels = torch.cat(class_labels, dim=0).cpu().numpy()  # shape: [total_N]

    # (3) t-SNE 수행
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_result = tsne.fit_transform(all_embeddings)

    # (4) 시각화 및 저장 

    num_domains = len(embedding)
    num_classes = 18

    markers = [
        'o',  # 원
        's',  # 정사각형
        '^',  # 위쪽 삼각형
        'v',  # 아래쪽 삼각형
        '<',  # 왼쪽 삼각형
        '>',  # 오른쪽 삼각형
        'D',  # 다이아몬드
        'd',  # 작은 다이아몬드
        'p',  # 오각형
        'P',  # 플러스 다각형
        '*',  # 별
        'h',  # 육각형1
        'H',  # 육각형2
        'X',  # X자
        'x',  # 소문자 x
        '+',  # +
        '|',  # 수직선
        '_',  # 수평선
        '.',  # 점
        ',',  # 픽셀 점
    ]
    # colors = plt.cm.get_cmap('tab10', num_classes)  # class 기준 색상
    colors_18 = [
        '#e41a1c',  # 빨강 (Red)
        '#377eb8',  # 파랑 (Blue)
        '#4daf4a',  # 초록 (Green)
        '#984ea3',  # 보라 (Purple)
        '#ff7f00',  # 주황 (Orange)
        '#ffff33',  # 노랑 (Yellow)
        '#a65628',  # 갈색 (Brown)
        '#f781bf',  # 핑크 (Pink)
        '#999999',  # 회색 (Gray)
        '#66c2a5',  # 청록 (Teal)
        '#fc8d62',  # 살구 (Apricot)
        '#8da0cb',  # 옅은 파랑
        '#e78ac3',  # 연핑크
        '#a6d854',  # 연두
        '#ffd92f',  # 밝은 노랑
        '#e5c494',  # 밝은 갈색
        '#b3b3b3',  # 중간 회색
        '#1b9e77',  # 짙은 청록
    ]

    for domain_id in range(num_domains):
        plt.figure(figsize=(10, 6))

        for class_id in range(num_classes):
            idx = (class_labels == class_id) & (domain_labels == domain_id)
            if np.sum(idx) == 0:
                continue
            plt.scatter(
                tsne_result[idx, 0],
                tsne_result[idx, 1],
                color=colors_18[class_id],
                marker=markers[class_id],
                s=3,
                alpha=0.7,
                label=f'Class {class_id}, Domain {domain_id}'
            )

        plt.title(f"{pretrain_dataset_names[domain_id]} t-SNE of Node Embeddings)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{backbone}_tsne_domain_class_{pretrain_dataset_names[domain_id]}.png", dpi=300)  

def plot_source_emb_tnse(embedding, backbone): 

    all_embeddings = torch.cat(embedding, dim=0).cpu().numpy()
    domain_labels = np.concatenate([[i] * len(embedding[i]) for i in range(len(embedding))])

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_result = tsne.fit_transform(all_embeddings)

    # (4) 시각화 및 저장
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'black']
    for i in range(len(embedding)):
        idx = domain_labels == i
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=f'Graph {i}', color=colors[i], alpha=0.6, s=10)

    plt.title(f"{backbone}_t-SNE of Node Embeddings from 6 Graphs (Domain Token Applied)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{backbone}_tsne_graph_domains.png", dpi=300)

        
    plt.figure(figsize=(8, 5))
    for i, z in enumerate(embedding):
        norms = torch.norm(z, dim=1).cpu().numpy()
        plt.hist(norms, bins=50, alpha=0.4, color=colors[i], label=f'Graph {i}', density=True)

    plt.xlabel("L2 Norm of Node Embedding")
    plt.ylabel("Density")
    plt.title(f"{backbone}_Embedding Norm Distribution per Graph")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{backbone}_norm_graph_domains.png", dpi=300)

def compare_embeddings(emb_org, emb_perm):

    w1dist = sliced_wasserstein_torch(emb_org, emb_perm)


    if emb_org.shape[0] == emb_perm.shape[0]: 
        sampled_org = emb_org
        sampled_perm = emb_perm 
    else: 
        sample_size = min(emb_org.shape[0], emb_perm.shape[0])

        # 공통 샘플 인덱스 선택 (무작위)
        idx1 = torch.randperm(emb_org.shape[0])[:sample_size]
        idx2 = torch.randperm(emb_perm.shape[0])[:sample_size]
        
        sampled_org = emb_org[idx1]
        sampled_perm = emb_perm[idx2]

    # (1) Cosine similarity (유사도, 1에 가까울수록 비슷)
    cos_sim = F.cosine_similarity(sampled_org, sampled_perm, dim=1)  # [num_nodes]
    cos_sim_mean = cos_sim.mean().item()
    cos_sim_std = cos_sim.std().item()

    # (2) Euclidean distance (거리, 0에 가까울수록 비슷)
    euc_dist = torch.norm(sampled_org - sampled_perm, p=2, dim=1)    # [num_nodes]
    euc_dist_mean = euc_dist.mean().item()
    euc_dist_std = euc_dist.std().item()

    print(f'Cosine Similarity (mean): {cos_sim_mean:.4f} ± {cos_sim_std:.4f}')
    print(f'Euclidean Distance (mean): {euc_dist_mean:.4f} ± {euc_dist_std:.4f}')

    return {
        'cosine_similarity': cos_sim,
        'cosine_mean': cos_sim_mean,
        'cosine_std': cos_sim_std,
        'euclidean_distance': euc_dist,
        'euclidean_mean': euc_dist_mean,
        'euclidean_std': euc_dist_std,
        'wasserstain': w1dist
    }

def plot_similarity_heatmap(embeddings, labels, downstream):
    """
    embeddings: list of 6 tensors, each shape [num_nodes, dim]
    labels: list of 6 strings for graph names
    """
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))

    # pairwise cosine similarity 평균값 계산
    for i in range(n):
        for j in range(n):
            result = compare_embeddings(embeddings[i], embeddings[j])
            # sim_matrix[i][j] = result['cosine_mean']
            sim_matrix[i][j] = result['wasserstain']


    # 히트맵 시각화 (matplotlib 사용)
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(sim_matrix, cmap='coolwarm', vmin=0, vmax=1)

    # 라벨 설정
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 각 셀에 값 표시
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color="black")

    # 색상 바 추가
    # fig.colorbar(cax, ax=ax, label="Mean Cosine Similarity")
    fig.colorbar(cax, ax=ax, label="wasserstain Distance")

    # ax.set_title("Pairwise Embedding Cosine Similarity Heatmap")
    ax.set_title("Pairwise Feature Wasserstain Distance Heatmap")
    plt.tight_layout()
    # plt.savefig(f'T_{downstream}_S_similarity.png', dpi=300)
    plt.savefig(f'T_{downstream}_S_wasserstain.png', dpi=300)