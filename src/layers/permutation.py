import torch.nn as nn
import torch 


def build_hard_permutation_from_logits(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: (d, d) - raw permutation logits (MLP output)
    return: (d, d) - hard permutation matrix
    """
    d = scores.shape[0]
    perm = torch.zeros_like(scores)

    # Step 1: 각 행의 분산 계산 (MLP output 기준)
    row_variances = scores.var(dim=1)  # shape: (d,)

    # Step 2: 분산 큰 행부터 인덱스 정렬
    row_order = torch.argsort(row_variances, descending=True).tolist()

    # Step 3: 각 행의 열 점수 내림차순 정렬 (우선순위 열)
    ranked_cols = torch.argsort(scores, dim=1, descending=True)  # shape: (d, d)

    assigned_cols = set()

    for row in row_order: # 분산이 가장 큰 행부터 
        for col in ranked_cols[row]: # 가장 큰 값 가지는 열 순서대로 방문 
            col = col.item() 
            if col not in assigned_cols: # 아직 할당 안 됐을 경우, 
                perm[row, col] = 1       # 헤당 열을 현재 행의 최댓값 행으로 할당. 
                assigned_cols.add(col)   # 만약 이미 할당 되었으면, 다음으로 큰 열을 할당
                break

    return perm

class NodeFeaturePermMLP(nn.Module): 
    def __init__(self, d, hidden_dim=128, n_layers=2, mlp_init=True):
        super().__init__()
        self.d = d

        layers = []
        in_dim = d  # 입력은 평균 낸 (d,) 벡터

        for i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # 마지막은 (hidden_dim → d*d)
        layers.append(nn.Linear(in_dim, d * d))
        self.net = nn.Sequential(*layers)

        if mlp_init:
            self._init_as_identity()

    def forward(self, node_feats):
        """
        node_feats: (N, d)
        return: permutation logits: (d, d)
        """
        graph_feat = node_feats.mean(dim=0)  # shape: (d,) # 노드 피처 평균내서 입력 
        logits = self.net(graph_feat)        # shape: (d*d,)
        # perm_matrix = build_hard_permutation_from_logits(logits.view(self.feature_dim, self.feature_dim))
        return logits.view(self.d, self.d)  # shape: (d, d)

    def _init_as_identity(self):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                in_dim, out_dim = layer.weight.shape
                if in_dim == out_dim:
                    torch.nn.init.eye_(layer.weight)
                else:
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                torch.nn.init.constant_(layer.bias, 0.0)

class PermutationLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.permute_weight = nn.Parameter(torch.randn(1, d))  # 초기화: 작은 값 or eye + noise

    def forward(self, x):  # x: [n, d]
        # P_soft = torch.softmax(self.permute_logits, dim=-1)  # soft row-normalized
        return x * self.permute_weight  # [n, d] @ [d, d] → [n, d]


class DimConv1D(nn.Module):
    def __init__(self, d, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):  # x: [n, d]
        x = x.unsqueeze(1)        # → [n, 1, d]
        x = self.conv(x)          # → [n, 1, d]
        return x.squeeze(1)       # → [n, d]