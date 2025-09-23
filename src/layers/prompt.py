import torch
import torch.nn as nn
from models.dimension import DimensionNN_FUG 

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

class balanceprompt(nn.Module):
    def __init__(self, hid_units, type_='mul'):
        super(balanceprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,3), requires_grad=True)
        self.hid_units = hid_units
        self.prompttype = type_
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    
    def get_normalized_weight(self):
        return F.softmax(self.weight, dim=1)
    
    def forward(self, graph_embedding):
        N, D = graph_embedding.shape
        part = D // 3
        x, ax, iax = graph_embedding[:, :part], graph_embedding[:, part:2*part], graph_embedding[:, 2*part:]

        w = self.get_normalized_weight()  # shape: [1, 3]
        w0, w1, w2 = w[0, 0], w[0, 1], w[0, 2]  # 각 파트 스칼라

        if self.prompttype == 'add':
            x   = x   + w0
            ax  = ax  + w1
            iax = iax + w2
        elif self.prompttype == 'mul':
            x   = x   * w0
            ax  = ax  * w1
            iax = iax * w2
        else:
            raise ValueError(f"Unknown prompt type: {self.prompttype} (use 'add' or 'mul')")

        return torch.cat([x, ax, iax], dim=1)
    
        # half = graph_embedding.shape[1]//2
        # norm_weight = self.get_normalized_weight()
        # if self.prompttype == 'add':
        #     # 각각의 파트에 weight[0, 0]과 weight[0, 1]을 더해줌
        #     x_part = graph_embedding[:, :half] + norm_weight[0, 0]
        #     ax_part = graph_embedding[:, half:] + norm_weight[0, 1]
        #     graph_embedding = torch.cat([x_part, ax_part], dim=1)
        # if self.prompttype == 'mul':
        #     x_part = graph_embedding[:, :half] * norm_weight[0, 0]
        #     ax_part = graph_embedding[:, half:] * norm_weight[0, 1]
        #     graph_embedding = torch.cat([x_part, ax_part], dim=1)
        # return graph_embedding
    def weight0(self, graph_embedding): 
        return graph_embedding * self.weight[0, 0]
    def weight1(self, graph_embedding): 
        return graph_embedding * self.weight[0, 1]
    
class weighted_prompt2(nn.Module):
    def __init__(self, mlps):
        super(weighted_prompt2, self).__init__()
        self.mlps = mlps
        self.weight= nn.Parameter(torch.FloatTensor(1, len(mlps)), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()
    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        # graph_embedding=torch.mm(self.weight, graph_embedding)
        # MLP 가중합 
        outputs = [mlp(graph_embedding) for mlp in self.mlps]
        assert all(out.shape == outputs[0].shape for out in outputs) #, "모든 MLP 출력 shape 같아야 함

        ans = torch.zeros_like(outputs[0])
        for i in range(len(self.mlps)):
            ans += self.weight[0][i] * outputs[i]
        return ans
        # assert len(graph_embedding) == self.weight.shape[1], 'length must equal'
        # ans = torch.zeros_like(graph_embedding[0])
        # for i in range(len(graph_embedding)):
        #     ans += self.weight[0][i] * graph_embedding(graph_embedding)
        # return ans
    
class composedtoken2(nn.Module):
    def __init__(self, mlps, type_='mul'):
        super(composedtoken2, self).__init__()
        # print(texttoken1.shape)
        # self.texttoken = torch.cat(texttokens,dim=0)
        # print(self.texttoken.shape)
        self.prompt = weighted_prompt2(mlps)
        self.type = type_

    def forward(self, seq):
        # print(seq.shape)
        
        texttoken = self.prompt(seq)
        
        # print(texttoken.shape)
        if self.type == 'add':
            #texttoken = texttoken.repeat(seq.shape[0],1)
            rets = seq + texttoken 
        if self.type == 'mul':
            rets = seq * texttoken
        return rets
    
class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        # 원래 방법 
        self.weight= nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

        # Attention 방식 
        dim = 3703
        k = 50 
        self.num_prompts = weightednum
        self.dim = dim
        self.k = k

        # 각 basis [dim, k] → scalar 로 projection할 weight
        self.attn_proj = nn.Parameter(torch.randn(weightednum, dim, k))  # a_j

    def reset_parameters(self):
        self.weight.data.uniform_(0, 1)

    # 어텐션 버전
    # def forward(self, prompt_stack):
    #     """
    #     graph_embedding: [num_prompts, dim] or [n, dim]
    #     self.weight: [num_prompts, dim]
    #     Returns weighted sum: [dim]
    #     """
    #     # x_i: aggregated query (e.g., mean of prompts or some input)
    #     # We'll treat query as mean of graph_embedding
    #     scores = (prompt_stack * self.attn_proj).sum(dim=(1, 2))  # [num_prompts]
    #     attn_weights = F.softmax(scores, dim=0)  # [num_prompts]

    #     # 가중합
    #     weighted = attn_weights.view(-1, 1, 1) * prompt_stack  # broadcasting → [num_prompts, dim, k]
    #     out = weighted.sum(dim=0)  # [dim, k]
    #     return out

    # 빠른 버전 
    def forward(self, graph_embedding):
        # graph_embedding: [n, d, k], self.weight: [1, n]
        assert graph_embedding.shape[0] == self.weight.shape[1], 'length must equal'
        
        # reshape weight to [n, 1, 1] for broadcasting
        weighted = self.weight.view(-1, 1, 1) * graph_embedding  # [n, d, k]
        ans = weighted.sum(dim=0)  # [d, k]
        # print('ans 150', ans.shape)
        return ans

    # 오리지널 
    # def forward(self, graph_embedding):
    #     # print("weight",self.weight)
    #     # graph_embedding=torch.mm(self.weight, graph_embedding)
    #     assert len(graph_embedding) == self.weight.shape[1], 'length must equal'
    #     ans = torch.zeros_like(graph_embedding[0])
    #     for i in range(len(graph_embedding)):
    #         ans += self.weight[0][i] * graph_embedding[i]
    #     return ans

    
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
        self.prompt = weighted_prompt( len(texttokens) )
        self.type = type_

    def forward(self, seq):
        
        texttoken = self.prompt(self.texttoken)
        
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
    
class composedW1MLP(nn.Module):
    def __init__(self, length):
        super(composedW1MLP, self).__init__()
        #self.texttoken = torch.cat(texttokens,dim=0)
        self.length = length
        self.prompt = weighted_prompt( length ).cuda()
        
    def forward(self, seq, src_mlp):
        # print(seq.shape)
        outputs = [dim(seq) for dim in src_mlp]
        stacked = torch.stack(outputs, dim=0)
        composed_dim = self.prompt(stacked) 
        return composed_dim
    

def clone_dim_pretexts(dim_pretexts):
    cloned = nn.ModuleList()
    for layer in dim_pretexts:
        # 생성자 인자 추출
        n_in = layer.n_in
        n_out = layer.n_out
        n_h = layer.n_h

        
        layers = layer.layers
        activator = layer.act.__class__


        # 새 인스턴스 생성
        new_layer = DimensionNN_FUG(n_in, n_h, n_out, activator, layers=layers)
        
        # 가중치 복사
        new_layer.load_state_dict(layer.state_dict())

        # requires_grad 유지 (원하는 경우 여기서 False로 설정 가능)
        for param in new_layer.parameters():
            param.requires_grad = False
        
        cloned.append(new_layer)
    return cloned

class composedBasisNode(nn.Module):
    def __init__(self, in_channels, p_num, basis_vectors=None):
        super(composedBasisNode, self).__init__()

        # 랜덤 초기화 - GPF-plus
        if basis_vectors == None: 
            # self.p_list = nn.Parameter(torch.empty(p_num, in_channels))  
            # nn.init.xavier_uniform_(self.p_list)   # 랜덤 초기화
            rand_init = torch.empty(p_num, in_channels)
            nn.init.xavier_uniform_(rand_init)     # Xavier 초기화
            self.register_buffer('p_list', rand_init)
        else: 
            # basis mean vector
            basis_vectors = torch.stack(basis_vectors, dim=0) # [6, 50]
            # domain token 
            # basis_vectors = torch.stack(basis_vectors, dim=1).squeeze(0) # [6, 50]
            # print(f'{basis_vectors.shape}')

            # basis 학습 X
            self.register_buffer('p_list', basis_vectors)

            # basis 학습 O
            # self.p_list = nn.Parameter(basis_vectors.clone(), requires_grad=True)

        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.p_list) # 무작위 초기화 
        self.a.reset_parameters()

    def forward(self, x):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        # return x * p
        return x + p
    
import torch.nn.functional as F
class composedFUG(nn.Module):
    def __init__(self, length, dim_pretexts, basis_matrix=None):
        super(composedFUG, self).__init__()
        #self.texttoken = torch.cat(texttokens,dim=0)
        self.length = length
        self.prompt = weighted_prompt( length ).cuda() # 가중합할 DE 개수 
        
        cloned_pretexts = clone_dim_pretexts(dim_pretexts)
        self.dim_pretexts = cloned_pretexts

        self.balance_token = basis_matrix
        

    def forward(self, sample, seq, ablation='mn', dimension_sig_open=None):
        # print(ablation)
        #XT_i w/ XT_t 도 같이 가중합 - prompt 길이 : length + 1 !! 
        # outputs = [dim.feature_sig_propagate(seq, dim(sample)) for dim in self.dim_pretexts]
        # outputs.append(dimension_sig_open)
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = self.prompt(stacked)

        # XT_i 가중합 
        # outputs = [dim.feature_sig_propagate(seq, dim(sample)) for dim in self.dim_pretexts]
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = self.prompt(stacked)

        # Ti 평균 w/ balnace token 
        # outputs = [dim((self.balance_token[i]*sample.T).T) for i, dim in enumerate(self.dim_pretexts)]
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = torch.mean(stacked, dim=0) 
        # composed_dim = F.normalize(seq @ composed_dim)

        # Ti 평균
        if ablation == 'mn': # Ti mean, No balance 
            outputs = [dim(sample) for dim in self.dim_pretexts]
            stacked = torch.stack(outputs, dim=0)
            composed_dim = torch.mean(stacked, dim=0) 
            composed_dim = F.normalize(seq @ composed_dim)

        # Ti 평균 w/ [1,2] balance token
        elif ablation == 'mb': # Ti mean with balance  
            outputs = []
            D = sample.shape[1]
            third = D // 3
            sizes = [third, third, D - 2 * third]
            x_part, ax_part, iax_part = torch.split(sample, sizes, dim=1)
            for i in range(len(self.dim_pretexts)):
                w = self.balance_token[i]                 # shape: [1,3]
                # w = F.softmax(w, dim=1)
                x_w    = x_part   * w[0, 0]
                ax_w   = ax_part  * w[0, 1]
                iax_w  = iax_part * w[0, 2]
                balanced = torch.cat([x_w, ax_w, iax_w], dim=1)
                outputs.append(self.dim_pretexts[i](balanced))
            stacked = torch.stack(outputs, dim=0)         # [Ti, B, D]
            composed_dim = torch.mean(stacked, dim=0)     # [B, D]
            composed_dim = F.normalize(seq @ composed_dim)
            # outputs = []
            # half = sample.shape[1]//2
            # for i in range(len(self.dim_pretexts)): 
            #     x_part = sample[:, :half] * self.balance_token[i][0, 0]
            #     ax_part = sample[:, half:] * self.balance_token[i][0, 1]
            #     outputs.append(self.dim_pretexts[i](torch.cat([x_part, ax_part], dim=1)))
            # stacked = torch.stack(outputs, dim=0)
            # composed_dim = torch.mean(stacked, dim=0) 
            # composed_dim = F.normalize(seq @ composed_dim)
        
        # Ti 가중합 w/ [1,2] balance token 
        elif ablation == 'wt': # Ti weighted sum, No balance 
            outputs = [dim((domain_token*sample.T).T) for dim, domain_token in zip(self.dim_pretexts, self.balance_token)]
            stacked = torch.stack(outputs, dim=0)
            composed_dim = self.prompt(stacked)
            composed_dim = F.normalize(seq @ composed_dim)
        
        elif ablation == 'ww': # Ti weighted sum, No balance 
            outputs = [dim(sample) for dim in self.dim_pretexts]
            stacked = torch.stack(outputs, dim=0)
            composed_dim = self.prompt(stacked)
            composed_dim = F.normalize(seq @ composed_dim)

        # Ti 가중합 w/ [1,2] balance token 
        elif ablation == 'wb': # Ti weighted sum with balance 
            outputs = []
            half = sample.shape[1]//2
            for i in range(len(self.dim_pretexts)): 
                x_part = sample[:, :half] * self.balance_token[i][0, 0]
                ax_part = sample[:, half:] * self.balance_token[i][0, 1]
                outputs.append(self.dim_pretexts[i](torch.cat([x_part, ax_part], dim=1)))
            stacked = torch.stack(outputs, dim=0)
            composed_dim = self.prompt(stacked)
            composed_dim = F.normalize(seq @ composed_dim)

        # Ti 가중합  w/ balance token 
        # outputs = [dim((self.balance_token[i]*sample.T).T) for i, dim in enumerate(self.dim_pretexts)]
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = self.prompt(stacked)
        # composed_dim = F.normalize(seq @ composed_dim)

        # Ti 가중합
        elif ablation == 'wn':  
            outputs = [dim(sample) for dim in self.dim_pretexts]
            stacked = torch.stack(outputs, dim=0)
            composed_dim = self.prompt(stacked)
            composed_dim = F.normalize(seq @ composed_dim)


        # Ti 가중합  w/ [1,2] balance token 
        # outputs = []
        # half = sample.shape[1]//2
        # for i in range(len(self.dim_pretexts)): 
        #     x_part = sample[:, :half] * self.balance_token[i][0, 0]
        #     ax_part = sample[:, half:] * self.balance_token[i][0, 1]
        #     outputs.append(self.dim_pretexts[i](torch.cat([x_part, ax_part], dim=1)))
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = self.prompt(stacked)
        # composed_dim = F.normalize(seq @ composed_dim)

        # Ti w/ Tt도 같이 가중합 
        # outputs = [dim(sample) for dim in self.dim_pretexts]
        # outputs.append(dimension_sig_open)
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = self.prompt(stacked)
        # composed_dim = F.normalize(seq @ composed_dim)

        # outputs = [basis.mean(dim=0) for basis in self.dim_pretexts] 

        # Ti 단순합 
        # outputs = [dim(sample) for dim in self.dim_pretexts]
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = stacked.sum(dim=0)
        # composed_dim = F.normalize(seq @ composed_dim)


        # Source Ti의 평균값들을 가중합 
        # outputs = [basis.mean(dim=0) for basis in self.balance_token] 
        # stacked = torch.stack(outputs, dim=0)
        # composed_dim = self.prompt(stacked)
        # composed_dim = F.normalize(seq @ composed_dim)

        # print(f'{composed_dim.shape}')
        # composed_dim = composed_dim.repeat(sample.shape[1], 1)

        # composed_dim = F.normalize(seq @ composed_dim)
        
        return composed_dim