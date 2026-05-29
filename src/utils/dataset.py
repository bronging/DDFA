import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import (
    Actor, Amazon, Coauthor, FacebookPagePage, Flickr,
    KarateClub, LastFMAsia, Planetoid, PPI, Reddit,
    Twitch, WebKB, WikipediaNetwork, Yelp,  #BitcoinOTC 
)
from torch_geometric.utils import degree, to_networkx

# import matplotlib.pyplot as plt
# from ogb.nodeproppred import PygNodePropPredDataset
# from ogb.lsc import MAG240MDataset

WebKB_datasets = ['Texas', 'Cornell', 'Wisconsin']
Planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
Amazon_datasets = ['Photo', 'Computers']
Coauthor_datasets = ['CS', 'Physics']
WikipediaNetwork_datasets = ['chameleon', 'squirrel']
Reddit_datasets = ['Reddit']
OGB_datasets = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-mag']

Flickr_datasets = ['Flickr']
PPI_datasets = ['PPI']
Yelp_datasets = ['Yelp']
Twitch_datasets = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
Actor_datasets = ['Actor']
KarateClub_datasets = ['KarateClub']
FacebookPagePage_datasets = ['FacebookPagePage']
LastFMAsia_datasets = ['LastFMAsia']
#BitcoinOTC_datasets = ['BitcoinOTC']
#MAG240MDatasets = ['MAG240MDataset']

def get_anchor_dataset(cache_dir, anchor_name, sparse): 
    domain0_feat = torch.load(f'{cache_dir}/{anchor_name}_aug_feature.pt')[0].cuda()
    domain0_adj = torch.load(f'{cache_dir}/{anchor_name}_aug_adj.pt')[0]
    if sparse: 
        domain0_adj = domain0_adj.cuda()
    else: 
        domain0_adj = torch.FloatTensor(domain0_adj.todense())
    return domain0_feat, domain0_adj

def load_dataset(name, path='./data'):
    if name in Planetoid_datasets:
        dataset = Planetoid(root=path, name=name)
    elif name in Amazon_datasets:
        dataset = Amazon(root=path, name=name)
    elif name in Coauthor_datasets:
        dataset = Coauthor(root=path, name=name)
    elif name in WebKB_datasets:
        dataset = WebKB(root=path, name=name)
    elif name in WikipediaNetwork_datasets:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in Reddit_datasets:
        dataset = Reddit(root=f'{path}/Reddit')
    # elif name in OGB_datasets:
    #     dataset = PygNodePropPredDataset(root=path, name=name)
    elif name in Flickr_datasets:
        dataset = Flickr(root=f'{path}/Flickr')
    elif name in PPI_datasets:
        dataset = PPI(root=f'{path}/PPI')
    elif name in Yelp_datasets:
        dataset = Yelp(f'{path}/Yelp')
    elif name in Twitch_datasets:
        dataset = Twitch(root=path,name=name)
    elif name in Actor_datasets:
        dataset = Actor(root=f'{path}/Actor')
    elif name in KarateClub_datasets:
        dataset = KarateClub()
    elif name in FacebookPagePage_datasets:
        dataset = FacebookPagePage(root=f'{path}/Facebook')
    elif name in LastFMAsia_datasets:
        dataset = LastFMAsia(root=f'{path}/LastFMAsia')
    #elif name in BitcoinOTC_datasets:
    #    dataset = BitcoinOTC(root=f'{path}/BitcoinOTC')
    #elif name in MAG240MDatasets:
    #    dataset = MAG240MDataset(root=f'{path}/MAG240MDataset')
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    #print(f'{name}: {dataset[0].num_nodes}')
    return dataset

def analyze_dataset(name, graph = False):
    data_ = load_dataset(name)
    #print(len(data_))
    data = data_[0]
    deg = degree(data.edge_index[0], data.num_nodes)
    average_degree = deg.mean().item()
    print(f'\n{name}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}')
    G = to_networkx(data, to_undirected=True)

    print('Clustering Coefficient:')
    global_clustering_coefficient = nx.transitivity(G) 
    print(f'Global Clustering Coefficient: {global_clustering_coefficient:.4f}')

    average_clustering_coefficient = nx.average_clustering(G)
    print(f'Average Clustering Coefficient: {average_clustering_coefficient:.4f}')

    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    path_lengths = []
    for node, lengths in shortest_path_lengths.items():
        path_lengths.extend(lengths.values())
    
    network_diameter = max(path_lengths)
    print(f'Network Diameter:')

    average_shortest_path_length = np.mean(path_lengths)
    print(f'Average Shortest Path Length: {average_shortest_path_length:.4f}')

    percentile_90_shortest_path_length = np.percentile(path_lengths, 90)
    print(f'90th Percentile Shortest Path Length: {percentile_90_shortest_path_length:.4f}')


def analyze_dataset_multi(name, graph = False):
    data_ = load_dataset(name)
    #print(len(data_))
    for i, data in enumerate(data_):
        deg = degree(data.edge_index[0], data.num_nodes)
        average_degree = deg.mean().item()
        print(f'{name}_{i+1}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}')

if __name__ == '__main__':
    selected_datasets = [#'Texas', 'Cornell', 'Wisconsin', 
                        #'Cornell',
                        'Cora', 'Citeseer', 'Pubmed', 
                        'Photo',
                        'Computers', 
                        #'CS', 'Physics', 
                        #'chameleon', 'squirrel', 
                        #'Reddit',
                        #'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', #'ogbn-mag',
                        #'Flickr', 
                        #'PPI', 
                        #'Yelp', 
                        #'Actor',
                        #'ES',
                        #'DE', 'EN', 'ES', 
                        #'FR', 'PT', 'RU',
                        #'KarateClub',
                        'FacebookPagePage', 'LastFMAsia', 
                        # #'BitcoinOTC'
                        #'MAG240MDataset'
                        ]
    for dataset in selected_datasets:
        analyze_dataset(dataset)
        #analyze_dataset_multi(dataset)