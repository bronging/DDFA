import os
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Coauthor, FacebookPagePage, Flickr, Twitch

from dataset import *
from process import *

def save_sample(data, path):
    os.makedirs(path, exist_ok=True)
    torch.save(data['idx'], os.path.join(path, 'idx.pt'))
    torch.save(data['labels'], os.path.join(path, 'labels.pt'))
    if 'batch' in data:
        torch.save(data['batch'], os.path.join(path, 'batch.pt'))

def generate_fewshot_samples(labels, num_shots, num_samples, exclude_last=1000):
    fewshot_samples = []
    unique_labels = torch.unique(labels)

    N = labels.shape[0]
    candidate_indices = torch.arange(0, N - exclude_last)  # exclude the last 'exclude_last' nodes from candidates
    candidate_labels = labels[candidate_indices]

    for _ in range(num_samples):
        samples = []
        for label in unique_labels:
            label_indices = (candidate_labels == label).nonzero(as_tuple=True)[0]
            if len(label_indices) < num_shots:
                continue 
            selected_indices = random.sample(label_indices.tolist(), num_shots)
            samples.extend(selected_indices)
        fewshot_samples.append({
            'idx': torch.tensor(samples),
            'labels': labels[samples]
        })
    return fewshot_samples

def create_folders(base_path, dataset_name, num_shots=10, num_samples=100):
    for shot in range(1, num_shots + 1):
        for i in range(1, num_samples + 1):
            os.makedirs(os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i)), exist_ok=True)

def save_fewshot_data(dataset_name, num_shots=10, num_samples=100, path = './data', exclude_last=1000):
    print(f'Generating node_data for {dataset_name}')
    dataset = load_dataset(dataset_name, path)
    data = dataset[0]
    labels = data.y

    base_path = os.path.join(path, f'fewshot_{dataset_name.lower()}')
    create_folders(base_path, dataset_name, num_shots, num_samples)

    for shot in range(1, num_shots + 1):
        samples = generate_fewshot_samples(labels, shot, num_samples, exclude_last=exclude_last)
        for i, sample in enumerate(samples):
            sample_path = os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i))
            save_sample(sample, sample_path)

def generate_fewshot_samples_graph(dataset_name, shotnum, num_samples, path='./data'):
    data = load_dataset(dataset_name, path)[0]
    features, adj = process_tu(data,data['x'].shape[1])
    samples = []
    for i in range(num_samples):
        idx_train = torch.load(f"{path}/fewshot_{dataset_name.lower()}/{shotnum}-shot_{dataset_name.lower()}/{i}/idx.pt").type(torch.long)
        lbl_train = torch.load(f"{path}/fewshot_{dataset_name.lower()}/{shotnum}-shot_{dataset_name.lower()}/{i}/labels.pt").type(torch.long)
        subgraph = build_subgraph(adj, idx_train)
        samples.append(
            {
            'idx' : subgraph['idx'],
            'batch' : subgraph['batch'],
            'labels' : lbl_train,
            }
        )
    return samples


def save_fewshot_graph_data(dataset_name, num_shots=10, num_samples=100):
    print(f'Generating graph_data for {dataset_name}')
    path = './data'
    base_path = os.path.join(path, f'fewshot_{dataset_name.lower()}_graph')
    create_folders(base_path, dataset_name, num_shots, num_samples)

    for shot in range(1, num_shots + 1):
        samples = generate_fewshot_samples_graph(dataset_name, shot, num_samples, path)
        for i, sample in enumerate(samples):
            sample_path = os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i))
            #batch = torch.arange(len(sample['idx']))
            #sample['batch'] = batch
            save_sample(sample, sample_path)

if __name__ == '__main__':
    datasets =   ['Cora', 'Citeseer', 'Pubmed', 'Photo', 'Computers', 'FacebookPagePage', 'LastFMAsia',]
    # datasets = ['Texas', 'Cornell', 'Wisconsin', 'chameleon', 'squirrel']
    # datasets = ['Reddit']
    # datasets = ['Citeseer']
    for dataset_name in datasets:
        save_fewshot_data(dataset_name, exclude_last=1000)
        save_fewshot_graph_data(dataset_name)
