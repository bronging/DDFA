import torch 

from torch.utils.data import Dataset

class PretrainDatasetAug(Dataset):
    def __init__(self, aug_features=None, aug_adjs=None, lbls=None):
        self.lbls = lbls
        self.aug_features = aug_features
        self.aug_adjs = aug_adjs

    def __len__(self):
        return len(self.aug_features)

    def __getitem__(self, idx):
        return {
            'feature': self.aug_features[idx],
            'adj': self.aug_adjs[idx],
            'lbls': self.lbls[idx]
        }

