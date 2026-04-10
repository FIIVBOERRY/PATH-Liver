import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np

class FeatureDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        feat = torch.load(self.file_paths[idx]).squeeze(0) # [1536]
        label = self.labels[idx]
        return feat, label

def get_dataloaders(feature_dir, batch_size=64, seed=42):
    categories = sorted([d for d in os.listdir(feature_dir)])
    class_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    all_paths = []
    all_labels = []
    for cat in categories:
        cat_dir = os.path.join(feature_dir, cat)
        paths = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith('.pt')]
        all_paths.extend(paths)
        all_labels.extend([class_to_idx[cat]] * len(paths))

    # 8:1:1 划分
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=seed, stratify=all_labels)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels)

    # 处理训练集不平衡 (WeightedRandomSampler)
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in train_labels])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_ds = FeatureDataset(train_paths, train_labels)
    val_ds = FeatureDataset(val_paths, val_labels)
    test_ds = FeatureDataset(test_paths, test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_idx