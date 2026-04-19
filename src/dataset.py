import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PatchFeatureDataset(Dataset):
    def __init__(self, mode, split_path, feature_dir, mapping_path, task_type='fine', downsample_ratio=0.0):
        """
        task_type: 'fine' 或 'coarse'
        """
        self.feature_dir = feature_dir
        self.task_type = task_type
        
        # 1. 加载映射表
        # 第一列:细粒度名, 第二列:粗粒度名, 第三列:细粒度ID, 第四列:粗粒度ID
        mapping = pd.read_excel(mapping_path, header=None)
        if task_type == 'fine':
            self.label_map = dict(zip(mapping[0].astype(str), mapping[2]))
            self.num_classes = len(mapping[2].unique())
        else:
            self.label_map = dict(zip(mapping[0].astype(str), mapping[3]))
            self.num_classes = len(mapping[3].unique())

        # 2. 读取划分文件
        with open(split_path, 'r') as f:
            patient_ids = [line.strip() for line in f.readlines() if line.strip()]

        # 3. 加载特征到内存
        self.features = []
        self.labels = []
        
        print(f"正在加载 {mode} 数据到内存...")
        for p_id in patient_ids:
            h5_path = os.path.join(feature_dir, f"{p_id}.hdf5")
            if not os.path.exists(h5_path):
                continue
            with h5py.File(h5_path, 'r') as f:
                feats = f['feature'][:]
                # 转换字符串标签为对应ID
                raw_labels = f['label'][:]
                if isinstance(raw_labels[0], bytes):
                    raw_labels = [l.decode('utf-8') for l in raw_labels]
                
                target_labels = [self.label_map[l] for l in raw_labels]
                
                self.features.append(feats)
                self.labels.append(np.array(target_labels))

        self.features = np.concatenate(self.features, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        # 4. 执行降采样 (仅针对训练集)
        if mode == 'train' and downsample_ratio > 0:
            self._downsample(downsample_ratio)

        # 转为 Tensor
        self.features = torch.from_numpy(self.features).float()
        self.labels = torch.from_numpy(self.labels).long()
        print(f"{mode} 数据加载完成: {len(self.labels)} patches.")

    def _downsample(self, downsample_ratio):
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        min_count = np.min(counts)
        
        indices_to_keep = []
        for label, original_count in zip(unique_labels, counts):
            # 应用公式: target = original - round((original - min) * ratio)
            target_count = original_count - int(np.round((original_count - min_count) * downsample_ratio))
            
            label_indices = np.where(self.labels == label)[0]
            np.random.shuffle(label_indices)
            indices_to_keep.extend(label_indices[:target_count])
        
        self.features = self.features[indices_to_keep]
        self.labels = self.labels[indices_to_keep]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloader(mode, split_txt, feature_dir, mapping_path, task_type, batch_size, downsample_ratio=0.0):
    dataset = PatchFeatureDataset(mode, split_txt, feature_dir, mapping_path, task_type, downsample_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=0) # 内存数据建议0
    return loader, dataset.num_classes