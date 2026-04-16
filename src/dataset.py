import os
import argparse
import torch
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

def downsample_patches(patch_info_list, seed=42, downsample_ratio=None):
    """
    按比例降采样，使所有类别保持相同的样本数量

    参数:
        patch_info_list: [(file_path, offset, label), ...] 列表
        seed: 随机种子
        downsample_ratio: 降采样比例 (0.0-1.0)，None表示不降采样

    返回:
        降采样后的 patch_info_list
    """
    if not patch_info_list or downsample_ratio is None or downsample_ratio >= 1.0:
        return patch_info_list

    # 统计每个label的样本数
    label_groups = {}
    for file_path, offset, label in patch_info_list:
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append((file_path, offset, label))

    # 找最少的样本数
    min_count = min(len(samples) for samples in label_groups.values())

    print(f"\n=== 样本降采样信息 (比例: {downsample_ratio:.2f}) ===")
    print(f"最少样本数: {min_count}")

    np.random.seed(seed)
    downsampled = []

    # 对每个类别进行降采样
    for label, samples in label_groups.items():
        original_count = len(samples)

        if original_count == min_count:
            # 最少的类不降采样
            downsampled.extend(samples)
            print(f"类别 {label}: {original_count} -> {original_count} (无降采样)")
        else:
            # 目标数量 = original_count - round((original_count - min_count) * downsample_ratio)
            target_count = original_count - int(np.round((original_count - min_count) * downsample_ratio))

            # 随机选择目标数量的样本
            indices = np.random.choice(len(samples), size=target_count, replace=False)
            selected_samples = [samples[i] for i in indices]
            downsampled.extend(selected_samples)
            print(f"类别 {label}: {original_count} -> {target_count} (比例: {downsample_ratio:.2f})")

    print(f"总样本数: {len(patch_info_list)} -> {len(downsampled)}\n")

    return downsampled

class HDF5FeatureDataset(Dataset):
    """
    针对大尺寸 HDF5 特征文件优化的 Dataset
    支持延迟加载和索引映射
    """
    def __init__(self, file_info_list, return_coords=False):
        """
        file_info_list: 包含 (file_path, internal_offset, label) 的列表
        return_coords: 是否在 __getitem__ 中返回 coords 字符串
        """
        self.file_info_list = file_info_list
        self.return_coords = return_coords
        # 缓存打开的文件对象，避免频繁打开关闭导致的 IO 瓶颈
        self.files = {}

    def __len__(self):
        return len(self.file_info_list)

    def __getitem__(self, idx):
        file_path, offset, label = self.file_info_list[idx]
        
        # 延迟打开文件
        if file_path not in self.files:
            self.files[file_path] = h5py.File(file_path, 'r')
        file = self.files[file_path]
        
        # 仅读取特定偏移量的特征 [1536]
        feat = torch.from_numpy(file['features'][offset]).float()
        if self.return_coords:
            coord = file['coords'][offset]
            if isinstance(coord, (bytes, bytearray)):
                coord = coord.decode('utf-8', errors='ignore')
            return feat, label, coord
        return feat, label

def get_dataloaders(feature_dir, mapping_path, batch_size=64, seed=42, include_coords=False, downsample_ratio=0.7, num_workers=4, pin_memory=True):
    # 1. 加载映射表 (不含表头)
    # 第一列：细亚型 (与文件名一致), 第二列：粗亚型
    df_map = pd.read_excel(mapping_path, header=None)
    fine_to_coarse = dict(zip(df_map[0].astype(str), df_map[1].astype(str)))
    
    # 构建粗分类的 class_to_idx
    coarse_categories = sorted(list(set(fine_to_coarse.values())))
    class_to_idx = {cat: i for i, cat in enumerate(coarse_categories)}
    
    # 2. 扫描文件并构建全局 Patch 索引
    all_patch_info = [] # 存储 (文件路径, 内部索引, 粗分类标签)
    
    # 获取目录下所有 hdf5
    h5_files = [f for f in os.listdir(feature_dir) if f.endswith('.hdf5')]
    
    print("正在扫描 HDF5 文件结构以建立全局索引...")
    for f_name in h5_files:
        print(f"Processing {f_name}...")
        fine_type = f_name.replace('.hdf5', '')
        if fine_type not in fine_to_coarse:
            continue
            
        coarse_type = fine_to_coarse[fine_type]
        label = class_to_idx[coarse_type]
        file_path = os.path.join(feature_dir, f_name)
        
        # 快速读取 shape 而不加载数据
        with h5py.File(file_path, 'r') as f:
            num_patches = f['features'].shape[0]
            
        # 记录该文件中每个 patch 的信息
        for i in range(num_patches):
            all_patch_info.append((file_path, i, label))

    # 进行比例降采样 (在划分前)
    all_patch_info = downsample_patches(all_patch_info, seed=seed, downsample_ratio=downsample_ratio)

    # 3. 划分数据集 (按 Patch 级别进行 8:1:1 划分)
    # 提取标签用于分层采样
    labels_only = [info[2] for info in all_patch_info]
    
    train_info, temp_info, train_labels, temp_labels = train_test_split(
        all_patch_info, labels_only, test_size=0.2, random_state=seed, stratify=labels_only)
    
    val_info, test_info, val_labels, test_labels = train_test_split(
        temp_info, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels)

    # 4. Patch 级别的 WeightedRandomSampler
    # 统计训练集中每个“粗分类”的总 Patch 数
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    class_weights = 1. / counts # 频率高的类权重低
    
    # 为训练集的每个样本分配权重
    sample_weights = np.array([class_weights[np.where(unique_labels == l)[0][0]] for l in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 5. 构建 DataLoader
    train_ds = HDF5FeatureDataset(train_info)
    val_ds = HDF5FeatureDataset(val_info, return_coords=include_coords)
    test_ds = HDF5FeatureDataset(test_info, return_coords=include_coords)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    # 注意：使用 HDF5 时，多进程读取 (num_workers > 0) 需要处理文件句柄冲突，
    # 对于 Windows 环境，通常需要每个 worker 单独打开 HDF5 文件，
    # 当前实现会在每个 worker 首次访问时打开文件，理论上可启用 num_workers>0。
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, class_to_idx


def print_class_mapping(feature_dir, mapping_path, downsample_ratio=None):
    _, _, _, class_to_idx = get_dataloaders(feature_dir, mapping_path, downsample_ratio=downsample_ratio)
    sorted_classes = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    print("\n类别索引映射 (类别名 -> 模型输出索引):")
    for cls_name, idx in class_to_idx.items():
        print(f"{cls_name} -> {idx}")
    print("\n按索引顺序的类别列表:")
    for idx, cls_name in enumerate(sorted_classes):
        print(f"{idx}: {cls_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="打印 class_to_idx 映射")
    parser.add_argument("--feature_dir", default="data/features", help="特征目录")
    parser.add_argument("--mapping_path", default="docs/mapping_coarse.xlsx", help="映射文件路径")
    parser.add_argument("--downsample_ratio", type=float, default=None, help="降采样比例 (0.0-1.0)，None表示不降采样")
    args = parser.parse_args()
    print_class_mapping(args.feature_dir, args.mapping_path, downsample_ratio=0.7)