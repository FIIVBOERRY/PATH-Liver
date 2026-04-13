import os
import torch
import timm
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset

# ================= 配置 =================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
IMG_DIR = "data/processed"
SAVE_DIR = "data/features"
MODEL_PATH = "assets/ckpts/uni2-h/pytorch_model.bin"
BATCH_SIZE = 128  # 适配 8GB 显存
NUM_WORKERS = 8   # 根据 CPU 核心数调整
# ==========================================

class PatchDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        filename = os.path.basename(path)
        if self.transform:
            img = self.transform(img)
        return img, filename

def get_uni2_model():
    print(f"Initializing UNI2-h model with custom architecture...")
    
    # 使用官方推荐参数
    timm_kwargs = {
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }

    # 1. 先构建模型骨架
    # 注意：这里使用 vit_giant_patch14_dinov2 只是为了获取基础类，参数会被 kwargs 覆盖
    model = timm.create_model(
        "vit_giant_patch14_dinov2", 
        **timm_kwargs
    )

    # 2. 加载本地权重
    # pytorch_model.bin 是完整的 HF 格式，通常包含 state_dict
    print(f"Loading local weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

def process_subfolders():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    model = get_uni2_model()
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 获取所有子文件夹（类别）
    subfolders = [f for f in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, f))]
    
    for subfolder in subfolders:
        save_path = os.path.join(SAVE_DIR, f"{subfolder}.hdf5")
        
        # 断点续传逻辑
        if os.path.exists(save_path):
            print(f"Skipping {subfolder}: HDF5 already exists.")
            continue

        print(f"\nProcessing class: {subfolder}")
        subfolder_path = os.path.join(IMG_DIR, subfolder)
        img_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not img_files:
            continue

        dataset = PatchDataset(img_files, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        all_features = []
        all_coords = []

        with torch.no_grad():
            for imgs, filenames in tqdm(dataloader, desc=f"Extracting {subfolder}", unit="batch"):
                imgs = imgs.to(DEVICE)
                features = model(imgs) # [B, 1536]
                
                all_features.append(features.cpu().numpy())
                all_coords.extend(filenames)

        # 合并特征
        all_features = np.concatenate(all_features, axis=0)
        
        # 写入 HDF5
        with h5py.File(save_path, "w") as f:
            f.create_dataset("features", data=all_features, compression="gzip")
            # 处理不定长字符串存储
            dt = h5py.special_dtype(vlen=str)
            dset_coords = f.create_dataset("coords", (len(all_coords),), dtype=dt)
            dset_coords[:] = all_coords
            
        print(f"Saved: {save_path} (Total: {len(all_coords)})")

if __name__ == "__main__":
    process_subfolders()
    print("\nAll tasks completed.")