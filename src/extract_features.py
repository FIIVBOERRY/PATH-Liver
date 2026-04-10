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
    
    # 自动处理可能嵌套的键名
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    # 载入模型
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"Successfully loaded weights: {msg}")

    model.to(DEVICE)
    model.eval()
    return model

class SimpleImageDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path

@torch.no_grad()
def extract():
    model = get_uni2_model()
    # UNI2 推荐的归一化 (根据其训练习惯)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    categories = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    
    for cat in categories:
        cat_in_path = os.path.join(IMG_DIR, cat)
        cat_out_path = os.path.join(SAVE_DIR, cat)
        os.makedirs(cat_out_path, exist_ok=True)
        
        img_paths = glob(os.path.join(cat_in_path, "*.png")) + glob(os.path.join(cat_in_path, "*.jpg"))
        print(f"Processing {cat}: {len(img_paths)} images")

        # --- 优化点 1: 使用 DataLoader 开启多线程读取 ---
        dataset = SimpleImageDataset(img_paths, preprocess)
        # num_workers 设置为你的 CPU 核心数的一半左右（例如 4 或 8）
        loader = DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)
        print(f"Processing {cat} with Batch Size {64}...")

        for imgs, paths in tqdm(loader):
            imgs = imgs.to(DEVICE)
            
            # --- 优化点 2: 批量特征提取 ---
            feats = model(imgs) # 一次处理 64 张
            
            # 保存结果
            for i in range(len(paths)):
                save_p = os.path.join(cat_out_path, os.path.basename(paths[i]).split('.')[0] + ".pt")
                torch.save(feats[i].cpu(), save_p)

if __name__ == "__main__":
    extract()