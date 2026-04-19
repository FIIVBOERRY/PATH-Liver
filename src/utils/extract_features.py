import os
import h5py
import torch
import timm
import numpy as np
import openslide
import xml.etree.ElementTree as ET
from tqdm import tqdm
from shapely.geometry import Polygon, Point, box
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# --- 配置参数 ---
MODEL_PATH = "assets/ckpts/uni2-h/pytorch_model.bin"
RAW_DATA_PATH = "data/raw"
OUTPUT_DIR = "data/features"
STRIDE = 224  # 默认步长
BATCH_SIZE = 128 # 根据 5060 显存调整，建议 32-128 之间

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

# --- 数据预处理 ---
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class PatchDataset(Dataset):
    def __init__(self, svs_path, coords):
        self.svs_path = svs_path
        self.coords = coords
        self.slide = None  # 初始化为空

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # 延迟初始化：每个子进程第一次运行时打开自己的句柄
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.svs_path)
            
        x, y = self.coords[idx]
        # 提取 224x224 patch
        img = self.slide.read_region((x, y), 0, (224, 224)).convert('RGB')
        return preprocess(img), torch.tensor([x, y])

def get_region_polygons(xml_path):
    """解析XML获取多边形及其标签"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions = []
    for region in root.iter('Region'):
        label = region.get('Text', 'Unknown').strip()
        vertices = []
        for vertex in region.iter('Vertex'):
            vertices.append((float(vertex.get('X')), float(vertex.get('Y'))))
        if len(vertices) >= 3:
            regions.append({'polygon': Polygon(vertices), 'label': label})
    return regions

def load_model():
    print(f"正在加载 UNI2 模型: {MODEL_PATH}")
    model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
    # 使用 map_location 确保在正确的设备上加载
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to('cuda').eval()
    return model

def process_single_patient(patient_id, model, device):
    xml_path = os.path.join(RAW_DATA_PATH, patient_id, f"{patient_id}.xml")
    svs_path = os.path.join(RAW_DATA_PATH, patient_id, f"{patient_id}.svs")
    h5_path = os.path.join(OUTPUT_DIR, f"{patient_id}.hdf5")
    # 断点续传检查
    if os.path.exists(h5_path):
        return f"[跳过] {patient_id} 已存在。"

    if not os.path.exists(xml_path) or not os.path.exists(svs_path):
        return f"[警告] {patient_id} 缺少必要文件。"  

    try:
        # 在主进程先解析 XML 确定坐标
        regions = get_region_polygons(xml_path)
        
        all_features = []
        all_labels = []
        all_coords = []

        for reg in regions:
            poly = reg['polygon']
            label = reg['label']
            minx, miny, maxx, maxy = poly.bounds
            
            x_coords = np.arange(int(minx), int(maxx) - 224 + 1, STRIDE)
            y_coords = np.arange(int(miny), int(maxy) - 224 + 1, STRIDE)
            
            valid_coords = []
            for x in x_coords:
                for y in y_coords:
                    patch_box = box(x, y, x + 224, y + 224)
                    if poly.contains(patch_box):
                        valid_coords.append((x, y))

            if not valid_coords:
                continue

            # 批量提取特征，传递 svs_path 而不是 slide 对象
            dataset = PatchDataset(svs_path, valid_coords)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            
            pbar = tqdm(total=len(dataset), desc=f"-> {patient_id} [{label[:12]}]", leave=False)
            
            for imgs, coords_batch in loader:
                imgs = imgs.to(device)
                with torch.no_grad():
                    features = model(imgs)
                
                all_features.append(features.cpu().numpy())
                all_coords.append(coords_batch.numpy())
                all_labels.extend([label] * len(features))
                pbar.update(len(imgs))
            pbar.close()

        if all_features:
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('feature', data=np.concatenate(all_features, axis=0))
                f.create_dataset('coords', data=np.concatenate(all_coords, axis=0))
                # HDF5 存储字符串需转换编码
                f.create_dataset('label', data=np.array(all_labels, dtype=h5py.string_dtype(encoding='utf-8')))
            return f"[完成] {patient_id} 处理成功。"
        else:
            return f"[提示] {patient_id} 未切出有效 Patch。"

    except Exception as e:
        return f"[错误] {patient_id} 发生异常: {e}"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model()

    patients = [d for d in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, d))]
    
    print(f"开始特征提取，共 {len(patients)} 名患者...")
    for p_id in tqdm(patients, desc="总进度"):
        result = process_single_patient(p_id, model, device)
        if result:
            print(result)

if __name__ == "__main__":
    main()