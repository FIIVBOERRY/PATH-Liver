import os
import platform
import torch
import torch.nn as nn
import numpy as np
import openslide
import cv2
import timm
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from src.model import MLPClassifier # 确保路径正确

# --- 设置matplotlib支持中文 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 配置区 ---
SVS_PATH = "data/raw/C0070/C0070.svs"
XML_PATH = "data/vis/C0070/C0070.xml"
UNI2_CKPT = "assets/ckpts/uni2-h/pytorch_model.bin"
MODEL_CKPT = "checkpoints/best_model.pth"
PATCH_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 类别定义与高级美观颜色 (RGB)
CLASS_INFO = {
    0: {"name": "低黏附型", "color": (231, 76, 60)},   # 珊瑚红
    1: {"name": "实性型",   "color": (52, 152, 219)},  # 晴空蓝
    2: {"name": "正常组织", "color": (46, 204, 113)},  # 翡翠绿
    3: {"name": "海绵状型", "color": (155, 89, 182)},  # 优雅紫
    4: {"name": "腺管型",   "color": (241, 196, 15)},  # 向日葵黄
    5: {"name": "鳞状细胞型", "color": (230, 126, 34)}  # 亮橙色
}

# --- 核心函数 ---

def parse_xml_to_mask(xml_path, level0_dims, downsample):
    """解析XML并将前景轮廓转为低倍率下的Mask"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    points = []
    for vertex in root.findall(".//Vertex"):
        x = float(vertex.get('X')) / downsample
        y = float(vertex.get('Y')) / downsample
        points.append([x, y])
    
    mask_size = (int(level0_dims[1] / downsample), int(level0_dims[0] / downsample))
    mask = np.zeros(mask_size, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

def load_models():
    # 1. 加载 UNI2
    print("Loading UNI2-h...")
    uni2 = timm.create_model(
        "vit_giant_patch14_dinov2", # 基于架构名
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8,
        dynamic_img_size=True, pretrained=False
    )
    uni2.load_state_dict(torch.load(UNI2_CKPT, map_location='cpu'))
    uni2.to(DEVICE).eval()

    # 2. 加载 MLP 分类器
    print("Loading MLP Classifier...")
    classifier = MLPClassifier(input_dim=1536, num_classes=6)
    classifier.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
    classifier.to(DEVICE).eval()
    
    return uni2, classifier

def run_visualization():
    slide = openslide.OpenSlide(SVS_PATH)
    w0, h0 = slide.dimensions
    
    # 获取缩略图作为背景 (Level 2 或更高，约 1/16 尺寸)
    vis_level = 2
    downsample = slide.level_downsamples[vis_level]
    thumb = slide.get_thumbnail(slide.level_dimensions[vis_level])
    canvas = np.array(thumb.convert("RGB"))
    overlay = canvas.copy()
    
    # 解析 Mask
    mask = parse_xml_to_mask(XML_PATH, (w0, h0), downsample)
    
    uni2, classifier = load_models()
    
    # 遍历全图 Patch (Level 0)
    print("Starting streaming inference...")
    total_patches = ((h0 + PATCH_SIZE - 1) // PATCH_SIZE) * ((w0 + PATCH_SIZE - 1) // PATCH_SIZE)
    with tqdm(total=total_patches, desc="Patches", unit="patch") as pbar:
        for y in range(0, h0, PATCH_SIZE):
            for x in range(0, w0, PATCH_SIZE):
                # 检查该 Patch 是否在前景内 (检查中心点坐标)
                mx, my = int((x + PATCH_SIZE//2) / downsample), int((y + PATCH_SIZE//2) / downsample)
                if my >= mask.shape[0] or mx >= mask.shape[1] or mask[my, mx] == 0:
                    pbar.update(1)
                    continue
                
                # 提取并推理
                region = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                img_t = torch.from_numpy(np.array(region)).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
                
                with torch.no_grad():
                    feat = uni2(img_t)
                    output = classifier(feat)
                    pred = torch.argmax(output, dim=1).item()
                
                # 绘制到 Overlay (计算在缩略图上的对应位置)
                vx, vy = int(x / downsample), int(y / downsample)
                vw, vh = int(PATCH_SIZE / downsample), int(PATCH_SIZE / downsample)
                color = CLASS_INFO[pred]["color"]
                cv2.rectangle(overlay, (vx, vy), (vx + vw, vy + vh), color, -1)
                pbar.update(1)

    # 合并图像 (Alpha 透明度叠加)
    alpha = 1.0
    res = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    
    # 添加图例 (Legend)
    res_pil = Image.fromarray(res)
    draw = ImageDraw.Draw(res_pil)
    # 加载中文字体
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 14)
    except:
        font = ImageFont.load_default()
    # 简单的文本列表绘制
    for i, info in CLASS_INFO.items():
        draw.rectangle([10, 10 + i*30, 30, 30 + i*30], fill=info["color"])
        draw.text((40, 10 + i*30), info["name"], fill=(255,255,255), font=font)
    
    # 保存结果
    save_path = "data/vis/C0070/C0070_visualization.png"
    res_pil.save(save_path)
    print(f"Visualization saved to: {save_path}")

def run_svs2png():
    slide = openslide.OpenSlide(SVS_PATH)
    w0, h0 = slide.dimensions
    vis_level = 2
    thumb = slide.get_thumbnail(slide.level_dimensions[vis_level])
    save_path = "data/vis/C0070/C0070_thumbnail.png"
    thumb.save(save_path)
    print(f"Thumbnail saved to: {save_path}")

def run_colorbar():
    # 生成颜色条图像
    bar_height = 40
    bar_width = 400
    gap = 15
    total_height = (bar_height + gap) * len(CLASS_INFO) + gap
    colorbar = Image.new("RGB", (bar_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(colorbar)
    # 加载中文字体
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    for i, info in CLASS_INFO.items():
        y_start = gap + i * (bar_height + gap)
        # 绘制颜色方块
        draw.rectangle([10, y_start, 40, y_start + bar_height], fill=info["color"], outline=(0, 0, 0), width=1)
        # 绘制文本
        draw.text((50, y_start + 5), info["name"], fill=(0, 0, 0), font=font)
    
    save_path = "data/vis/colorbar.png"
    colorbar.save(save_path)
    print(f"Colorbar saved to: {save_path}")


if __name__ == "__main__":
    run_visualization()
    #run_svs2png()
    #run_colorbar()