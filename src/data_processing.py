import os
import xml.etree.ElementTree as ET
import openslide
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

def is_point_in_path(x, y, poly):
    """射线法判断点是否在多边形内部"""
    num = len(poly)
    j = num - 1
    res = False
    for i in range(num):
        if ((poly[i][1] > y) != (poly[j][1] > y)) and \
                (x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
            res = not res
        j = i
    return res

def is_patch_inside(x, y, size, poly):
    """判断 Patch 的四个角是否都在多边形内"""
    corners = [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]
    for px, py in corners:
        if not is_point_in_path(px, py, poly):
            return False
    return True

def slice_with_detailed_stats(svs_path, xml_path, output_dir, folder_name, patch_size=224, stride=224):
    """处理单个SVS文件
    
    Args:
        svs_path: SVS文件路径
        xml_path: 对应的XML文件路径
        output_dir: 输出目录（所有patch的父目录）
        folder_name: 源文件夹名称（用于patch文件名前缀）
        patch_size: patch尺寸（像素）
        stride: 相邻patch中心点之间的距离（像素）
    
    Returns:
        dict: 该文件的统计信息
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        slide = openslide.OpenSlide(svs_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ 文件初始化失败: {folder_name} - {e}")
        return None

    # 统计模块初始化
    category_stats = defaultdict(int) # {类别名: 总数}
    total_patches = 0
    
    # 预提取
    all_regions = []
    for ann in root.findall("Annotation"):
        # 尝试获取层名称（在Attributes里）
        layer_name = "Unknown"
        attr = ann.find(".//Attribute")
        if attr is not None:
            layer_name = attr.get('Value')
        
        regions_node = ann.find("Regions")
        if regions_node is not None:
            for r in regions_node.findall("Region"):
                all_regions.append((r, layer_name))

    print(f"  🚀 开始处理: {folder_name} ({len(all_regions)} 个连通区域)")

    # 主循环
    for idx, (region, layer_name) in enumerate(tqdm(all_regions, desc=f"    {folder_name}", unit="region", leave=False)):
        label = (region.get("Text") or layer_name).strip().replace("/", "_")
        region_id = region.get("Id", str(idx)) # 获取Region ID
        
        vertices = [(float(v.get("X")), float(v.get("Y"))) for v in region.findall(".//Vertex")]
        if len(vertices) < 3:
            continue

        min_x, max_x = int(min(v[0] for v in vertices)), int(max(v[0] for v in vertices))
        min_y, max_y = int(min(v[1] for v in vertices)), int(max(v[1] for v in vertices))
        
        save_path = os.path.join(output_dir, label)
        os.makedirs(save_path, exist_ok=True)

        region_patch_count = 0  # 当前连通区域计数
        
        steps_x = range(min_x, max_x - patch_size, stride)
        steps_y = range(min_y, max_y - patch_size, stride)
        
        with tqdm(total=len(steps_x) * len(steps_y), desc=f"      区域 ID:{region_id} [{label}]", leave=False) as pbar:
            for y in steps_y:
                for x in steps_x:
                    if is_patch_inside(x, y, patch_size, vertices):
                        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                        # 文件名包含源文件夹名称以防坐标重复
                        file_name = f"{folder_name}_{x}_{y}.jpg"
                        patch.save(os.path.join(save_path, file_name))
                        
                        region_patch_count += 1
                    pbar.update(1)
        
        # 累加统计
        category_stats[label] += region_patch_count
        total_patches += region_patch_count

    slide.close()

    return dict(category_stats), total_patches

def batch_process_folders(raw_data_dir, output_dir="data/processed/", patch_size=224, stride=224):
    """批量处理data/raw目录下的所有子文件夹
    
    Args:
        raw_data_dir: 原始数据目录路径（包含多个子文件夹）
        output_dir: 输出目录路径
        patch_size: patch尺寸（像素）
        stride: 相邻patch中心点之间的距离（像素），默认与patch_size相同（无重叠）
    """
    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    
    if not raw_data_dir.exists():
        print(f"❌ 错误：输入目录不存在 - {raw_data_dir}")
        return
    
    # 扫描所有子文件夹
    subfolders = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    
    if not subfolders:
        print(f"⚠️  警告：在 {raw_data_dir} 中未找到任何子文件夹")
        return
    
    print("=" * 70)
    print(f"📋 批量处理开始")
    print(f"  输入目录: {raw_data_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  Patch大小: {patch_size}x{patch_size} 像素")
    print(f"  步长 (Stride): {stride} 像素")
    print(f"  共发现 {len(subfolders)} 个子文件夹")
    print("=" * 70)
    print()
    
    # 全局统计
    global_stats = defaultdict(int)  # {标签: 总patch数}
    total_patches_all = 0
    skipped_folders = []
    processed_folders = []
    
    # 逐个处理文件夹
    for idx, folder in enumerate(subfolders, 1):
        folder_name = folder.name
        svs_file = folder / f"{folder_name}.svs"
        xml_file = folder / f"{folder_name}.xml"
        
        print(f"[{idx}/{len(subfolders)}] 处理文件夹: {folder_name}")
        
        # 检查文件是否存在
        if not svs_file.exists():
            print(f"  ⚠️  跳过: 缺少SVS文件 ({svs_file.name})")
            skipped_folders.append((folder_name, "缺少SVS文件"))
            continue
        
        if not xml_file.exists():
            print(f"  ⚠️  跳过: 缺少XML文件 ({xml_file.name})")
            skipped_folders.append((folder_name, "缺少XML文件"))
            continue
        
        # 处理文件
        result = slice_with_detailed_stats(str(svs_file), str(xml_file), str(output_dir), folder_name, patch_size, stride)
        
        if result is not None:
            category_stats, total_patches = result
            
            # 合并到全局统计
            for label, count in category_stats.items():
                global_stats[label] += count
            total_patches_all += total_patches
            
            print(f"  ✅ 完成: 生成 {total_patches} 个 Patch\n")
            processed_folders.append(folder_name)
        else:
            skipped_folders.append((folder_name, "处理失败"))
            print()
    
    # --- 最终汇总报告 ---
    print("\n" + "=" * 70)
    print("📊 批量处理汇总报告")
    print("=" * 70)
    print(f"{'类别名称 (Category)':<30} | {'Patch 总数':<15}")
    print("-" * 70)
    
    if global_stats:
        for label in sorted(global_stats.keys()):
            count = global_stats[label]
            print(f"{label:<30} | {count:<15}")
        print("-" * 70)
    
    print(f"{'总计 (Total)':<30} | {total_patches_all:<15}")
    print("=" * 70)
    
    print(f"\n✅ 处理成功: {len(processed_folders)} 个文件夹")
    if skipped_folders:
        print(f"⚠️  跳过: {len(skipped_folders)} 个文件夹")
        for folder_name, reason in skipped_folders:
            print(f"     - {folder_name}: {reason}")
    
    print("=" * 70)


# --- 运行 ---
if __name__ == "__main__":
    # 示例1: 默认配置（patch_size=224, stride=224，无重叠）
    # batch_process_folders("data/raw/", "data/processed/")
    
    # 示例2: 自定义stride（patch_size=224, stride=112，有50%重叠）
    batch_process_folders("data/raw/", "data/processed_1/", patch_size=224, stride=224)