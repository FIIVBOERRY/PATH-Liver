# 提取svs前景，用于可视化

import os
import numpy as np
import cv2
import openslide
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

def create_aperio_xml(contours, scaling_factor, output_path):
    """生成符合 Aperio ImageScope 规范的 XML 文件"""
    annotations = Element('Annotations', MicronsPerPixel="0.252000") # MPP可根据实际修改
    annotation = SubElement(annotations, 'Annotation', Id="1", Name="", ReadOnly="0", 
                            NameReadOnly="0", LineColor="65280", Visible="1", 
                            Type="4", LineWidth="2")
    regions = SubElement(annotation, 'Regions')
    
    # 只要最大的一个连通域
    if len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)
        
        region = SubElement(regions, 'Region', Id="1", Type="0", Zoom="1.0", Selected="0",
                            ImageFocus="-1", NegativeROA="0", InputRegionId="0", Analyze="1")
        vertices = SubElement(region, 'Vertices')
        
        for point in main_contour:
            x, y = point[0]
            # 缩放回 Level 0 坐标轴
            SubElement(vertices, 'Vertex', X=str(float(x * scaling_factor)), Y=str(float(y * scaling_factor)))
            
    tree = ElementTree(annotations)
    indent(tree, space="\t", level=0)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def extract_foreground_xml(svs_path, output_xml_path=None, level=2):
    """提取前景并保存 XML"""
    if not os.path.exists(svs_path):
        raise FileNotFoundError(f"SVS 文件未找到: {svs_path}")
    slide = openslide.OpenSlide(svs_path)
    
    # 1. 获取低倍率缩放因子
    scaling_factor = slide.level_downsamples[level]
    dims = slide.level_dimensions[level]
    
    # 2. 读取低倍率图像
    img_rgba = slide.read_region((0, 0), level, dims)
    img_rgb = np.array(img_rgba.convert('RGB'))
    
    # 3. 图像处理提取前景
    # 转到 HSV 空间，通常背景饱和度极低
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)
    
    # 使用 Otsu 自动阈值处理饱和度通道
    _, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 形态学闭运算：填充组织内部的小孔洞，连接断开的部分
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 5. 提取最外层轮廓 (RETR_EXTERNAL 自动忽略孔洞)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. 生成 XML
    if output_xml_path is None:
        output_xml_path = os.path.splitext(svs_path)[0] + ".xml"
    else:
        output_dir = os.path.dirname(output_xml_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    create_aperio_xml(contours, scaling_factor, output_xml_path)
    
    print(f"任务完成: {os.path.basename(svs_path)}")
    print(f"坐标缩放因子: {scaling_factor:.2f}, 已保存至: {output_xml_path}")

if __name__ == "__main__":
    extract_foreground_xml(
        "data/raw/C0033/C0033.svs",
        output_xml_path="data/vis/C0033/C0033.xml",
        level=2
    )