import os
import shutil
import xml.etree.ElementTree as ET
import pandas as pd

def batch_update_xml_tags(data_root, mapping_excel):
    # 1. 加载对应表 (无表头，第一列原标签，第二列目标标签)
    try:
        mapping_df = pd.read_excel(mapping_excel, sheet_name='亚型汇总', header=None)
        # 转换为字典方便查找: { 'OldName': 'NewName' }
        tag_map = dict(zip(mapping_df[0].astype(str), mapping_df[1].astype(str)))
    except Exception as e:
        print(f"[致命错误] 无法读取对应表: {e}")
        return

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    
    print("开始执行批量修改任务...\n" + "-"*50)

    for folder in subfolders:
        xml_path = os.path.join(data_root, folder, f"{folder}.xml")
        
        if not os.path.exists(xml_path):
            continue
            
        try:
            # 2. 创建备份 (.xml.bak)
            bak_path = xml_path + ".bak"
            shutil.copy2(xml_path, bak_path)
            
            # 3. 解析并修改 XML
            # ImageScope 的 XML 包含编码信息，通常需要保持原有格式
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            modified = False
            for region in root.iter('Region'):
                old_text = region.get('Text')
                
                if old_text:
                    old_text_stripped = old_text.strip()
                    if old_text_stripped in tag_map:
                        # 执行修改
                        region.set('Text', tag_map[old_text_stripped])
                        modified = True
                    else:
                        # 4. 如果对应表中不包含该字段，在控制台输出
                        print(f"[未匹配] 文件: {folder}.xml | 字段: '{old_text}'")
            
            # 5. 保存修改（覆盖原文件）
            if modified:
                # 使用 original encoding 尽量保持一致
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)
                print(f"[已完成] {folder}.xml 修改成功并已备份。")
            else:
                print(f"[跳过] {folder}.xml 无需修改内容。")
                
        except Exception as e:
            print(f"[错误] 处理患者 {folder} 时发生异常: {e}")

    print("-"*50 + "\n所有任务执行完毕。")

if __name__ == "__main__":
    # 路径配置
    DATA_RAW_PATH = "data/raw"
    MAPPING_FILE = "data/pathology_statistics.xlsx"
    
    batch_update_xml_tags(DATA_RAW_PATH, MAPPING_FILE)