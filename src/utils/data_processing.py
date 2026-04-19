import os
import xml.etree.ElementTree as ET
import pandas as pd
import sys

def process_pathology_data(root_path, output_path, mapping_path):
    # --- 1. 加载并验证映射关系 ---
    try:
        # 无表头，0列为细粒度，1列为粗粒度
        map_df = pd.read_excel(mapping_path, sheet_name='Sheet1', header=None)
        # 建立 细粒度 -> 粗粒度 的映射字典
        fine_to_coarse = dict(zip(map_df[0].astype(str), map_df[1].astype(str)))
        # 获取所有唯一的粗粒度标签
        all_coarse_subtypes = sorted(list(set(fine_to_coarse.values())))
    except Exception as e:
        print(f"[错误] 无法读取映射文件 {mapping_path}: {e}")
        sys.exit(1)

    all_fine_subtypes = set()
    patient_data = [] # 存储结构: {'id': str, 'fine': set, 'coarse': set}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    
    # --- 2. 遍历患者数据 ---
    for folder in subfolders:
        xml_file = os.path.join(root_path, folder, f"{folder}.xml")
        
        if not os.path.exists(xml_file):
            print(f"[警告] 患者 {folder} 缺少 XML 文件，已跳过。")
            continue
            
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            curr_fine = set()
            curr_coarse = set()

            for region in root.iter('Region'):
                subtype = region.get('Text')
                if subtype:
                    subtype = subtype.strip()
                    # 关键逻辑：检查是否在映射表中
                    if subtype not in fine_to_coarse:
                        print(f"\n[致命错误] 患者 {folder} 的 XML 中发现未定义标签: '{subtype}'")
                        print(f"该标签不在 {mapping_path} 的第一列中。脚本停止运行。")
                        sys.exit(1)
                    
                    curr_fine.add(subtype)
                    all_fine_subtypes.add(subtype)
                    # 映射到粗粒度
                    curr_coarse.add(fine_to_coarse[subtype])
            
            patient_data.append({
                'PatientID': folder,
                'FineSubtypes': curr_fine,
                'CoarseSubtypes': curr_coarse
            })
            
        except ET.ParseError:
            print(f"[错误] 患者 {folder} 的 XML 文件损坏。")
        except SystemExit:
            sys.exit(1)
        except Exception as e:
            print(f"[异常] 处理患者 {folder} 时发生未知错误: {e}")

    # --- 3. 准备数据报表 ---
    sorted_fine = sorted(list(all_fine_subtypes))

    # Sheet 1: 亚型汇总 (细粒度)
    df_summary = pd.DataFrame(sorted_fine, columns=['病理亚型名称(细粒度)'])

    # Sheet 2: 细粒度矩阵
    fine_matrix = []
    for item in patient_data:
        row = {'患者ID': item['PatientID']}
        for s in sorted_fine:
            row[s] = 1 if s in item['FineSubtypes'] else 0
        fine_matrix.append(row)
    df_fine = pd.DataFrame(fine_matrix)

    # Sheet 3: 粗粒度矩阵
    coarse_matrix = []
    for item in patient_data:
        row = {'患者ID': item['PatientID']}
        for s in all_coarse_subtypes:
            row[s] = 1 if s in item['CoarseSubtypes'] else 0
        coarse_matrix.append(row)
    df_coarse = pd.DataFrame(coarse_matrix)

    # --- 4. 保存 Excel ---
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='亚型汇总', index=False)
            df_fine.to_excel(writer, sheet_name='患者亚型统计细粒度', index=False)
            df_coarse.to_excel(writer, sheet_name='患者亚型统计粗粒度', index=False)
        print(f"\n[成功] 统计完成！")
        print(f"总计细粒度亚型: {len(sorted_fine)} 种")
        print(f"总计粗粒度亚型: {len(all_coarse_subtypes)} 种")
        print(f"结果已保存至: {output_path}")
    except Exception as e:
        print(f"[错误] 写入 Excel 失败: {e}")

if __name__ == "__main__":
    # 配置路径
    RAW_DATA_PATH = "data/raw"
    MAPPING_FILE = "data/mapping.xlsx"
    OUTPUT_FILE = "data/pathology_statistics.xlsx"
    
    process_pathology_data(RAW_DATA_PATH, OUTPUT_FILE, MAPPING_FILE)