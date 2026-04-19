import pandas as pd
import os
import random

def split_dataset(excel_path, output_dir):
    # 1. 加载数据
    try:
        df = pd.read_excel(excel_path, sheet_name='患者亚型统计细粒度')
    except Exception as e:
        print(f"[错误] 读取Excel失败: {e}")
        return

    # 获取亚型列名（排除'患者ID'）
    subtype_cols = [col for col in df.columns if col != '患者ID']
    patients = df.to_dict('records')
    
    # 检查稀有亚型约束 (至少3次)
    low_freq_subtypes = []
    for col in subtype_cols:
        count = df[col].sum()
        if count < 3:
            low_freq_subtypes.append(f"{col}({count}例)")
    
    if low_freq_subtypes:
        print(f"[致命警告] 以下亚型样本不足3例，无法满足三集各至少一例的要求: {', '.join(low_freq_subtypes)}")
        # 根据要求，我们尽量分配，不直接退出，但会在控制台警告

    # 2. 初始化容器
    train_list, val_list, test_list = [], [], []
    # 记录每个集合中亚型的出现次数
    counts = {
        'train': {col: 0 for col in subtype_cols},
        'val': {col: 0 for col in subtype_cols},
        'test': {col: 0 for col in subtype_cols}
    }

    # 按照亚型稀有程度对亚型排序（出现次数少的优先处理）
    sorted_subtypes = sorted(subtype_cols, key=lambda x: df[x].sum())
    
    remaining_patients = patients.copy()
    random.shuffle(remaining_patients)

    # 3. 贪心分配策略
    def assign_patient(patient, target_set):
        p_id = patient['患者ID']
        if target_set == 'train':
            train_list.append(p_id)
        elif target_set == 'val':
            val_list.append(p_id)
        else:
            test_list.append(p_id)
        
        # 更新该集合的亚型计数
        for col in subtype_cols:
            if patient[col] == 1:
                counts[target_set][col] += 1

    # 第一步：确保每个集合至少有每种亚型的一例 (优先满足测试和验证，因为它们名额少)
    for col in sorted_subtypes:
        for target in ['test', 'val', 'train']:
            if counts[target][col] == 0:
                # 寻找拥有该亚型且尚未被分配的患者
                for i, p in enumerate(remaining_patients):
                    if p[col] == 1:
                        assign_patient(p, target)
                        remaining_patients.pop(i)
                        break

    # 第二步：将剩余患者按比例分配，同时尽量维持亚型平衡
    # 目标数量 (8:1:1)
    total = len(patients)
    target_val_size = total // 10
    target_test_size = total // 10
    
    for p in remaining_patients[:]:
        # 优先填满验证集和测试集到 10% 比例
        if len(val_list) < target_val_size:
            assign_patient(p, 'val')
        elif len(test_list) < target_test_size:
            assign_patient(p, 'test')
        else:
            assign_patient(p, 'train')
        remaining_patients.remove(p)

    # 4. 输出结果与控制台统计
    os.makedirs(output_dir, exist_ok=True)
    
    for name, data in [('train', train_list), ('val', val_list), ('test', test_list)]:
        with open(os.path.join(output_dir, f"{name}.txt"), 'w', encoding='utf-8') as f:
            for pid in data:
                f.write(f"{pid}\n")

    print("\n" + "="*40)
    print(f"划分完成！总样本: {len(patients)}")
    print(f"训练集: {len(train_list)} | 验证集: {len(val_list)} | 测试集: {len(test_list)}")
    print("="*40)
    
    # 统计展示
    stats_data = []
    for col in subtype_cols:
        stats_data.append({
            '亚型': col,
            '总计': df[col].sum(),
            '训练集': counts['train'][col],
            '验证集': counts['val'][col],
            '测试集': counts['test'][col]
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # 检查是否有集合缺失亚型
    for target in ['train', 'val', 'test']:
        missing = [s for s in subtype_cols if counts[target][s] == 0]
        if missing:
            print(f"\n[注意] {target}集 缺失以下亚型: {missing}")

if __name__ == "__main__":
    INPUT_EXCEL = "data/pathology_statistics.xlsx"
    OUTPUT_FOLDER = "data/split"
    
    split_dataset(INPUT_EXCEL, OUTPUT_FOLDER)