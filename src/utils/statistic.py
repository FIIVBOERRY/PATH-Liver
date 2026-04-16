import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def count_samples_by_category(processed_dir, mapping_path):
    """
    统计每个类别下的样本数量

    参数:
        processed_dir: data/processed/ 目录路径
        mapping_path: docs/mapping_fine.xlsx 文件路径

    返回:
        category_counts: 字典，类别名 -> 样本数量
    """
    # 读取映射文件
    df_map = pd.read_excel(mapping_path, header=None)
    folder_to_category = dict(zip(df_map[0].astype(str), df_map[1].astype(str)))

    # 统计每个类别的样本数
    category_counts = defaultdict(int)

    # 遍历所有子文件夹
    for folder_name in os.listdir(processed_dir):
        folder_path = os.path.join(processed_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 获取类别名
        category = folder_to_category.get(folder_name, "未知类别")

        # 统计该文件夹中的文件数（假设都是图片文件）
        file_count = 0
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                file_count += 1

        category_counts[category] += file_count
        print(f"{folder_name} -> {category}: {file_count} 个样本")

    return category_counts

def plot_category_distribution(category_counts, save_path="category_distribution.png"):
    """
    绘制类别分布柱状图

    参数:
        category_counts: 类别计数字典
        save_path: 保存路径
    """
    # 转换为DataFrame便于绘图
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    # 按样本数量排序
    sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
    categories_sorted, counts_sorted = zip(*sorted_data)

    # 创建柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(categories_sorted)), counts_sorted, color='skyblue', alpha=0.8)

    # 添加数值标签
    for i, (category, count) in enumerate(zip(categories_sorted, counts_sorted)):
        plt.text(i, count + max(counts_sorted) * 0.01, f'{count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 设置标签和标题
    plt.xlabel('类别', fontsize=12, fontweight='bold')
    plt.ylabel('样本数量', fontsize=12, fontweight='bold')
    plt.title('各类别样本数量分布', fontsize=14, fontweight='bold')

    # 设置x轴刻度
    plt.xticks(range(len(categories_sorted)), categories_sorted, rotation=45, ha='right')

    # 添加网格
    plt.grid(axis='y', alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"柱状图已保存至: {save_path}")

    # 显示图像
    plt.show()

def main():
    processed_dir = "data/processed"
    mapping_path = "docs/mapping_fine.xlsx"
    output_path = "category_distribution.png"

    print("开始统计类别样本数量...")
    category_counts = count_samples_by_category(processed_dir, mapping_path)

    print(f"\n统计结果:")
    total_samples = sum(category_counts.values())
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"{category}: {count} 个样本 ({percentage:.1f}%)")

    print(f"\n总样本数: {total_samples}")
    print(f"类别数量: {len(category_counts)}")

    print("\n生成柱状图...")
    plot_category_distribution(category_counts, output_path)

if __name__ == "__main__":
    main()