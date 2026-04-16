import os
import shutil
import re

def main():
    # --- 配置区域 ---
    # 存放原始文本的文件名（请确保该文件与脚本在同一目录，或者使用绝对路径）
    input_text_file = 'list.txt' 
    source_base = 'data/processed'
    target_base = 'data/chosen'
    
    # 用于记录缺失的文件
    missing_files = []
    copy_count = 0
    skip_count = 0

    # 检查输入文件是否存在
    if not os.path.exists(input_text_file):
        print(f"错误: 找不到输入文件 {input_text_file}")
        return

    # 正则表达式说明：
    # coord=(.*?).jpg  -> 匹配文件名
    # true_label=(.*)  -> 匹配子文件夹名（去除末尾空白）
    pattern = re.compile(r"coord=(?P<filename>.*?\.jpg),\s+true_label=(?P<label>\S+)")

    print("开始处理文件...")

    with open(input_text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 尝试匹配包含文件信息的行
            match = pattern.search(line)
            
            if match:
                filename = match.group('filename')
                label = match.group('label')

                # 构建路径
                src_dir = os.path.join(source_base, label)
                dst_dir = os.path.join(target_base, label)
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)

                # 1. 如果目标子文件夹不存在，则创建
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                # 2. 检查源文件是否存在
                if os.path.exists(src_path):
                    # 3. 如果目标文件已存在，则跳过
                    if os.path.exists(dst_path):
                        skip_count += 1
                    else:
                        shutil.copy2(src_path, dst_path)
                        copy_count += 1
                else:
                    missing_files.append(src_path)

    # --- 报告结果 ---
    print("-" * 30)
    print(f"处理完成！")
    print(f"成功复制: {copy_count} 个文件")
    print(f"跳过已存在: {skip_count} 个文件")
    
    if missing_files:
        print(f"警告: 有 {len(missing_files)} 个文件未找到。")
        with open('missing_report.log', 'w', encoding='utf-8') as log:
            for m in missing_files:
                log.write(f"{m}\n")
        print("缺失文件列表已记录至: missing_report.log")
    else:
        print("未发现缺失文件。")

if __name__ == "__main__":
    main()