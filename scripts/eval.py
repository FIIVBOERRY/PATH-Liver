import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from model import MLP
from dataset import get_dataloader
from tqdm import tqdm

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    test_loader, num_classes = get_dataloader('test', 'data/split/test.txt', 'data/features', 'data/mapping.xlsx', args.task, 1024)

    # 2. 加载模型
    model = MLP(input_dim=1536, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    all_preds = []
    all_labels = []

    # 3. 推理
    for feats, labels in tqdm(test_loader, desc="Testing"):
        feats = feats.to(device)
        with torch.no_grad():
            outputs = model(feats)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. 打印报告
    print("\nFinal Test Report:")
    print(classification_report(all_labels, all_preds))

    # 5. 绘制混淆矩阵图
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {args.task}')
    plt.savefig(f'results/cm_{args.task}.png')
    print(f"混淆矩阵已保存至 results/cm_{args.task}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    test()