import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.model import MLPClassifier as MLP
from src.dataset import get_dataloader
from tqdm import tqdm
import os

def test(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    test_loader, num_classes = get_dataloader('test', '/root/autodl-tmp/data/split/test.txt', '/root/autodl-tmp/data/features', '/root/autodl-tmp/data/mapping.xlsx', args.task, 1024)

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
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {args.task}')
    plt.savefig(f'results/cm_{args.task}_80.png')
    print(f"混淆矩阵已保存至 results/cm_{args.task}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fine', choices=['fine', 'coarse'])
    #parser.add_argument('--model_path', type=str, default='checkpoints/best_coarse_model.pth')
    parser.add_argument('--model_path', type=str, default='checkpoints/fine/epoch_80_fine_model.pth')
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    test(args)