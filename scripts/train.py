import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np

# 导入自定义模块
from src.model import MLPClassifier as MLP
from src.dataset import get_dataloader

def train():
    # 初始化 WandB
    wandb.init(project="pathology_patch_classification", name=f"{args.task}_{args.name}", config=args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, num_classes = get_dataloader('train', '/root/autodl-tmp/data/split/train.txt', '/root/autodl-tmp/data/features', '/root/autodl-tmp/data/mapping.xlsx', args.task, args.batch_size, args.downsample)
    val_loader, _ = get_dataloader('val', '/root/autodl-tmp/data/split/val.txt', '/root/autodl-tmp/data/features', '/root/autodl-tmp/data/mapping.xlsx', args.task, args.batch_size)

    # 初始化模型 (假设输入维度1536)
    model = MLP(input_dim=1536, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for feats, labels in pbar:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        #每10个epoch保存一次模型，包括最后一个epoch
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/fine/epoch_{epoch+1}_{args.task}_model.pth")
        elif (epoch + 1) == args.epochs:
            torch.save(model.state_dict(), f"checkpoints/fine/epoch_{epoch+1}_{args.task}_model.pth")

        # 验证
        val_metrics = evaluate(model, val_loader, device, num_classes)
        print(f"Epoch {epoch+1} Val Acc: {val_metrics['acc']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        wandb.log({"train_loss": train_loss/len(train_loader), **val_metrics})

        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save(model.state_dict(), f"checkpoints/fine/best_{args.task}_model.pth")

def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    # AUC 处理多分类
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc = 0.0

    # 混淆矩阵传给 WandB
    cm = confusion_matrix(all_labels, all_preds)
    wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=all_labels, preds=all_preds)})

    return {'acc': acc, 'precision': p, 'recall': r, 'f1': f1, 'auc': auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fine', choices=['fine', 'coarse'])
    parser.add_argument('--name', type=str, default='jida_classification_fine')
    parser.add_argument('--downsample', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    train()