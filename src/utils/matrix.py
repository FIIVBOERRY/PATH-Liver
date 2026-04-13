import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def calculate_metrics(y_true, y_pred, y_prob):
    """
    y_true: List or np.array
    y_pred: List or np.array (hard labels)
    y_prob: List or np.array (softmax probabilities)
    """
    acc = accuracy_score(y_true, y_pred)
    # macro 平均适合类别不平衡情况
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    # AUC 计算
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except:
        auc = 0.0 # 在某些极端划分情况下可能无法计算
        
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 对混淆矩阵进行归一化（按行归一化，即显示 Recall）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    # 使用蓝色色调 (Blues)，标注原始数值和百分比
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Actual Label (True)')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()