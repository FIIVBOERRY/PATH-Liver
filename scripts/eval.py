import torch
from src.dataset import get_dataloaders
from src.model import MLPClassifier
from src.utils.matrix import calculate_metrics, plot_confusion_matrix

def test():
    _, _, test_loader, class_to_idx = get_dataloaders("data/features")
    model = MLPClassifier(num_classes=len(class_to_idx)).cuda()
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.cuda()
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    print("--- Final Test Results ---")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    # 获取类别名称（确保顺序与训练时一致）
    _, _, _, class_to_idx = get_dataloaders("data/features")
    # 将字典转为按索引排序的列表: ['Glandular_Microacinar', 'NormalFibrous', 'NormalLiver']
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    
    # 绘图
    plot_confusion_matrix(all_labels, all_preds, class_names)

if __name__ == "__main__":
    test()