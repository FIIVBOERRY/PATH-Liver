import torch
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.model import MLPClassifier
from src.utils.matrix import calculate_metrics, plot_confusion_matrix

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_top10_predictions(all_preds, all_probs, all_coords, all_labels, class_names, top_k=10):
    """打印每个预测类别下概率最高的 top_k patch 来源与坐标"""
    class_groups = {i: [] for i in range(len(class_names))}
    for pred, prob, coord, true_label in zip(all_preds, all_probs, all_coords, all_labels):
        score = float(prob[pred])
        class_groups[pred].append((score, coord, true_label))

    print("\n--- 每个预测类别的 Top 10 Patch 来源与坐标 ---")
    for class_idx, class_name in enumerate(class_names):
        print(f"\n[{class_idx}] {class_name}:")
        top_patches = sorted(class_groups[class_idx], key=lambda x: x[0], reverse=True)[:top_k]
        if not top_patches:
            print("  (无预测样本)")
            continue
        for rank, (score, coord, true_label) in enumerate(top_patches, start=1):
            true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
            print(f"  {rank:2d}. prob={score:.4f}, coord={coord}, true_label={true_name}")


def test(print_top10=False):
    _, _, test_loader, class_to_idx = get_dataloaders(
        "data/features",
        "docs/mapping_coarse.xlsx",
        batch_size=256,
        include_coords=True,
        num_workers=4,
        pin_memory=True
    )
    model = MLPClassifier(num_classes=len(class_to_idx)).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
    
    model.eval()
    all_preds, all_labels, all_probs, all_coords = [], [], [], []
    
    with torch.no_grad():
        for feats, labels, coords in tqdm(test_loader, desc="Evaluating", unit="batch"):
            feats = feats.to(DEVICE, non_blocking=True)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_coords.extend([c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else str(c) for c in coords])
            
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    print("--- Final Test Results ---")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    # 获取类别名称（确保顺序与训练时一致）
    _, _, _, class_to_idx = get_dataloaders("data/features","docs/mapping_coarse.xlsx")
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]

    if print_top10:
        print_top10_predictions(all_preds, all_probs, all_coords, all_labels, class_names, top_k=10)

    # 绘图
    plot_confusion_matrix(all_labels, all_preds, class_names)

if __name__ == "__main__":
    # 默认不打印 top10，如需打印取消下面注释
    # test(print_top10=True)
    test(print_top10=False)