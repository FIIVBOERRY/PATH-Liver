import torch
import torch.nn as nn
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.model import MLPClassifier
from src.utils.matrix import calculate_metrics

# 超参
EPOCHS = 20
LR = 1e-3
FEATURE_DIR = "data/features"
MAPPING_PATH = "docs/mapping.xlsx"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    train_loader, val_loader, _, class_to_idx = get_dataloaders(FEATURE_DIR, MAPPING_PATH)
    model = MLPClassifier(num_classes=len(class_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_val_f1 = 0
    print(f"使用设备: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        val_metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Val Acc {val_metrics['acc']:.4f}, F1 {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="Evaluating"):
            feats = feats.to(DEVICE)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return calculate_metrics(all_labels, all_preds, all_probs)

if __name__ == "__main__":
    train()