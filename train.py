"""
胸部X光肺炎分类（Pneumonia Classification on Chest X-ray）
模型：ResNet-18 迁移学习（ImageNet 预训练 + 微调）
数据：Kermany et al. 2018, Chest X-Ray Images (Pneumonia)
结果：AUC ~0.95，准确率 86%，敏感度 100% / 特异度 55%（阈值0.5，可调）
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# ===== 配置 =====
BASE = "D:/AAA/ChestXRay2017/chest_xray"   # 改成你的数据路径
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# ===== 数据预处理 =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),          # (H,W,C) -> (C,H,W) 且归一化到 0~1
])


class ChestDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):   # 0=正常 1=肺炎
            folder = os.path.join(root, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")   # ResNet 要 3 通道
        return transform(img), label


train_loader = DataLoader(ChestDataset(f"{BASE}/train"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ChestDataset(f"{BASE}/test"), batch_size=BATCH_SIZE)

# ===== 模型：ResNet-18 迁移学习 =====
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)   # 1000 类 -> 2 类

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)   # 微调用小 lr

# ===== 训练 =====
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} 完成")

# ===== 评估 =====
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_labels.extend(yb.tolist())
        all_probs.extend(logits.softmax(dim=1)[:, 1].tolist())   # 肺炎概率

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()
print("混淆矩阵:\n", cm)
print("敏感度:", tp / (tp + fn), " 特异度:", tn / (tn + fp))
print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))
print("AUC:", roc_auc_score(all_labels, all_probs))

# ===== 阈值扫描（筛查 vs 确诊）=====
print("\n阈值    敏感度    特异度")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    preds = (np.array(all_probs) > t).astype(int)
    c = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = c.ravel()
    print(f"{t}    {tp / (tp + fn):.3f}    {tn / (tn + fp):.3f}")
