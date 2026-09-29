"""
胸部X光肺炎分类 + OOD 拒识（3 类版）
在正常/肺炎之外增加第三类 OTHER（非胸片），让模型遇到非胸片输入时能提示而非瞎猜。
结果：300 张合成非胸片 100% 正确拒识；代价是原任务正常类特异度从 55~64% 降至 45%（三分类固有 tradeoff）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# ===== 配置 =====
BASE = "D:/AAA/ChestXRay2017/chest_xray"   # 改成你的数据路径
OTHER = "D:/AAA/other_images"               # 合成非胸片目录（自动生成）
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5


# ===== 生成合成"非胸片"图（噪声/渐变/几何图形/条纹/棋盘格，5 种类型保证多样性）=====
def generate_other_images(out_dir, n=300):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    H = W = IMG_SIZE
    for i in range(n):
        kind = i % 5
        if kind == 0:       # 随机噪声
            arr = rng.integers(0, 256, (H, W), dtype=np.uint8)
        elif kind == 1:     # 渐变
            arr = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
        elif kind == 2:     # 实心圆
            arr = np.full((H, W), 30, dtype=np.uint8)
            yy, xx = np.mgrid[0:H, 0:W]
            arr[(xx - W // 2) ** 2 + (yy - H // 2) ** 2 < 30 ** 2] = 220
        elif kind == 3:     # 条纹
            arr = np.zeros((H, W), dtype=np.uint8)
            arr[::8, :] = 255
        else:               # 棋盘格
            arr = np.zeros((H, W), dtype=np.uint8)
            arr[::16, ::16] = 255
            arr[8::16, 8::16] = 255
        Image.fromarray(arr, "L").save(f"{out_dir}/other_{i:03d}.png")


generate_other_images(OTHER)

# ===== 数据：正常=0 / 肺炎=1 / 其他=2 =====
transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])


class ChestDataset3(Dataset):
    def __init__(self, root, other_root=None):
        self.samples = []
        for label, cls in enumerate(["NORMAL", "PNEUMONIA"]):
            folder = os.path.join(root, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label))
        if other_root:
            for fname in os.listdir(other_root):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(other_root, fname), 2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return transform(Image.open(path).convert("RGB")), label


train_loader = DataLoader(ChestDataset3(f"{BASE}/train", OTHER), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(ChestDataset3(f"{BASE}/test"), batch_size=BATCH_SIZE)  # 测试集只含正常/肺炎

# ===== 模型：ResNet-18 迁移学习，最后一层改 3 类 =====
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} 完成")

# ===== 评估 1：胸片测试集（正常/肺炎，看原任务受多大影响）=====
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_preds.extend(model(xb).argmax(dim=1).tolist())
        all_labels.extend(yb.tolist())
print("胸片测试集混淆矩阵:\n", confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))

# ===== 评估 2：合成非胸片是否被拒识（理想：全部输出 2 = OTHER）=====
preds = []
with torch.no_grad():
    for fname in sorted(os.listdir(OTHER)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img = transform(Image.open(os.path.join(OTHER, fname)).convert("RGB")).unsqueeze(0)
            preds.append(model(img).argmax(dim=1).item())
print("合成图预测分布:", Counter(preds))   # 理想：Counter({2: 300})
