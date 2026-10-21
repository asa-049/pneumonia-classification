# -*- coding: utf-8 -*-
"""
胸部X光肺炎分类 · 可复现训练 + 诚实评测
=========================================
ResNet-18 迁移学习，二分类（NORMAL / PNEUMONIA）。

和 2026-09 那版的关键区别（也是这个文件存在的理由）：
1. 固定随机种子：同一条命令跑两次，results.json 一致。
2. 按病人切分 train/val：同一病人的多张片子不会分到两边（按图片切会泄漏，
   val AUC 会虚高到 0.99）。test 原封不动，只在最后跑一次。
3. 阈值只在 val 上选，test 只用来报数。
4. 补齐 ImageNet 归一化：预训练权重要求输入按 mean/std 归一化，原版漏了。
5. 存最好的权重（按 val AUC）和完整结果 results.json。

用法：
    python train.py                     # 默认：128px、5 轮、lr 1e-4
    python train.py --epochs 1 --out ..\\_tmp\\smoke.pt   # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXT = (".jpeg", ".jpg", ".png")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- 工具
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_class_images(root: str):
    """返回 (相对路径列表, 标签, 跳过的非图片数)。sorted 保证顺序稳定，否则切分会漂移。"""
    files, labels, skipped = [], [], 0
    for cls, lab in (("NORMAL", 0), ("PNEUMONIA", 1)):
        for f in sorted(os.listdir(os.path.join(root, cls))):
            if f.lower().endswith(IMG_EXT):
                files.append(f"{cls}/{f}")
                labels.append(lab)
            else:
                skipped += 1
    return files, np.array(labels), skipped


def patient_id(rel_path: str) -> str:
    """person1000_bacteria_1.jpeg -> person1000；IM-0115-0001.jpeg -> IM-0115-0001。"""
    base = os.path.basename(rel_path)
    m = re.match(r"(person\d+)", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def split_by_patient(files, labels, seed: int, val_ratio: float):
    """按病人分组切 train/val。切的是病人，不是图片。"""
    pids = np.array([patient_id(f) for f in files])
    uniq = np.unique(pids)
    grp_label = np.array([labels[pids == g][0] for g in uniq])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    _, g_va = next(sss.split(uniq, grp_label))
    val_groups = set(uniq[g_va])
    mask = np.isin(pids, list(val_groups))
    return np.where(~mask)[0], np.where(mask)[0], pids


# ---------------------------------------------------------------- 数据
class ChestDataset(Dataset):
    def __init__(self, root: str, files, labels, tf):
        self.root, self.files, self.labels, self.tf = root, files, labels, tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("RGB")
        return self.tf(img), int(self.labels[i])


def build_transforms(img_size: int):
    """增强只给 train；不做水平翻转（胸片镜像会把心脏翻到右边，解剖上错误）。"""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.90, 1.0)),
        transforms.RandomAffine(degrees=7, translate=(0.03, 0.03)),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_model() -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


# ---------------------------------------------------------------- 评测
@torch.no_grad()
def predict(model, loader):
    model.eval()
    ps, ys = [], []
    for xb, yb in loader:
        ps.append(model(xb).softmax(dim=1)[:, 1])
        ys.append(yb)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def metrics(probs, labels, t: float) -> dict:
    pred = (probs >= t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    return {
        "threshold": round(float(t), 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sens": round(tp / max(tp + fn, 1), 4),
        "spec": round(tn / max(tn + fp, 1), 4),
        "acc": round((tp + tn) / max(len(labels), 1), 4),
    }


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:\AAA\ChestXRay2017\chest_xray")
    ap.add_argument("--out", default=None, help="权重保存路径，默认放脚本同级 best_resnet18.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.out or os.path.join(here, "best_resnet18.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    set_seed(args.seed)
    t_start = time.time()

    # ---- 读数据 + 按病人切分 ----
    train_root = os.path.join(args.data, "train")
    files, labels, skipped = list_class_images(train_root)
    tr_idx, va_idx, pids = split_by_patient(files, labels, args.seed, args.val_ratio)

    X, y = np.array(files), labels
    tr_files, tr_y = X[tr_idx], y[tr_idx]
    va_files, va_y = X[va_idx], y[va_idx]

    # 硬性自检：坏数据宁可停，不要静默训下去
    assert len(y) == 5232, f"图片数不对: {len(y)}（期望 5232）"
    assert not (set(pids[tr_idx]) & set(pids[va_idx])), "train/val 病人重叠"
    assert not (set(tr_files) & set(va_files)), "train/val 图片重叠"

    print(f"图片 {len(y)} 张（跳过非图片 {skipped} 个）")
    print(f"病人 {len(set(pids))} 个 | train {len(tr_y)} 张 | val {len(va_y)} 张")
    print(f"肺炎占比 train={tr_y.mean():.2%} val={va_y.mean():.2%}")

    # ---- DataLoader ----
    train_tf, eval_tf = build_transforms(args.img_size)
    train_loader = DataLoader(ChestDataset(train_root, tr_files, tr_y, train_tf),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(ChestDataset(train_root, va_files, va_y, eval_tf),
                            batch_size=args.batch, shuffle=False, num_workers=0)

    # ---- 训练 ----
    model = build_model()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best_auc, best_ep = [], -1.0, -1
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        tr_loss = tot / n

        vp, vy = predict(model, val_loader)
        val_auc = float(roc_auc_score(vy, vp))
        val_acc = float(((vp > 0.5) == vy).mean())
        sec = round(time.time() - t0, 1)
        history.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                        "val_auc": round(val_auc, 4), "val_acc": round(val_acc, 4), "sec": sec})

        mark = ""
        if val_auc > best_auc:                     # 只按 val 决定留哪一轮
            best_auc, best_ep = val_auc, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "val_auc": val_auc,
                        "seed": args.seed, "img_size": args.img_size,
                        "classes": ["NORMAL", "PNEUMONIA"]}, ckpt_path)
            mark = "   <- 存盘"
        print(f"ep {ep}/{args.epochs} | train loss {tr_loss:.4f} | "
              f"val AUC {val_auc:.4f} | val acc {val_acc:.4f} | {sec}s{mark}", flush=True)

    print(f"最佳 epoch={best_ep}  val AUC={best_auc:.4f}")

    # ---- 阈值只在 val 上选 ----
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    vp, vy = predict(model, val_loader)
    table = [metrics(vp, vy, t) for t in np.arange(0.05, 0.96, 0.01)]
    youden = max(table, key=lambda r: r["sens"] + r["spec"])                       # 敏+特之和最大
    screening = max([r for r in table if r["sens"] >= 0.99], key=lambda r: r["spec"])  # 漏诊 ≤1%

    # ---- test 只跑一次 ----
    test_root = os.path.join(args.data, "test")
    te_files, te_y, _ = list_class_images(test_root)
    test_loader = DataLoader(ChestDataset(test_root, te_files, te_y, eval_tf),
                             batch_size=args.batch, shuffle=False, num_workers=0)
    tp, ty = predict(model, test_loader)
    test_auc = float(roc_auc_score(ty, tp))
    ty_youden = metrics(tp, ty, youden["threshold"])
    ty_screen = metrics(tp, ty, screening["threshold"])

    print(f"\ntest {len(ty)} 张，肺炎占比 {ty.mean():.2%}，AUC {test_auc:.4f}")
    print(f"  @Youden  t={youden['threshold']}: sens {ty_youden['sens']:.4f}  spec {ty_youden['spec']:.4f}")
    print(f"  @筛查点  t={screening['threshold']}: sens {ty_screen['sens']:.4f}  spec {ty_screen['spec']:.4f}")

    results = {
        "seed": args.seed, "img_size": args.img_size, "epochs": args.epochs,
        "split": f"patient-level {args.val_ratio:.0%}",
        "n_train": int(len(tr_y)), "n_val": int(len(va_y)), "n_test": int(len(ty)),
        "best_epoch": int(ck["epoch"]), "val_auc": round(float(ck["val_auc"]), 4),
        "test_auc": round(test_auc, 4), "history": history,
        "val_thresholds": {"youden": youden, "screening": screening},
        "test_metrics": {"youden": ty_youden, "screening": ty_screen},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    res_path = os.path.join(here, "results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已存 {res_path}\n权重 {ckpt_path}")


if __name__ == "__main__":
    main()
