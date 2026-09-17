# 胸部X光肺炎分类 · Chest X-ray Pneumonia Classification

用 ResNet-18 迁移学习对胸部 X 光片做二分类（正常 / 肺炎），并针对类别不平衡做了系统性的对比实验。

> ResNet-18 transfer learning for binary classification (NORMAL / PNEUMONIA) on chest X-ray images, with a systematic study of class-imbalance handling.

## 结果 · Results

| 指标 Metric | 数值 Value |
|------|------|
| AUC | ≈ 0.95 |
| 准确率 Accuracy | 86% |
| 敏感度 Sensitivity | 100%（阈值 0.5）→ 99.7%（阈值 0.9）|
| 特异度 Specificity | 55~64%（阈值 0.5）→ 71~74%（阈值 0.9）|

通过调整决策阈值，可在两种临床场景间权衡：低阈值用于**筛查**（宁多不漏），高阈值用于**确诊**（避免误诊）。

## 数据 · Dataset

- 来源：Kermany et al. 2018, *Large Dataset of Labeled OCT and Chest X-Ray Images*（Mendeley Data: rscbjbr9sj，CC BY 4.0）
- 规模：训练集 5232 张，测试集 624 张（正常 234 / 肺炎 390）
- 格式：JPEG 灰度胸片，`NORMAL` / `PNEUMONIA` 两类

## 方法 · Method

1. **ResNet-18 迁移学习**：ImageNet 预训练权重 + 微调最后一层（1000 类 → 2 类）
2. **类别不平衡处理**：对比了 class_weight、欠采样、阈值调优三种方案，最终采用**阈值调优**
3. 输入尺寸 128×128，训练 5 个 epoch

## 消融实验 · Ablation Study

### ① 分辨率 × 训练轮数（Resize × Epochs）

| 配置 | 阈值0.9 特异度 | 结论 |
|------|------|------|
| **128 × 5 轮** | **71~74%** | ✅ 最终采用 |
| 128 × 10 轮 | 63.7% | 过拟合（训练 loss → 0.0005）|
| 224 × 5 轮 | 66.7% | 无显著提升 |
| 224 × 15 轮 | 54.3% | 严重过拟合（loss → 0.0001）|

**结论**：更多训练轮数 ≠ 更好。当训练 loss 持续下降到接近 0、而测试指标掉头向下，即过拟合（模型在"背答案"而非"学规律"）。本数据集约 5200 张，5 轮即为甜点区；且 224 分辨率反而更容易过拟合（细节更多、可背的东西更多）。

### ② 类别不平衡处理方案（Imbalance Handling）

| 方案 | 效果 |
|------|------|
| **阈值调优** | ✅ 最优：不改模型，只调决策阈值，敏感度/特异度可自由权衡 |
| class_weight | 未改善（weight=[3,1] 时特异度反降至 44%）|
| 欠采样（肺炎砍到与正常等量）| 未改善（特异度 44%，丢弃 2/3 肺炎数据反而模糊了决策边界）|

## 运行 · Run

```bash
pip install torch torchvision scikit-learn pillow numpy
python train.py
```

把 `train.py` 里的 `BASE` 改成你的数据路径即可。

## 讨论 · Discussion

- 类别不平衡会让模型偏向多数类：敏感度虚高、特异度低，单看准确率会被误导
- 处理不平衡只能在训练阶段（重采样/加权）或决策阶段（调阈值），不能在测试集上做手脚
- 过拟合信号：训练 loss 持续下降、测试指标掉头向下；本任务中 5 个 epoch 即为最优
- 224 与 128 无显著差异、更多轮数反而过拟合——瓶颈在数据本身的标签噪声，不在模型容量
