# SRCNN 超分辨率工程

用 PyTorch 从零实现的 SRCNN（Super-Resolution CNN），2014 年第一篇把深度学习
用于图像超分辨率的论文（Dong et al.），是入门超分辨率绕不开的经典模型。

环境：`D:\Aigeneration\venvs\pytorch`（Python 3.13 + torch 2.11 + CUDA，RTX 5090）

## 文件说明

| 文件 | 作用 |
|---|---|
| `model.py` | SRCNN 模型定义，可单独运行自检 |
| `dataset.py` | 数据集：HR 图 → (bicubic 模糊图, 原图) 训练对 |
| `train.py` | 训练脚本，自动保存 PSNR 最优模型到 `models/srcnn_best.pth` |
| `infer.py` | 推理：放大一张图，输出结果图 + 左右对比图，打印 PSNR |
| `make_demo_data.py` | 生成演示图，不下载任何数据集就能跑通 |
| `utils.py` | PSNR 计算 |
| `data/train/`、`data/eval/` | 训练图 / 验证图（PNG/JPG 随便放，尺寸不必一致） |
| `models/` | 权重 + 训练日志 `train_log.txt` |
## 已验证（2026-10-15）

- 模型前向：输入输出形状一致，69,251 参数
- 训练：RTX 5090 CUDA，演示数据 30 epoch，验证 PSNR 18.17 → 26.59 dB
- 推理：SRCNN 26.21 dB vs bicubic 26.02 dB（演示数据规模小，真实数据提升会更明显）

