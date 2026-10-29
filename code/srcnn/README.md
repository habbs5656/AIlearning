# SRCNN 超分辨率工程

用 PyTorch 从零实现的 SRCNN（Super-Resolution CNN），2014 年第一篇把深度学习
用于图像超分辨率的论文（Dong et al.），是入门超分辨率绕不开的经典模型。

环境：`D:\Aigeneration\venvs\pytorch`（Python 3.13 + torch 2.11 + CUDA，RTX 5090）

## 模型是什么

一句话：**把「双三次插值放大的模糊图」变清晰的三层卷积网络。**

```
低分辨率图 --bicubic放大--> 输入 x --┐
                                     ├─ SRCNN 三层卷积 ──> 高分辨率图
                高分辨率图(标签) ----┘ (MSE 监督训练)

Conv 9x9 (3→64)  特征提取
Conv 5x5 (64→32) 非线性映射   共 69,251 参数
Conv 5x5 (32→3)  重建
```

指标是 PSNR (dB)，比 bicubic 基线高就是有效。

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

## 快速开始

在 `D:\Aigeneration\srcnn` 目录下：

```powershell
# 1. 生成演示数据（8 张训练图 + 2 张验证图）
D:\Aigeneration\venvs\pytorch\Scripts\python.exe make_demo_data.py

# 2. 冒烟测试（2 epoch 极小规模，验证链路通不通）
D:\Aigeneration\venvs\pytorch\Scripts\python.exe train.py --smoke

# 3. 正式训练（30~100 epoch）
D:\Aigeneration\venvs\pytorch\Scripts\python.exe train.py --epochs 30 --num-workers 0

# 4. 推理看效果（输出 xxx_sr.png 和 xxx_compare.png）
D:\Aigeneration\venvs\pytorch\Scripts\python.exe infer.py --input 某张图.png
```

## 用自己的图训练

把你的图放进 `data\train`（越多越好，几十张起步）和 `data\eval`（几张即可），
直接跑 train.py。训练集会随机裁 96x96 的 patch 并翻转增广，尺寸不需要统一。

**一个现成的数据来源：你的 Blender 渲染图。** 渲染 256/512 的高采样图当 HR 标签，
SRCNN 学的就是「低采样渲染 → 接近高采样」的映射——以后渲染大图可以先用低采样
快速出图再用它补清晰度（虽然 SRCNN 本身是 baseline，效果有限）。

标准学术数据集（可选）：BSDS500 / DIV2K / Set5 / Set14，下载后把图放进对应文件夹即可。

## 常用参数

```
--scale 3        放大倍数 (2/3/4，默认 3)
--epochs 100     训练轮数
--batch-size 16  批大小
--patch-size 96  训练裁剪尺寸
--lr 1e-4        学习率 (Adam)
--num-workers 0  Windows 下建议 0，否则可能卡住
```

## 已验证（2026-10-15）

- 模型前向：输入输出形状一致，69,251 参数
- 训练：RTX 5090 CUDA，演示数据 30 epoch，验证 PSNR 18.17 → 26.59 dB
- 推理：SRCNN 26.21 dB vs bicubic 26.02 dB（演示数据规模小，真实数据提升会更明显）

## 下一步可以玩的

1. 换真实图片训练，看 PSNR 能比 bicubic 高多少
2. 把中间层激活可视化，直观理解三层各在干什么
3. 进阶模型：FSRCNN（SRCNN 加速版）→ ESPCN → EDSR → SRGAN（SRCNN 一脉相承）
