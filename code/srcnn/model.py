"""SRCNN 模型定义。

论文: Image Super-Resolution Using Deep Convolutional Networks
      (Dong et al., ECCV 2014 / TPAMI 2015)

结构就三层卷积（所以叫 Super-Resolution CNN）:
  1. 特征提取   9x9 卷积, 3 -> 64
  2. 非线性映射 5x5 卷积, 64 -> 32
  3. 重建       5x5 卷积, 32 -> 3

输入是「双三次插值(bicubic)放大到目标尺寸」的低分辨率图，
输出是对应的高分辨率图。padding='same'，尺寸不变。
"""
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),  # 特征提取
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),            # 非线性映射
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),  # 重建
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == '__main__':
    # 自检: 随机输入跑一遍, 打印形状
    model = SRCNN()
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print('input :', tuple(x.shape))
    print('output:', tuple(y.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params:,}')
