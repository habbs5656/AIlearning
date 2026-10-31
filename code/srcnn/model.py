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
    def __init__(self, num_channels: int = 3,
                 kernels=(9, 5, 5), widths=(64, 32)):
        super().__init__()
        assert len(kernels) == 3, 'kernels 要 3 个: 特征提取/非线性映射/重建'
        assert len(widths) == 2, 'widths 要 2 个: 前两层的输出通道数'
        assert all(k % 2 == 1 for k in kernels), '核大小必须是奇数, 否则尺寸会变'
        k1, k2, k3 = kernels
        w1, w2 = widths
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, w1, kernel_size=k1, padding=k1 // 2),  # 特征提取
            nn.ReLU(inplace=True),
            nn.Conv2d(w1, w2, kernel_size=k2, padding=k2 // 2),           # 非线性映射
            nn.ReLU(inplace=True),
            nn.Conv2d(w2, num_channels, kernel_size=k3, padding=k3 // 2), # 重建
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



if __name__ == '__main__':
    model = SRCNN()
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print('input :', tuple(x.shape))
    print('output:', tuple(y.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params:,}')
