"""小工具: PSNR (峰值信噪比) —— 超分辨率最常用的客观指标, 单位 dB, 越高越好。"""
import math

import numpy as np


def psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    """输入为 [0,1] 取值的数组, 形状一致。"""
    mse = np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)
