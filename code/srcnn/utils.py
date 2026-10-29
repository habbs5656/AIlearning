import math

import numpy as np


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """[3,H,W] 的 RGB -> [H,W] 的亮度 Y, 用 BT.601 权重。"""
    return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]


def psnr(sr: np.ndarray, hr: np.ndarray, y_only: bool = True) -> float:
    if y_only and sr.ndim == 3 and sr.shape[0] == 3:
        sr, hr = rgb_to_y(sr), rgb_to_y(hr)
    mse = np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)
