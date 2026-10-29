"""数据集：把高分辨率图(HR)变成「低分辨率输入 + 高分辨率标签」的训练对。

流程:
    HR --双三次缩小 1/scale--> 小图 --双三次放大回原尺寸--> LR(网络输入)
    HR 本身作为标签。
SRCNN 学的就是 LR -> HR 的映射，即「把模糊的放大图变清晰」。
"""
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def list_images(root: str):
    if not os.path.isdir(root):
        raise RuntimeError(f'目录不存在: {root}')
    files = [os.path.join(root, f) for f in sorted(os.listdir(root))
             if f.lower().endswith(IMG_EXTS)]
    if not files:
        raise RuntimeError(f'目录里没有图片: {root}')
    return files


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL 图 -> [C,H,W] float32, 取值 [0,1]"""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def degrade(img: Image.Image, scale: int) -> Image.Image:
    """HR -> 双三次缩小再放大, 得到模糊的 LR(尺寸与 HR 相同)"""
    w, h = img.size
    small = img.resize((w // scale, h // scale), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


class TrainDataset(Dataset):
    """训练集: 随机裁 patch + 随机翻转, 每张图复用 samples_per_image 次"""

    def __init__(self, root: str, patch_size: int = 96, scale: int = 3,
                 samples_per_image: int = 100):
        self.files = list_images(root)
        self.patch_size = patch_size
        self.scale = scale
        self.samples_per_image = samples_per_image

    def __len__(self):
        return len(self.files) * self.samples_per_image

    def __getitem__(self, idx):
        img = Image.open(random.choice(self.files)).convert('RGB')

        # 图片太小就先等比放大到能裁出 patch
        p = self.patch_size
        if img.size[0] < p or img.size[1] < p:
            r = max(p / img.size[0], p / img.size[1])
            img = img.resize((max(p, round(img.size[0] * r)),
                              max(p, round(img.size[1] * r))), Image.BICUBIC)

        # 随机裁 HR patch
        w, h = img.size
        x0 = random.randint(0, w - p)
        y0 = random.randint(0, h - p)
        hr_patch = img.crop((x0, y0, x0 + p, y0 + p))

        # 随机水平/垂直翻转
        if random.random() < 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)

        lr_patch = degrade(hr_patch, self.scale)
        return to_tensor(lr_patch), to_tensor(hr_patch)


class EvalDataset(Dataset):
    """验证集: 整张图评测 (裁到 scale 的整数倍, 方便对齐)"""

    def __init__(self, root: str, scale: int = 3, limit: int = 0):
        files = list_images(root)
        if limit:
            files = files[:limit]
        self.files = files
        self.scale = scale
        # 图片统一缩到最长边 512 以内, 避免 5090 上也占太多显存
        self.max_side = 512

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        w, h = img.size
        if max(w, h) > self.max_side:
            r = self.max_side / max(w, h)
            w, h = round(w * r), round(h * r)
            img = img.resize((w, h), Image.BICUBIC)
        # 裁到 scale 整数倍
        w -= w % self.scale
        h -= h % self.scale
        img = img.crop((0, 0, w, h))

        hr = img                       # 标签
        lr = degrade(img, self.scale)  # 输入
        return to_tensor(lr), to_tensor(hr)
