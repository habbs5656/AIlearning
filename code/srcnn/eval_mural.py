"""跨域评测: 在 BSD(自然图)上训练的 SRCNN, 直接用到敦煌壁画上, PSNR 掉多少。

两种口径:
  mural512 : 取 Mural512 的 512 裁块, 用和训练/Set5 完全相同的退化(自造 bicubic x scale)
             -> 只测「域偏移」, 能和 Set5 的 29.72 直接比
  x4       : 用数据集自带的 HR(2048) 与官方 LR X4(512) 配对
             -> 测「域偏移 + 尺度改变」, 模型训练时是 x3, 这里是 x4

用法:
    python eval_mural.py --mode mural512 --limit 40
    python eval_mural.py --mode x4 --limit 40
"""
import argparse
import os
import random

import numpy as np
import torch
from PIL import Image

from dataset import degrade, to_tensor
from model import SRCNN
from utils import psnr

MURAL_ROOT = 'D:/datasets/MuralDH'


def load_model(ckpt: str, device):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def run_patch(model, lr_img: Image.Image, hr_img: Image.Image, device,
              min_std: float = 3 / 255):
    """lr_img 已放大到与 hr_img 同尺寸; 返回 (bicubic 分, 模型分)

    min_std: 几乎纯色的块(bicubic 与 HR 完全相同, PSNR = inf)直接跳过,
             否则一张平墙就把平均值拉成 inf。壁画里这类块很常见。
    """
    hr = to_tensor(hr_img)
    if float(hr.std()) < min_std:
        return None
    lr = to_tensor(lr_img)
    with torch.no_grad():
        sr = model(lr.unsqueeze(0).to(device))[0].clamp(0, 1).cpu()
    return psnr(lr.numpy(), hr.numpy()), psnr(sr.numpy(), hr.numpy())


def eval_mural512(model, limit, scale, device, seed=0):
    root = os.path.join(MURAL_ROOT, 'Mural512')
    files = sorted(os.path.join(root, f) for f in os.listdir(root)
                   if f.lower().endswith('.png'))
    random.Random(seed).shuffle(files)
    rows = []
    for p in files[:limit]:
        hr_img = Image.open(p).convert('RGB')
        lr_img = degrade(hr_img, scale)
        out = run_patch(model, lr_img, hr_img, device)
        if out is None:
            continue
        b, sco = out
        rows.append((os.path.basename(p), b, sco))
    return rows


def eval_x4(model, limit, device, patch=512, seed=0):
    hr_dir = os.path.join(MURAL_ROOT, 'Mural_SR', 'Mural_DataSet')
    lr_dir = os.path.join(MURAL_ROOT, 'Mural_SR', 'Mural_DataSet_LR', 'X4')
    files = sorted(f for f in os.listdir(hr_dir) if f.lower().endswith('.png'))
    random.Random(seed).shuffle(files)
    rows = []
    skipped = 0
    for f in files:
        if len(rows) >= limit:
            break
        hr_img = Image.open(os.path.join(hr_dir, f)).convert('RGB')
        lr_img = Image.open(os.path.join(lr_dir, f.replace('.png', 'x4.png'))).convert('RGB')
        w, h = hr_img.size
        off = (min(w, h) - patch) // 2
        hr_crop = hr_img.crop((off, off, off + patch, off + patch))
        # LR 是 HR 的 1/4, 中心偏移先除以 4 再裁剪对应区域
        s = lr_img.size[0] / hr_img.size[0]
        lo = int(off * s)
        side = int(patch * s)
        lr_crop = lr_img.crop((lo, lo, lo + side, lo + side)).resize(
            (patch, patch), Image.BICUBIC)
        out = run_patch(model, lr_crop, hr_crop, device)
        if out is None:
            skipped += 1
            continue
        b, sco = out
        rows.append((f, b, sco))
    return rows


def report(rows, title):
    b = np.array([r[1] for r in rows])
    s = np.array([r[2] for r in rows])
    gain = s - b
    print(f'\n=== {title}  ({len(rows)} 张) ===')
    print(f'bicubic : {b.mean():6.2f} dB   (std {b.std():.2f})')
    print(f'model   : {s.mean():6.2f} dB   (std {s.std():.2f})')
    print(f'gain    : {gain.mean():+6.2f} dB   (std {gain.std():.2f}, '
          f'中位数 {np.median(gain):+.2f}, 提升的图 {int((gain > 0).sum())}/{len(rows)})')
    for name, bb, ss in rows[:8]:
        print(f'  {name:<22} {bb:6.2f} -> {ss:6.2f}   ({ss - bb:+.2f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='models/srcnn_best.pth')
    ap.add_argument('--mode', default='mural512', choices=['mural512', 'x4', 'both'])
    ap.add_argument('--limit', type=int, default=40)
    ap.add_argument('--scale', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)
    print(f'ckpt: {args.ckpt}  device: {device}  limit: {args.limit}')

    if args.mode in ('mural512', 'both'):
        rows = eval_mural512(model, args.limit, args.scale, device, args.seed)
        report(rows, f'壁画 Mural512, 自造退化 x{args.scale}')

    if args.mode in ('x4', 'both'):
        rows = eval_x4(model, args.limit, device, seed=args.seed)
        report(rows, '壁画 官方配对 X4 (HR 2048 中心裁 512)')


if __name__ == '__main__':
    main()
