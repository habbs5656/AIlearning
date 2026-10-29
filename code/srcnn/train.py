r"""训练脚本。

用法(在 D:\Aigeneration\srcnn 目录下):
    快速冒烟测试:
        D:\Aigeneration\venvs\pytorch\Scripts\python.exe train.py --smoke
    正式训练(自己放图到 data\train 和 data\eval):
        D:\Aigeneration\venvs\pytorch\Scripts\python.exe train.py --epochs 100

日志同时打印到控制台并写入 models\train_log.txt,
验证集 PSNR 最优的模型存到 models\srcnn_best.pth。
"""
import argparse
import random
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SRCNN
from dataset import TrainDataset, EvalDataset
from utils import psnr


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def evaluate(model: nn.Module, ds: EvalDataset, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            lr, hr = ds[i]
            sr = model(lr.unsqueeze(0).to(device))[0].clamp(0, 1).cpu().numpy()
            total += psnr(sr, hr.numpy())
    model.train()
    return total / len(ds
                       )
def evaluate_baseline(ds: EvalDataset)->float:
    total = 0.0
    with torch.no_grad():
         for i in range(len(ds)):
             lr, hr = ds[i]
             total += psnr(lr.numpy(), hr.numpy())
    return total / len(ds)


def main():
    ap = argparse.ArgumentParser(description='Train SRCNN')
    ap.add_argument('--train-dir', default='data/train')
    ap.add_argument('--eval-dir', default='data/eval')
    ap.add_argument('--scale', type=int, default=3, help='放大倍数(2/3/4)')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--patch-size', type=int, default=96)
    ap.add_argument('--lr', type=float, default=1e-4, help='学习率')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--samples-per-image', type=int, default=100,
                    help='每张训练图每 epoch 采样的 patch 数')
    ap.add_argument('--smoke', action='store_true',
                    help='冒烟模式: 极小规模快速跑通全流程')
    ap.add_argument('--log-file', default='models/train_log.txt')
    args = ap.parse_args()

    if args.smoke:  # 冒烟模式参数覆盖
        args.epochs = 2
        args.batch_size = 8
        args.patch_size = 64
        args.num_workers = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f'device : {device}'
        + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
    log(f'args   : {vars(args)}')

    train_ds = TrainDataset(args.train_dir, patch_size=args.patch_size,
                            scale=args.scale,
                            samples_per_image=args.samples_per_image)
    eval_ds = EvalDataset(args.eval_dir, scale=args.scale)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    log(f'train  : {len(train_ds)} samples ({len(train_ds.files)} images)')
    log(f'eval   : {len(eval_ds)} images')

    baseline = evaluate_baseline(eval_ds)
    log(f'bicubic: {baseline:.2f} dB  <- 网络的起点, 也是底线')

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_psnr = -1.0
    os.makedirs(os.path.dirname(args.log_file) or '.', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss_sum, n = 0.0, 0
        for lr_img, hr_img in train_dl:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(lr_img), hr_img)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * lr_img.size(0)
            n += lr_img.size(0)
        train_loss = loss_sum / n

        eval_psnr = evaluate(model, eval_ds, device)
        mark = ''
        if eval_psnr > best_psnr:
            best_psnr = eval_psnr
            torch.save(model.state_dict(), 'models/srcnn_best.pth')
            mark = '  <- best, saved'
        log(f'epoch {epoch:3d}/{args.epochs} | loss {train_loss:.6f} | '
            f'eval {eval_psnr:.2f} dB ({eval_psnr - baseline:+.2f} vs bicubic) | '
            f'{time.time() - t0:.1f}s{mark}')


    torch.save(model.state_dict(), 'models/srcnn_last.pth')
    log(f'done. best eval PSNR = {best_psnr:.2f} dB -> models/srcnn_best.pth')

    with open(args.log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')


if __name__ == '__main__':
    main()
