"""训练脚本。"""
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
def evaluate(model: nn.Module, ds: EvalDataset, device):
    """返回 (平均 PSNR, 每张图的 PSNR 列表)"""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(len(ds)):
            lr, hr = ds[i]
            sr = model(lr.unsqueeze(0).to(device))[0].clamp(0, 1).cpu().numpy()
            scores.append(psnr(sr, hr.numpy()))
    model.train()
    return float(np.mean(scores)), scores


def evaluate_baseline(ds: EvalDataset):
    """bicubic 基线: 输入就是 bicubic 放大的结果, 直接拿它判分"""
    scores = []
    with torch.no_grad():
        for i in range(len(ds)):
            lr, hr = ds[i]
            scores.append(psnr(lr.numpy(), hr.numpy()))
    return float(np.mean(scores)), scores



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
    ap.add_argument('--log-file', default=None,
                    help='日志路径, 不填自动用 models/train_log_<tag>.txt')

    ap.add_argument('--kernels', type=int, nargs=3, default=[9, 5, 5],
                    help='三层卷积的核大小, 必须都是奇数')
    ap.add_argument('--widths', type=int, nargs=2, default=[64, 32],
                    help='前两层输出通道数')
    ap.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam',
                    help='优化器')
    ap.add_argument('--tag', default='default',
                    help='实验名, 决定模型和日志文件名')

    args = ap.parse_args()

    if args.smoke:
        args.epochs = 2
        args.batch_size = 8
        args.patch_size = 64
        args.num_workers = 0
    if args.log_file is None:
        args.log_file = f'models/train_log_{args.tag}.txt'
    best_path = f'models/srcnn_{args.tag}_best.pth'
    last_path = f'models/srcnn_{args.tag}_last.pth'

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

    baseline, base_per_img = evaluate_baseline(eval_ds)
    log(f'bicubic: {baseline:.2f} dB  <- 网络的起点, 也是底线')

    model = SRCNN(kernels=tuple(args.kernels), widths=tuple(args.widths)).to(device)

    criterion = nn.MSELoss()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

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

        eval_psnr, per_img = evaluate(model, eval_ds, device)

        mark = ''
        if eval_psnr > best_psnr:
            best_psnr = eval_psnr
            torch.save(model.state_dict(), best_path)
            mark = '  <- best, saved'
        log(f'epoch {epoch:3d}/{args.epochs} | loss {train_loss:.6f} | '
            f'eval {eval_psnr:.2f} dB ({eval_psnr - baseline:+.2f} vs bicubic) | '
            f'{time.time() - t0:.1f}s{mark}')


    torch.save(model.state_dict(), last_path)
    log(f'done. best eval PSNR = {best_psnr:.2f} dB -> {best_path}')
    log('per-image PSNR  (bicubic -> model):')
    for name, b, s in zip(eval_ds.files, base_per_img, per_img):
        log(f'  {os.path.basename(name):<28} {b:6.2f} -> {s:6.2f}   ({s - b:+.2f})')
    log(f'spread: bicubic std {np.std(base_per_img):.3f} | '
        f'model std {np.std(per_img):.3f} | '
        f'model range {max(per_img) - min(per_img):.2f} dB')


    with open(args.log_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')


if __name__ == '__main__':
    main()
