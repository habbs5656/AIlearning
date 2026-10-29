r"""推理脚本: 用训练好的模型放大一张图。

用法:
    D:\Aigeneration\venvs\pytorch\Scripts\python.exe infer.py --input 图.png

输出(和输入同目录):
    图_sr.png      —— SRCNN 放大结果
    图_compare.png —— 左: bicubic 直接放大 | 右: SRCNN, 直观看差别

如果不传 --input, 会自动拿 data\eval 里第一张图做演示。
"""
import argparse
import os

import numpy as np
from PIL import Image
import torch

from model import SRCNN
from dataset import degrade, to_tensor, list_images
from utils import psnr


def main():
    ap = argparse.ArgumentParser(description='SRCNN inference')
    ap.add_argument('--input', default='', help='输入图片路径')
    ap.add_argument('--weights', default='models/srcnn_best.pth')
    ap.add_argument('--scale', type=int, default=3)
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit(f'找不到模型权重: {args.weights}\n请先运行 train.py 完成训练。')

    in_path = args.input
    if not in_path:
        eval_files = list_images('data/eval')
        in_path = eval_files[0]
        print(f'--input 未指定, 自动使用: {in_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # HR = 原图(裁到 scale 整数倍), LR = 降质后的输入
    hr_img = Image.open(in_path).convert('RGB')
    w, h = hr_img.size
    w -= w % args.scale
    h -= h % args.scale
    hr_img = hr_img.crop((0, 0, w, h))
    lr_img = degrade(hr_img, args.scale)

    # SRCNN 超分
    with torch.no_grad():
        x = to_tensor(lr_img).unsqueeze(0).to(device)
        sr = model(x)[0].clamp(0, 1).cpu().numpy()
    sr_img = Image.fromarray((sr.transpose(1, 2, 0) * 255).round().astype(np.uint8))

    root, ext = os.path.splitext(in_path)
    sr_path = root + '_sr' + ext
    cmp_path = root + '_compare' + ext
    sr_img.save(sr_path)

    # 对比图: 左 bicubic / 右 SRCNN
    border = 4
    cmp = Image.new('RGB', (lr_img.width * 2 + border, lr_img.height), (255, 0, 0))
    cmp.paste(lr_img, (0, 0))
    cmp.paste(sr_img, (lr_img.width + border, 0))
    cmp.save(cmp_path)

    hr_arr = np.asarray(hr_img, dtype=np.float64) / 255.0
    lr_arr = np.asarray(lr_img, dtype=np.float64) / 255.0
    sr_arr = sr.transpose(1, 2, 0).astype(np.float64)
    print(f'input : {in_path} ({w}x{h}, scale x{args.scale})')
    print(f'device: {device}')
    print(f'bicubic PSNR : {psnr(lr_arr, hr_arr):.2f} dB')
    print(f'SRCNN   PSNR : {psnr(sr_arr, hr_arr):.2f} dB')
    print(f'saved : {sr_path}')
    print(f'saved : {cmp_path}')


if __name__ == '__main__':
    main()
