import argparse
import time
import json
from easydict import EasyDict
import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from tqdm import tqdm
from datasets import compile_data
from tools import get_nusc_maps
from modules.Locator import Locator

import evaluate

import warnings
warnings.filterwarnings("ignore")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

@torch.no_grad()
def test_model_fps(model, val_dataset, device, warmup=5, max_batches=100):
    """
    测试模型的推理速度（FPS）
    Args:
        model: 已加载好权重的 PyTorch 模型
        val_dataset: 验证集 DataLoader（与训练时相同）
        device: 设备 (cuda 或 cpu)
        warmup: 前几次不计入统计（用于 GPU 预热）
        max_batches: 最多测试多少个 batch（避免太慢）
    Returns:
        平均FPS 和 每帧平均耗时(ms)
    """
    model.eval()
    times = []
    total_images = 0

    print(f"🔥 Start measuring inference FPS (warmup={warmup}, max_batches={max_batches})")

    for i, data_blob in enumerate(tqdm(val_dataset)):
        imgs, rots, trans, intrins, post_rots, post_trans, \
        sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, \
        angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center = [x.to(device) for x in data_blob]

        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.time()

        # forward only
        preds_bev, preds_sdmap, flow_preds = model(
            imgs, rots, trans, intrins, post_rots, post_trans, sdmap_input
        )

        torch.cuda.synchronize() if device.type == "cuda" else None
        t2 = time.time()

        if i >= warmup:  # 跳过前几次
            times.append(t2 - t1)
            total_images += imgs.shape[0]

        if i >= max_batches + warmup - 1:
            break

    avg_time_per_batch = np.mean(times)
    avg_time_per_image = avg_time_per_batch / val_dataset.batch_size
    fps = 1.0 / avg_time_per_image

    print(f"\n✅ Average inference time per image: {avg_time_per_image*1000:.2f} ms")
    print(f"🚀 Inference FPS: {fps:.2f}")
    print(f"📦 Tested on {total_images} images, batch size = {val_dataset.batch_size}")

    return fps, avg_time_per_image

def train(args):
    grid_conf = {
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
    }
    sdmap_conf = {
        'xbound': args.xbound_sd,
        'ybound': args.ybound_sd,
        'zbound': args.zbound_sd,
    }
    data_aug_conf = {
        'resize_lim': args.resize_lim,
        'final_dim': args.final_dim,
        'H': args.H,
        'W': args.W,
        'bot_pct_lim': args.bot_pct_lim,
        'rot_lim': args.rot_lim,
        'rand_flip': args.rand_flip,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': args.ncams,
        'outC': args.outC,
    }
    nusc_maps = get_nusc_maps(args.map_folder)
    trainloader, valloader = compile_data(args.version, args.dataroot, nusc_maps=nusc_maps, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, sdmap_conf=sdmap_conf, map_preprocess=False, osm_path=args.OSM_path,
                                          is_sdmap=args.is_sdmap, bsz=args.batch_size, nworkers=args.nworkers, parser_name='segmentationdata')
    device = torch.device('cpu') if args.gpuid < 0 else torch.device(f'cuda:{args.gpuid}')
    model = Locator(grid_conf, data_aug_conf, outC=args.outC)
    model.to(device)
    print("Parameter Count: %d" % count_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    best_dist = 100000000
    print(args.ckpt_path)
    if args.ckpt_path is not None:
        PATH = args.ckpt_path   # 'checkpoints/best_checkpoint.pth'
        if os.path.isfile(PATH):
            checkpoint = torch.load(PATH)
            best_dist = checkpoint['best_dist']
            args.start_step = checkpoint['steps']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_schedule'])
            print("Have load state_dict from: {}".format(args.ckpt_path))
            print('Load checkpoint at steps {}.'.format(checkpoint['steps']))
    print('Best distance so far {}.'.format(best_dist))
    # test_model_fps(model, valloader, device)
    results = evaluate.validate_process(model, valloader, args)
    val_mdis = results['val_mace']
    print(' Loss: {}. '.format(val_mdis.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/train_cfg.json")
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, help='checkpoint path')
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config = EasyDict(config)
    config['start_step'] = args.start_step
    config['batch_size'] = args.batch_size
    config['ckpt_path'] = args.ckpt_path
    print(config)
    setup_seed(42)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(config)

