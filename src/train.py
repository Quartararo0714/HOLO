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

from datasets import compile_data
from tools import SimpleLoss, get_val_info, get_batch_iou, get_nusc_maps, localize_loss, Logger_train
from modules.Locator import Locator
from torch.cuda.amp import GradScaler

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
    
    model.train()

    total_steps = args.start_step
    scaler = GradScaler(enabled=False)
    logger = Logger_train(args, scheduler, optimizer, len(trainloader))
    logger.total_steps = total_steps

    epoch = args.start_step//len(trainloader)
    num_epochs = args.num_steps//len(trainloader)
    print(f'Starting training from epoch {epoch}, total epochs: {num_epochs}')

    loss_bce = SimpleLoss(args.pos_weight).to(device)
    should_keep_training = True
    while should_keep_training:
        model.train()
        for i, batch in enumerate(trainloader):
            optimizer.zero_grad()
            imgs, rots, trans, intrins, post_rots, post_trans, sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center= [x.to(device) for x in batch]
            preds_bev, preds_sdmap, flow_preds = model(imgs, rots, trans, intrins, post_rots, post_trans, sdmap_input)
            loss_sem_bev = loss_bce(preds_bev, sdmap_bev_mask)
            loss_sem_sdmap = loss_bce(preds_sdmap, sdmap_mask)
            loss_pose, metrics = localize_loss(flow_preds, sd_delta, angle_gt, bev_center, [256,256], egopose_gt, sdmap_center, sdmap_angle)
            loss = 1000*(loss_sem_bev + loss_sem_sdmap) + loss_pose
            # loss = 1000*(loss_sem_bev) + loss_pose
            # loss = loss_pose
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step()
            else:
                print(f'Skipping LR scheduler step at step {total_steps} due to scale {scale}')
            
            metrics.update({'loss': loss.cpu().item()})
            _, _, iou_list_bev = get_batch_iou(preds_bev, sdmap_bev_mask)
            _, _, iou_list_sdmap = get_batch_iou(preds_sdmap, sdmap_mask)
            metrics.update({
                'iou_bev_road': iou_list_bev[0],
                'iou_bev_building': iou_list_bev[1],
                'iou_sdmap_road': iou_list_sdmap[0],
                'iou_sdmap_building': iou_list_sdmap[1],
            })
            logger.push(metrics)
            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break
        if epoch % 2 == 1 and not epoch >= num_epochs:     
            epoch+=1     
            continue

        results = evaluate.validate_process(model, valloader, args)
        val_mdis = results['val_mace']
        print('\033[1;94m'+'Epoch: [{}/{}], Loss: {}. \033[0m'
                  .format(epoch+1, num_epochs, val_mdis.item()))

        if val_mdis < best_dist:
            best_dist = val_mdis
            checkpoint = {
                'best_dist': best_dist,
                'steps': total_steps,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule':scheduler.state_dict()
            }
            PATH = 'checkpoints/best_checkpoint_{}_corr1.pth'.format(args.name)
            torch.save(checkpoint, PATH)
            print('\033[1;94m'+"Save the best of {}, at {}\033[0m".format(val_mdis, PATH))
        else:
            print('\033[1;91m'+"Val has no improvement vs {}!\033[0m".format(best_dist))
        with open(logger.file_name, 'a') as file:
            file.write('Epoch: [{}/{}], mdis: {}, the best: {}, at {}.'
                .format(epoch+1, num_epochs, val_mdis.item(), best_dist, checkpoint['steps']) + '\n')                       
        epoch+=1

    print("The minist distance is {}m!".format(best_dist))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config/train_cfg.json")
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ckpt_path', type=str, help='checkpoint path')#/data/mapLoc/LSS_seg/src/checkpoints/best_checkpoint_map_loc_hd.pth
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





