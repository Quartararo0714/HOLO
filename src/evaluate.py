import torch
import numpy as np
import time
from tools import Logger, localize_loss

@torch.no_grad()
def validate_process(model, val_dataset, args):
    """ Perform evaluation on the validation split """
    device = torch.device('cpu') if args.gpuid < 0 else torch.device(f'cuda:{args.gpuid}')
    model.eval()
    mace_list = []
    timeall=[]
    logger = Logger(args)
    for i_batch, data_blob in enumerate(val_dataset):
        time_start = time.time()
        imgs, rots, trans, intrins, post_rots, post_trans, sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center  = [x.to(device) for x in data_blob]

        preds_bev, preds_sdmap, flow_preds = model(imgs, rots, trans, intrins, post_rots, post_trans, sdmap_input)       
        _, metrics = localize_loss(flow_preds, sd_delta, angle_gt, bev_center, [256,256], egopose_gt, sdmap_center, sdmap_angle)
        logger.push(metrics)
       
        mace_list.append(metrics['edist'])
        torch.cuda.empty_cache()
        time_end = time.time()
        timeall.append(time_end-time_start)
    
    mace = np.mean(np.array(mace_list))
    logger._print_training_status()
    print("Validation MDIS: %f" % mace)
    print("Average use time:  {:.2f} ms. All  use time: {:.3f}s".format(np.mean(np.array(timeall[1:-1]))*1000, np.sum(np.array(timeall))))
    torch.cuda.empty_cache()
    model.train()
    return {'val_mace': mace}