import torch
import torch.nn.functional as F
from torch import nn
from .Lift_splat import compile_model
from .map_encoder import MapEncoder
from .homo_estimator import IHN
# from .pose_regressor import decoder
# from .trans_fusion import TR_Fusion
import time
class Locator(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(Locator, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.outC = outC

        self.bev_extractor = compile_model(
            grid_conf=grid_conf,
            data_aug_conf=data_aug_conf,
            outC=self.outC
        )
        self.map_encoder = MapEncoder(embedding_dim=16, out_channels=self.outC, bilinear=True)
        self.homo_estimator = IHN()
        # self.decoder = decoder()
        # self.decoder = TR_Fusion()

    def forward(self, rgb_imgs, rots, trans, intrins, post_rots, post_trans, sdmap_input):
        bev_features, bev_masks = self.bev_extractor(rgb_imgs, rots, trans, intrins, post_rots, post_trans)
        map_features, sdmap_masks = self.map_encoder(sdmap_input)

        # bev_features_padded = F.pad(bev_features, pad=(64, 64, 64, 64), mode='constant', value=0)
        # flow_predictions= self.decoder(bev_features, map_features)
        flow_predictions, fmap1, fmap2 = self.homo_estimator(bev_features, map_features)

        return bev_masks, sdmap_masks, flow_predictions #, fmap1, fmap2