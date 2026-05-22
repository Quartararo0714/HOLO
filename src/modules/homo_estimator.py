import torch
import torch.nn as nn
from .update import GMA, GMA_update
from .corr import CorrBlock
from .homo_extractor import BasicEncoder, RHWF_Encoder
from .utils.utils import *
from .utils.torch_geometry import get_perspective_transform
# from .ATT.attention_layer import Correlation, FocusFormer_Attention
# import torchgeometry as tgm


autocast = torch.cuda.amp.autocast

class IHN(nn.Module):
    def __init__(self, attn_depth=2):
        super().__init__()
        self.fnet = BasicEncoder(output_dim=256,norm_fn='instance')
        in_dim = 164
        sz = 64
        self.update_block_4 = GMA(sz, in_dim)

        #使用特征融合
        # self.transformer_0 = FocusFormer_Attention(256, 1, 96, 96)
        # self.kernel_list = [0] * attn_depth
        # self.pad_list = [0] * attn_depth
        # self.attn_depth = attn_depth
        # self.transformer_blocks = nn.ModuleList([
        #     FocusFormer_Attention(256, 1, 96, 96)
        #     for _ in range(attn_depth)
        # ])

    def get_flow_now_k(self, four_point, k = 4):
        N,_,h,w = self.sz
        h, w = h//k, w//k
        four_point = four_point / k # four_point is at original size， coordinate is at feature map size
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
        four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(N, 1, 1, 1)
        four_point_new = torch.autograd.Variable(four_point_org) + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        # H = tgm.get_perspective_transform(four_point_org, four_point_new)
        H = get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, w-1, steps=w), torch.linspace(0,h-1, steps=h),indexing='ij')
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, w * h))),
                           dim=0).unsqueeze(0).repeat(N, 1, 1).to(four_point.device)
        points_new = H.bmm(points) # (N,3,3) (N,3,w*h)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(N, w, h).unsqueeze(1),
                          points_new[:, 1, :].reshape(N, w, h).unsqueeze(1)), dim=1)
        return flow, H

    def initialize_flow_k(self, img, k= 4):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//k, W//k).to(img.device) # [batch,2, H, W]
        coords1 = coords_grid(N, H//k, W//k).to(img.device)

        return coords0, coords1 # [x,y]

    def forward(self, image1, image2, iters_lev0 = 6, Feature_fusion=False):    
        self.sz = image1.shape # [N, 3, 128, 128]
        
        ## 1.Get features by backbone
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)

        sz = fmap1.shape

        fmap1 = fmap1.float()
        fmap2 = fmap2.float() 

        # 2. Calculate Correlation Matrix      
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4)
        coords0, coords1 = self.initialize_flow_k(image1, k=self.sz[-1]//sz[-1])         
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)

        # 3. Recurrent Homography Estimation
        flow_predictions =[] ## for train 
        for itr in range(iters_lev0):
            corr = corr_fn(coords1) # batch,channel,H,W  correlation
            flow = coords1 - coords0 # [batch,2, H, W] mean x, y
            delta_four_point = self.update_block_4(corr, flow) # input shape: [b,2+channel,H,W] , output shape [b,2,2,2]
            four_point_disp =  four_point_disp + delta_four_point
            coords1, _ = self.get_flow_now_k(four_point_disp, k=self.sz[-1]//sz[-1])
            flow_predictions.append(four_point_disp) ## for train 
        
        return flow_predictions, fmap1, fmap2
