import os
import numpy as np
import torch
import torchvision
import geopandas as gpd
import tqdm
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap
import pickle
from process_osm.geo_opensfm import TopocentricConverter
from shapely.geometry import MultiLineString, Point
from shapely import affinity
from modules.utils.utils import get_homograpy
def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):  # 数据增强
    # adjust image
    img = img.resize(resize_dims)  # 图像缩放
    img = img.crop(crop)  # 图像裁剪
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)  # 左右翻转
    img = img.rotate(rotate)  # 旋转

    # post-homography transformation

    # 数据增强后的图像上的某一点的坐标需要对应回增强前的坐标
    post_rot *= resize  # [[0.22,0],[0,0.22]]
    post_tran -= torch.Tensor(crop[:2])  # [0,-48]
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)  # 得到数据增强时旋转操作的旋转矩阵
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2  # 裁剪保留部分图像的中心坐标 (176, 64)
    b = A.matmul(-b) + b  # 0
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran

class NormalizeInverse(torchvision.transforms.Normalize):
#  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))

def gen_dx_bx(xbound, ybound, zbound):  # 划分网格
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])  # dx=[0.5,0.5,20] 分别为x, y, z三个方向上的网格间距
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])  # bx=[-49.75,-49.75,0]  分别为x, y, z三个方向上第一个格子的坐标
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])  # nx=[200,200,1]  分别为x, y, z三个方向上格子的数量

    return dx, bx, nx

def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):  # 根据旋转角度得到旋转矩阵
    return torch.Tensor([
        [np.cos(h), -np.sin(h)],
        [np.sin(h), np.cos(h)],
    ])

def preprocess_map(map_names:list, preprocess_folder:str,osm_folder:str):
    if not os.path.exists(preprocess_folder):
        os.makedirs(preprocess_folder)

    ##四个城市的原点坐标
    map_origin = {'boston-seaport':           (42.336849169438615, -71.05785369873047, 0.),
                  'singapore-onenorth':       (1.2882100868743724, 103.78475189208984, 0.),
                  'singapore-hollandvillage': (1.2993652317780957, 103.78217697143555, 0.),
                  'singapore-queenstown':     (1.2782562240223188, 103.76741409301758, 0.)}
    map_drift  = {'boston-seaport':           (0.00002,-0.00003),
                  'singapore-onenorth':       (0.00012,0),
                  'singapore-hollandvillage': (0.00011,0),
                  'singapore-queenstown':     (0.00012,0)}
    
    options = [
            'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', # road
            'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'# road link
            'living_street',  'road',  # Special road  'service'
        ]
    for map_name in map_names:
        building_shp_path = os.path.join(osm_folder, map_name, 'buildings.shp')
        road_shp_path = os.path.join(osm_folder, map_name, 'roads.shp')
        cache_file1 = os.path.join(preprocess_folder, f'buildings_{map_name}.pkl')
        cache_file2 = os.path.join(preprocess_folder, f'roads_{map_name}.pkl')

        converter = TopocentricConverter(*map_origin[map_name])
        building_gdf = gpd.read_file(building_shp_path)
        road_gdf = gpd.read_file(road_shp_path)
        road_gdf = road_gdf[road_gdf['type'].isin(options)]
        lines = []
        polys = []
        for _, row in building_gdf.iterrows():
            geom = row.geometry
            if geom.is_empty:
                continue
            if geom.geom_type == 'Polygon':
                poly_list = [geom]
            elif geom.geom_type == 'MultiPolygon':
                poly_list = list(geom.geoms)
            else:
                continue
            for poly in poly_list:
                exterior = []
                for lonlat in poly.exterior.coords:
                    lon, lat = lonlat
                    topo = converter.to_topocentric(lat + map_drift[map_name][0], lon + map_drift[map_name][1], 0.0)
                    exterior.append((topo[0], topo[1]))
                interiors = []
                for interior in poly.interiors:
                    ring = []
                    for lonlat in interior.coords:
                        lon, lat = lonlat
                        topo = converter.to_topocentric(lat + map_drift[map_name][0], lon + map_drift[map_name][1], 0.0)
                        ring.append((topo[0], topo[1]))
                    interiors.append(ring)
                polys.append((np.array(exterior, dtype=np.float32), [np.array(r, dtype=np.float32) for r in interiors]))
        try:
            with open(cache_file1, 'wb') as f:
                pickle.dump(polys, f)
        except Exception:
            pass
        
        for _, row in road_gdf.iterrows():
            tmp_sd_data = list(row.geometry.coords)
            tmp_sd_data_topo = [converter.to_topocentric(lonlat[1] + map_drift[map_name][0], lonlat[0] + map_drift[map_name][1], 0.)[:2] for lonlat in tmp_sd_data]
            lines.append(np.array(tmp_sd_data_topo, dtype=np.float32))
        lines = MultiLineString(lines)
        
        try:
            with open(cache_file2, 'wb') as f:
                pickle.dump(lines, f)
        except Exception:
            pass


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage",
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = [(pred & tgt)[:, i:i+1, :, :].sum().float().item() for i in range(preds.shape[1])]
        union = [(pred | tgt)[:, i:i+1, :, :].sum().float().item() for i in range(preds.shape[1])]
        iou_list = [intersect[i] / union[i] if union[i] > 0 else 1 for i in range(preds.shape[1])]
    return intersect, union, iou_list


def get_val_info(model, valloader, loss_fn, device, outC, use_tqdm=True):
    model.eval()
    total_loss = 0.0
    total_loss_bev = 0.0
    total_loss_sdmap = 0.0
    total_intersect_bev = [0]*outC
    total_intersect_sdmap = [0]*outC
    total_union_bev = [0]*outC
    total_union_sdmap = [0]*outC
    iou_list_bev = []
    iou_list_sdmap = []
    print('running eval...')
    loader = tqdm.tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binmap, sdmap_input, sdmap_mask = batch
            preds_bev, preds_sdmap = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device), sdmap_input.to(device))
            binmap = binmap.to(device)
            sdmap_mask = sdmap_mask.to(device)
            
            # loss
            total_loss_bev += loss_fn(preds_bev, binmap).item() * preds_bev.shape[0]
            total_loss_sdmap += loss_fn(preds_sdmap, sdmap_mask).item() * preds_sdmap.shape[0]
            total_loss += (loss_fn(preds_bev, binmap).item() * preds_bev.shape[0] + loss_fn(preds_sdmap, sdmap_mask).item() * preds_sdmap.shape[0])

            # iou
            intersect_bev, union_bev, _ = get_batch_iou(preds_bev, binmap)
            intersect_sdmap, union_sdmap, _ = get_batch_iou(preds_sdmap, sdmap_mask)
            for i in range(preds_bev.shape[1]):
                total_intersect_bev[i] += intersect_bev[i]
                total_union_bev[i] += union_bev[i]
                total_intersect_sdmap[i] += intersect_sdmap[i]
                total_union_sdmap[i] += union_sdmap[i]

    for i in range(preds_bev.shape[1]):
        iou_list_bev.append(total_intersect_bev[i] / total_union_bev[i])
    for i in range(preds_sdmap.shape[1]):
        iou_list_sdmap.append(total_intersect_sdmap[i] / total_union_sdmap[i])
    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'loss_bev': total_loss_bev / len(valloader.dataset),
            'loss_sdmap': total_loss_sdmap / len(valloader.dataset),
            'iou_bev': iou_list_bev,
            'iou_sdmap': iou_list_sdmap,
            }

def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys

def pix2distance(origin, target, translation, rotation, boundary=[-64,-64], ppm=0.5):
    origin_xy = origin[:, [1, 0]].to(origin.device)
    b = torch.as_tensor(boundary, dtype=torch.float32, device=origin.device)
    origin_m = origin_xy * ppm
    origin_local_m = torch.stack([
        origin_m[:, 0] + b[0],
        origin_m[:, 1] + b[1]
    ], dim=1).to(origin.device)

    cos_t = torch.cos(rotation).squeeze(1)  # [B]
    sin_t = torch.sin(rotation).squeeze(1)  # [B]

    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=1),  # row0
        torch.stack([sin_t,  cos_t], dim=1)   # row1
    ], dim=1).to(origin.device)  # [B,2,2]

    origin_local_unsq = origin_local_m.unsqueeze(-1)  # [B,2,1]
    origin_global = torch.bmm(R, origin_local_unsq).squeeze(-1) + translation  # [B,2]

    dist = torch.norm(origin_global - target.to(origin.device), dim=1)

    return dist

def localize_loss(four_preds, uv_trans, ori_gt, refrence_center, sz, egopose, trans_global, rot_global, w3=10, gamma=0.85):
    n_predictions = len(four_preds) if type(four_preds) == list else 1
    sz = [uv_trans.shape[0]] + [1] + sz
    y = uv_trans
    total_loss = 0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions-i-1)
        H = get_homograpy(four_preds[i],sz)
        points = torch.cat((refrence_center, torch.ones((sz[0], 1, refrence_center.shape[-1])).to(uv_trans.device)), dim=1).to(uv_trans.device)
        x = H.bmm(points)
        x = x / x[:, 2, :].unsqueeze(1)
        x[:,:2,:] = x[:,:2,:]
        dx = x[:, 0, 1] - x[:, 0, 0]
        dy = x[:, 1, 0] - x[:, 1, 1]
        ori = torch.rad2deg(torch.atan2(dx, dy))
        ori_error = (ori - ori_gt).abs()
        ori_loss = ori_error.nanmean()

        x = x[:,:2, 0]
        i_loss = torch.nanmean((x-y)**2)
        i_loss += ori_loss*w3
        total_loss += i_weight * i_loss

    err_meters = pix2distance(x, egopose, trans_global, rot_global, ppm=0.5)


    metrics = {
        'edist': err_meters.nanmean().item(),
        'eorien': ori_loss.item(),
        '1': (err_meters < 1).float().mean().item(),
        '2': (err_meters < 2).float().mean().item(),
        '5': (err_meters < 5).float().mean().item(),
        '10': (err_meters < 10).float().mean().item(),
        'o1': (ori_error < 1).float().mean().item(),
        'o2': (ori_error < 2).float().mean().item(),
        'o5': (ori_error < 5).float().mean().item(),
        'o10': (ori_error < 10).float().mean().item(),
    }

    return total_loss, metrics

SUM_FREQ = 100
class Logger_train:
    # def __init__(self, args, optimizer, need_steps, print_log = True):
    def __init__(self, args, scheduler, optimizer, need_steps, print_log = True):
        self.args = args
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.need_steps = need_steps
        self.print_log = print_log

        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")  

        self.file_name = "./watch/{}/{}_{}.log".format(args.dataset, args.name, timestamp)

        os.makedirs("./watch/", exist_ok=True)
        os.makedirs("./watch/{}".format(args.dataset), exist_ok=True)

        print(f"Log will be saved to:{self.file_name}")

        with open(self.file_name, 'a') as file:
            file.write(str(args)  + '\n')
            file.write(self.format_args())

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.args.SUM_FREQ for k in sorted(self.running_loss.keys())] # ['1px', '3px', 'epe', 'loss']
        training_str = "[{:6d},{:4d}/{:4d},{:10.7f}] ".format(self.total_steps+1, self.total_steps%self.need_steps+1, self.need_steps, self.optimizer.state_dict()['param_groups'][0]['lr'] ) # optimizer.state_dict()['param_groups'][0]['lr']  self.scheduler.get_last_lr()[0]
        metrics_str = ("{:6.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        if self.print_log : print(training_str + metrics_str)
        with open(self.file_name, 'a') as file:
            file.write(training_str + metrics_str + '\n')

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.args.SUM_FREQ == self.args.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def format_args(self):
        args = self.args
        params = {
            'name': args.name,
            'batch_size': args.batch_size,
            'dataset': args.dataset,
            'start_step': args.start_step,
            'num_steps': args.num_steps,
            'learning_rate': args.lr,
            'ckpt_path': args.ckpt_path,         
        }
        formatted_params = "\n".join(f"{key}: {value}" for key, value in params.items())
        return formatted_params+'\n'
    
class Logger:
    def __init__(self, args):
        self.args = args
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.nanmean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}] ".format(self.total_steps)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        # print the training status
        print(training_str + metrics_str)

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []
            self.running_loss_dict[key].append(metrics[key])
            if np.isnan(metrics[key]):
                print('\033[1;91m'+"There is a nan at {}!\033[0m".format(len(self.running_loss_dict[key])))
