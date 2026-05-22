import torch
import torchvision
import os
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.utils.data_classes import Box
from PIL import Image
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, Point
from glob import glob
from tools import img_transform, gen_dx_bx, normalize_img, get_rot, preprocess_map, get_nusc_maps
import pickle
import time
mpl.use('Agg')

from modules.Locator import Locator

class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, sdmap_conf,
                 map_preprocess, nusc_maps, osm_path, is_sdmap):
        self.nusc = nusc
        self.is_train = is_train  # 是否为训练集
        self.data_aug_conf = data_aug_conf  # 数据增强配置
        self.grid_conf = grid_conf  # 网格配置
        self.sdmap_conf = sdmap_conf  # SDMap配置
        self.map_preprocess = map_preprocess  # 是否预处理地图
        self.nusc_maps = nusc_maps  # nusc地图，以字典储存
        self.osm_path = osm_path  # OSM地图的路径
        self.sdmap_dir = os.path.join(os.path.dirname(self.nusc.dataroot), 'sdmaps')
        
        self.is_sdmap = is_sdmap  # True表示使用SDMap，False表示使用NuScenesMap
        self.scenes = self.get_scenes()  # 得到scene名字的列表list: [scene-0061, scene-0103,...]
        self.ixes = self.prepro()  # 得到属于self.scenes的所有sample

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])

        self.dx_grid, self.bx_grid, self.nx_grid = dx.numpy(), bx.numpy(), nx.numpy()  # 转化成numpy

        dx, bx, nx = gen_dx_bx(sdmap_conf['xbound'], sdmap_conf['ybound'], sdmap_conf['zbound'])
        self.dx_sd, self.bx_sd, self.nx_sd = dx.numpy(), bx.numpy(), nx.numpy()

        self.stretch_sd = -self.bx_sd[0] + self.dx_sd[0] / 2.0
        self.stretch_grid = -self.bx_grid[0] + self.dx_grid[0] / 2.0 

        self.fix_nuscenes_formatting()
        
        if self.map_preprocess:
            preprocess_map(self.nusc_maps.keys(), self.building_dir, self.osm_path)  # 预处理地图
        
        self.preloaded_building_maps = {}
        self.preloaded_road_maps = {}
        for map_name in self.nusc_maps.keys():
            building_path = os.path.join(self.sdmap_dir, f'buildings_{map_name}.pkl')
            with open(building_path, 'rb') as f:
                self.preloaded_building_maps[map_name] = pickle.load(f)
            road_path = os.path.join(self.sdmap_dir, f'roads_{map_name}.pkl')
            with open(road_path, 'rb') as f:
                self.preloaded_road_maps[map_name] = pickle.load(f)

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]  # 根据 self.nusc.version 场景分为训练集和验证集，得到的是场景名字的list: [scene-0061,
        # scene-0103,...]

        return scenes

    def prepro(self):  # 将self.scenes中的所有sample取出并依照 scene_token和timestamp排序
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']  # (900,1600)
        fH, fW = self.data_aug_conf['final_dim']  # (128, 352)，表示变换之后最终的图像大小
        if self.is_train:  # 训练集数据增强
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:  # 测试集数据增强
            resize = max(fH / H, fW / W)  # 缩小的倍数取二者较大值: 0.22
            resize_dims = (int(W * resize), int(H * resize))  # 保证H和W以相同的倍数缩放，resize_dims=(352, 198)
            newW, newH = resize_dims  # (352,198)
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH  # 48
            crop_w = int(max(0, newW - fW) / 2)  # 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # (0, 48, 352, 176)，对应裁剪的左上角和右下角的坐标
            flip = False  # 不翻转
            rotate = 0  # 不旋转
        return resize, resize_dims, crop, flip, rotate
    
    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])  # 根据相机通道选择对应的sample_data
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])  # 图片的路径
            img = Image.open(imgname)  # 读取图像 1600 x 900
            post_rot = torch.eye(2)  # 增强前后像素点坐标的旋转对应关系
            post_tran = torch.zeros(2)  # 增强前后像素点坐标的平移关系

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])  # 相机record
            intrin = torch.Tensor(sens['camera_intrinsic'])  # 相机内参
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  # 相机坐标系相对于ego坐标系的旋转矩阵
            tran = torch.Tensor(sens['translation'])  # 相机坐标系相对于ego坐标系的平移矩阵

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()  # 获取数据增强的参数
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )  # 进行数据增强: resize->crop,并得到增强前后像素点坐标的对应关系

            # for convenience, make augmentation matrices 3x3

            # 写成3维矩阵的形式
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))  # 标准化: ToTensor, Normalize 3,128,352
            intrins.append(intrin)  # 3,3
            rots.append(rot)  # 3,3
            trans.append(tran)  # 3,
            post_rots.append(post_rot)  # 3,3
            post_trans.append(post_tran)  # 3,

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))  # 使用torch.stack组装到一起
    
    def get_binmap(self, rec, translation_range=30.0, rotation_range=30.0):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = egopose['translation']
        rot = Quaternion(egopose['rotation'])
        center = np.array(trans[:2])
        yaw_rad = rot.yaw_pitch_roll[0]     # 原始朝向（弧度）
        yaw_deg = yaw_rad / np.pi * 180 
        rot_matrix = get_rot(yaw_rad)

        random_translation = (np.random.uniform(-1, 1, size=2) * translation_range)
        random_rotation = np.random.uniform(-1, 1) * rotation_range
        sdmap_angle = (yaw_deg + random_rotation) / 180 * np.pi
        sdmap_center = center + random_translation

        map_loc = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']

        if self.is_sdmap:
            roads_topo_mask, buildings_topo_mask, bevCenter_uv = self.get_sdmap_geom(center=center, patch_angle=yaw_deg, bev=True, location=map_loc)
            sdmap_bev_mask = self.rast_sdmap(road_lines=roads_topo_mask, building_polys=buildings_topo_mask, bev=True)
            roads_topo_input, buildings_topo_input, sd_delta = self.get_sdmap_geom(center=center, patch_angle=yaw_deg, noise_trans=random_translation, 
                                                                                noise_rot=random_rotation, noise=True, bev=False, location=map_loc)
            sdmap_mask, sdmap_input = self.rast_sdmap(road_lines=roads_topo_input, building_polys=buildings_topo_input, bev=False)
            
        else:
            nmap = self.nusc_maps[map_loc]
            hdmap_bev_mask, bevCenter_uv = self.get_hdmap_mask(nmap, center=center, patch_angle=yaw_deg, bev=True, location=map_loc)       
            hdmap_mask, sd_delta = self.get_hdmap_mask(nmap, center=center, patch_angle=yaw_deg, noise_trans=random_translation, 
                                        noise_rot=random_rotation, noise=True, bev=False, location=map_loc)
            roads_topo_input, buildings_topo_input, _ = self.get_sdmap_geom(center=center, patch_angle=yaw_deg, noise_trans=random_translation, 
                                                                                noise_rot=random_rotation, noise=True, bev=False, location=map_loc)
            _, sdmap_input = self.rast_sdmap(road_lines=roads_topo_input, building_polys=buildings_topo_input, bev=False)

        bevOrient = [bevCenter_uv[0], bevCenter_uv[1], bevCenter_uv[0], bevCenter_uv[1]-bevCenter_uv[1]/4]
        bevOrient = torch.tensor(bevOrient).view(2,2).permute(1,0).float()

        # H, W = self.nx_sd[:2]
        # rgb = np.zeros((H, W, 3), dtype=np.uint8)
        # rgb[hdmap_mask[1] == 1] = (255, 0, 0)  # 红色：building
        # rgb[hdmap_mask[0] == 1] = (0, 255, 0)  # 绿色：drivable
        # plt.figure(figsize=(8, 8))
        # plt.imshow(rgb)
        # plt.title("Semantic Segmentation (Green: Drivable, Red: Building)")
        # plt.plot(sd_delta[0], sd_delta[1], 'bo', markersize=5)  # 在中心点标记
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('hdmap_masks.png', dpi=300)
        # rgb = np.zeros((H, W, 3), dtype=np.uint8)
        # rgb[hdmap_bev_mask[1] == 1] = (255, 0, 0)  # 红色：building
        # rgb[hdmap_bev_mask[0] == 1] = (0, 255, 0)  # 绿色：drivable
        # plt.figure(figsize=(8, 8))
        # plt.imshow(rgb)
        # plt.title("Semantic Segmentation (Green: Drivable, Red: Building)")
        # plt.plot(bevOrient[0,0], bevOrient[0,1], 'bo', markersize=5)  # 在中心点标记
        # plt.plot(bevOrient[1,0], bevOrient[1,1], 'mo', markersize=5)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('hdmap_bev_masks.png', dpi=300)
        if self.is_sdmap:
            return torch.Tensor(sdmap_bev_mask), torch.tensor(sdmap_input).long(), torch.Tensor(sdmap_mask), \
                torch.Tensor(sd_delta), torch.tensor(random_rotation), torch.Tensor(center), bevOrient, \
                torch.Tensor([sdmap_angle]), torch.Tensor(sdmap_center)
        else:
            return torch.Tensor(hdmap_bev_mask), torch.tensor(sdmap_input).long(), torch.Tensor(hdmap_mask), \
                torch.Tensor(sd_delta), torch.tensor(random_rotation), torch.Tensor(center), bevOrient, \
                torch.Tensor([sdmap_angle]), torch.Tensor(sdmap_center)  

    def choose_cams(self):  # 随机选择摄像机通道
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams
    
    def get_hdmap_mask(self, nusc_map, center, patch_angle, noise_trans=[0,0], noise_rot=0, noise=False, bev=False, location=None):
        hdmap_buildings = self.preloaded_building_maps[location]
        if bev:
            H,W = self.nx_grid[:2]
            stretch = self.stretch_grid
            _bx = self.bx_grid[:2]
            _dx = self.dx_grid[:2]
        else:
            H,W = self.nx_sd[:2]
            stretch = self.stretch_sd
            _bx = self.bx_sd[:2]
            _dx = self.dx_sd[:2]

        if noise:
            center_new = center + np.array(noise_trans)
            patch_angle = patch_angle + noise_rot
            patch_x = center_new[0]
            patch_y = center_new[1]
            pt = Point(center[0], center[1])
            pt_rot = affinity.rotate(pt, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            pt_trans = affinity.affine_transform(pt_rot, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            coords = np.array(pt_trans.coords[0])
            pix = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
            pix = (pix[1], pix[0])  

        else:
            patch_x = center[0]
            patch_y = center[1]
            pt = Point(center[0], center[1])
            pt_rot = affinity.rotate(pt, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            pt_trans = affinity.affine_transform(pt_rot, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            coords = np.array(pt_trans.coords[0])
            pix = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
            pix = (pix[1], pix[0])  

        layer_names = ['road_segment', 'lane']
        patch_box = (patch_x, patch_y, 2*stretch, 2*stretch)
        road_mask_raw = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, (H, W))
        road_mask = np.logical_or(road_mask_raw[0], road_mask_raw[1]).astype(np.uint8)
        road_mask = np.expand_dims(road_mask, axis=0).transpose(0, 2, 1)  # (1, H, W)

        building_mask = np.zeros((1, H, W), dtype=np.uint8)
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        building_polys_list = []
        for exterior, _ in hdmap_buildings:
            poly = Polygon(exterior)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid:
                continue  # 跳过无法修复的多边形
            clipped_poly = poly.intersection(patch)
            if not clipped_poly.is_empty:
                geoms = (clipped_poly.geoms if isinstance(clipped_poly, MultiPolygon) else [clipped_poly])
                for g in geoms:
                    g = affinity.rotate(g, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    g = affinity.affine_transform(g, [1,0,0,1, -patch_x, -patch_y])
                    building_polys_list.append(g)
        # patch_map = nusc_map.get_records_in_patch(patch, layer_names=layer_names, mode='intersect')
        building_pts = []
        for poly in building_polys_list:
            coords = np.array(poly.exterior.coords)
            pts = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
            pts = pts[:, [1, 0]].reshape((-1, 1, 2))
            building_pts.append(pts)

        cv2.fillPoly(building_mask[0], building_pts, color=1)
        map_mask = np.concatenate((road_mask, building_mask), axis=0)  # (2, H, W)

        return map_mask, pix


    def get_sdmap_geom(self, center, patch_angle, noise_trans=[0,0], noise_rot=0, noise=False, bev=False, location=None):
        sdmap_roads = self.preloaded_road_maps[location]
        sdmap_buildings = self.preloaded_building_maps[location]
        if bev:
            stretch = self.stretch_grid
            _bx = self.bx_grid[:2]
            _dx = self.dx_grid[:2]
        else:
            stretch = self.stretch_sd
            _bx = self.bx_sd[:2]
            _dx = self.dx_sd[:2]

        if noise:
            center_new = center + np.array(noise_trans)
            patch_angle = patch_angle + noise_rot
            patch_x = center_new[0]
            patch_y = center_new[1]
            pt = Point(center[0], center[1])
            pt_rot = affinity.rotate(pt, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            pt_trans = affinity.affine_transform(pt_rot, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            coords = np.array(pt_trans.coords[0])
            pix = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
            pix = (pix[1], pix[0])  

        else:
            patch_x = center[0]
            patch_y = center[1]
            pt = Point(center[0], center[1])
            pt_rot = affinity.rotate(pt, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            pt_trans = affinity.affine_transform(pt_rot, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            coords = np.array(pt_trans.coords[0])
            pix = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
            pix = (pix[1], pix[0])  

        patch_box = (patch_x, patch_y, 2*stretch, 2*stretch)

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)

        road_lines_list = []
        for road_lines in sdmap_roads.geoms:
            if not road_lines.is_empty:
                road_lines = road_lines.intersection(patch)
                if not road_lines.is_empty:
                    road_lines = affinity.rotate(road_lines, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    road_lines = affinity.affine_transform(road_lines,
                                                        [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    road_lines_list.append(road_lines)

        building_polys_list = []
        for exterior, _ in sdmap_buildings:
            poly = Polygon(exterior)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid:
                continue  # 跳过无法修复的多边形
            clipped_poly = poly.intersection(patch)
            if not clipped_poly.is_empty:
                geoms = (clipped_poly.geoms if isinstance(clipped_poly, MultiPolygon) else [clipped_poly])
                for g in geoms:
                    g = affinity.rotate(g, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    g = affinity.affine_transform(g, [1,0,0,1, -patch_x, -patch_y])
                    building_polys_list.append(g)

        return road_lines_list, building_polys_list, pix

    def rast_sdmap(self, road_lines, building_polys, bev = False):
        if bev:
            H,W = self.nx_grid[:2]
            mask = np.zeros((2, H, W), dtype=np.uint8)
            _bx = self.bx_grid[:2]
            _dx = self.dx_grid[:2]
        else:
            H,W = self.nx_sd[:2]
            sdmap_input = np.zeros((2, H, W), dtype=np.uint8)
            mask = np.zeros((2, H, W), dtype=np.uint8)
            _bx = self.bx_sd[:2]
            _dx = self.dx_sd[:2]

        road_pts = []
        building_pts = []

        for line in road_lines:
            if line.geom_type == 'MultiLineString':
                for subline in line.geoms:
                    coords = np.array(subline.coords)
                    pts = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
                    if len(pts) >= 2:
                        pts = pts[:, [1, 0]].reshape((-1, 1, 2))
                        road_pts.append(pts)
            else:
                coords = np.array(line.coords)
                pts = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
                if len(pts) >= 2:
                    pts = pts[:, [1, 0]].reshape((-1, 1, 2))
                    road_pts.append(pts)

        for poly in building_polys:
            coords = np.array(poly.exterior.coords)
            pts = np.round((coords - _bx + _dx / 2.) / _dx).astype(np.int32)
            pts = pts[:, [1, 0]].reshape((-1, 1, 2))
            building_pts.append(pts)

        if bev:
            cv2.polylines(mask[0], road_pts, isClosed=False, color=1, thickness=20)
            cv2.fillPoly(mask[1], building_pts, color=1)
        else:
            cv2.polylines(mask[0], road_pts, isClosed=False, color=1, thickness=10)
            cv2.fillPoly(mask[1], building_pts, color=1)
            
            cv2.polylines(sdmap_input[0], road_pts, isClosed=False, color=1, thickness=1)
            cv2.fillPoly(sdmap_input[1], building_pts, color=1)

        return (mask, sdmap_input) if bev == False else mask
    
    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)

class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]  # 按索引取出sample

        cams = self.choose_cams()  # 对于训练集且data_aug_conf中Ncams<6的，随机选择摄像机通道，否则选择全部相机通道
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)  # 读取图像数据、相机参数和数据增强的像素坐标映射关系
        # imgs: 6,3,128,352  图像数据
        # rots: 6,3,3  相机坐标系到自车坐标系的旋转矩阵
        # trans: 6,3  相机坐标系到自车坐标系的平移向量
        # intrins: 6,3,3  相机内参
        # post_rots: 6,3,3  数据增强的像素坐标旋转映射关系
        # post_trans: 6,3  数据增强的像素坐标平移映射关系
        sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center = self.get_binmap(rec)  # 得到rec中所有box相对于车辆的box坐标的平面投影图, 1x200x200
        # binimg中在box内的位置值为1，其他位置的值为0

        return imgs, rots, trans, intrins, post_rots, post_trans, \
               sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, \
               angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center  # 返回图像数据、相机参数、数据增强的像素坐标映射关系和语义分割图

def worker_rnd_init(x):  # x是线程id
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, sdmap_conf, map_preprocess, nusc_maps, osm_path, is_sdmap, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=False)  # 加载nuScenes数据集
    parser = {
        'segmentationdata': SegmentationData,
    }[parser_name]  # 根据传入的参数选择数据解析器
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf, sdmap_conf=sdmap_conf, map_preprocess=map_preprocess, nusc_maps=nusc_maps, osm_path=osm_path, is_sdmap=is_sdmap)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf, sdmap_conf=sdmap_conf, map_preprocess=map_preprocess, nusc_maps=nusc_maps, osm_path=osm_path, is_sdmap=is_sdmap)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)  # 给每个线程设置随机种子
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
    
if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data/datasets/nuscenes', verbose=False)
    xbound=[-32.0, 32.0, 0.5]
    ybound=[-32.0, 32.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    dbound=[4.0, 27.0, 1.0]
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    xbound=[-64.0, 64.0, 0.5]
    ybound=[-64.0, 64.0, 0.5]
    zbound=[-10.0, 10.0, 20.0]
    sdmap_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
    }
    H=900
    W=1600
    resize_lim=(0.193, 0.225)
    final_dim=(128, 352)
    bot_pct_lim=(0.0, 0.22)
    rot_lim=(-5.4, 5.4)
    rand_flip=True
    ncams=5
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                    'outC': 2
                }
    map_folder='/data/datasets/nuscenes'
    nusc_maps = get_nusc_maps(map_folder)
    # trainloader, valloader = compile_data('trainval', '/data/datasets/nuscenes', nusc_maps=nusc_maps, data_aug_conf=data_aug_conf,
    #                                     grid_conf=grid_conf, sdmap_conf=sdmap_conf, map_preprocess=False, osm_path='/data/datasets/OSM',
    #                                     is_sdmap=False, bsz=4, nworkers=32, parser_name='segmentationdata')
    
    # device = torch.device('cuda:1')
    # model = Locator(grid_conf, data_aug_conf, outC=2)
    # model.to(device)
    # model.eval()
    # prev_t = time.perf_counter()
    # for batch_i, batch in enumerate(trainloader):
    #     cur_t = time.perf_counter()
    #     load_time = (cur_t - prev_t) * 1000
    #     print(f"[Batch {batch_i}] load time: {load_time:.1f} ms")
    #     imgs, rots, trans, intrins, post_rots, post_trans, sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center= batch
    #     preds_bev, preds_sdmap, flow_preds, corr_map = model(imgs.to(device),
    #                     rots.to(device),
    #                     trans.to(device),
    #                     intrins.to(device),
    #                     post_rots.to(device),
    #                     post_trans.to(device),
    #                     sdmap_input.to(device)
    #                     )
    #     prev_t = time.perf_counter()
    #     print(preds_bev.shape, preds_sdmap.shape, flow_preds.shape, corr_map.shape)
    #     if batch_i == 10:
    #         break
    
    # t = torch.full((4, 2), 100, dtype=torch.float32)
    # prev_t = time.perf_counter()
    # for batch_i, batch in enumerate(trainloader):
    #     cur_t = time.perf_counter()
    #     load_time = (cur_t - prev_t) * 1000
    #     #print(f"[Batch {batch_i}] load time: {load_time:.1f} ms")
    #     imgs, rots, trans, intrins, post_rots, post_trans, sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center= batch
    #     if batch_i % 10 == 0:
    #         print(sdmap_bev_mask.shape, sdmap_input.shape, sdmap_mask.shape, sd_delta.shape, angle_gt.shape, egopose_gt.shape, bev_center.shape, sdmap_angle.shape, sdmap_center.shape)
    #     prev_t = time.perf_counter()

    #     if(batch_i == 100):
    #         break
        
    dataset = NuscData(nusc, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf, sdmap_conf=sdmap_conf, map_preprocess=False, nusc_maps=nusc_maps, osm_path='/data/datasets/OSM', is_sdmap=False)
    # # for i in range(1):
    # start_time = time.time()
    rec = dataset.ixes[15]
    sdmap_bev_mask, sdmap_input, sdmap_mask, sd_delta, angle_gt, egopose_gt, bev_center, sdmap_angle, sdmap_center=dataset.get_binmap(rec)
    print(sdmap_bev_mask.shape, sdmap_input.shape, sdmap_mask.shape)
    # # print(sd_delta)
    # # print(angle_gt)

    # end_time = time.time()
    #print(f"Time taken for sample {i}: {end_time - start_time} seconds")
