import os
import json
import math
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions, get_wl_ray_directions
from utils.misc import get_rank


import imageio 
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,np.sin(th),0],
    [0,1,0,0],
    [-np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_gamma = lambda ga : torch.Tensor([
    [np.cos(ga),-np.sin(ga), 0, 0],
    [np.sin(ga), np.cos(ga), 0, 0],
    [0,0,1,0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, gamma, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_gamma(gamma/180.*np.pi) @ c2w
    return c2w[:3, :4]

def create_transformation_matrix(rotation, translation, scale):
    transformation_matrix = np.identity(4)

    rotation_matrix = np.array(rotation).reshape((3, 3))
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, :3] *= scale
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

def read_overall_transform(xml_file_path):
    # read camera XML
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    transform = root.find('.//component/transform')

    rotation_text = transform.find('rotation').text
    rotation_values = list(map(float, rotation_text.split()))

    translation_text = transform.find('translation').text
    translation_values = list(map(float, translation_text.split()))

    scale_text = transform.find('scale').text
    scale_value = float(scale_text)
    
    overall_transform = create_transformation_matrix(rotation_values, translation_values, scale_value)

    return overall_transform

def read_camera_matrices(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    transform_matrices = []
    cameras = root.findall('.//camera')
    
    for camera in cameras:
        transform_text = camera.find('transform').text
        transform_values = list(map(float, transform_text.split()))

        transform_matrix = np.array(transform_values).reshape((4, 4))
        
        transform_matrices.append(transform_matrix)
    
    focal = root.find('.//calibration/f')
    
    return transform_matrices, float(focal.text)

def read_point_cloud_transform(transform_file_path):
    with open(transform_file_path, 'r') as file:
        lines = file.readlines()[3:7] 

    transform_matrix = np.array([list(map(float, line.split())) for line in lines])

    return transform_matrix

class WLIOMDatasetBase():
    def print_status(self):
        print('Status:', self.data_status)
        print('Mix Status:', self.data_mix_status)
        print('Focal:', self.focal)
        print('depth_data_num:', self.depth_data_num)
        print('all_c2w', self.all_c2w.shape)
        print('all_images', self.all_images.shape)
        print('all_fg_masks', self.all_fg_masks.shape)
        print('all_depth_imgs', self.all_depth_imgs.shape)
        print('all_gt_depth_imgs', self.all_gt_depth_imgs.shape)
        print('all_depth_masks', self.all_depth_masks.shape)
    
    def set_data_status(self, status):
        if self.data_status == status:
            return
        if status == 'WL':
            self.img_wh = self.dp_img_wh
            self.focal = self.depth_focal
            self.w, self.h = self.dp_w, self.dp_h
            self.all_images   = self.all_wl_images
            self.all_fg_masks = self.all_wl_fg_masks
            self.directions = self.depth_directions
            self.data_status = 'WL'
        elif status == 'OP':
            self.img_wh = self.op_img_wh
            self.focal = self.optical_focal
            self.w, self.h = self.op_w, self.op_h
            self.all_images   = self.all_op_images
            self.all_fg_masks = self.all_op_fg_masks
            self.directions = self.optical_directions
            self.data_status = 'OP'
    
    def switch_data_status(self):
        if self.data_status == 'WL' and self.data_mix_status != 'WL_ONLY':
            self.set_data_status('OP')
        elif self.data_status == 'OP' and self.data_mix_status != 'OP_ONLY':
            self.set_data_status('WL')

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = False
        self.apply_mask = False
        
        self.op_w, self.op_h = self.config.optical_img_wh
        self.op_img_wh = (self.op_w, self.op_h)
        self.dp_w, self.dp_h = self.config.depth_img_wh
        self.dp_img_wh = (self.dp_w, self.dp_h)

        self.w, self.h = self.config.depth_img_wh
        self.img_wh = self.dp_img_wh

        self.data_mix_status = self.config.data_mix_status
        self.data_status = 'WL'
        self.depth_data_num = 5
        self.depth_focal = self.config.depth_focal
        self.focal = self.depth_focal

        wl_root_dir = os.path.join(self.config.root_dir, 'wl')
        op_root_dir = os.path.join(self.config.root_dir, 'op')
        camera_matrices, camera_focal = read_camera_matrices(os.path.join(op_root_dir, 'camera.xml'))
        self.optical_focal = camera_focal

        self.optical_directions = \
            get_ray_directions(self.op_w, self.op_h, self.optical_focal, self.optical_focal, self.op_w//2, self.op_h//2).to(self.rank) 

        self.depth_directions = \
            get_wl_ray_directions(self.dp_w, self.dp_h, self.depth_focal).to(self.rank) 
        
        self.directions = self.depth_directions

        self.near, self.far = 2.0, 16.0

        self.all_c2w, self.all_images, self.all_gt_depth_imgs, self.all_depth_imgs, self.all_fg_masks, self.all_depth_masks = [], [], [], [], [], []
        
        self.all_wl_images, self.all_wl_fg_masks = [], []
        
        for i in range(self.depth_data_num):
            img = np.ones((self.dp_w, self.dp_h, 3), dtype=np.float32)
            
            meta_data = np.load(os.path.join(wl_root_dir, '{:04d}.npz'.format(i)))
                        
            self.all_wl_images.append(torch.from_numpy(img[...,:3]))
            
            depth_img = meta_data['depth_map']
            depth_img = torch.from_numpy(depth_img)
            self.all_depth_imgs.append(depth_img)

            gt_depth_mask = np.isnan(depth_img)       
            gt_depth_img = depth_img  
            self.all_gt_depth_imgs.append(gt_depth_img)
            self.all_depth_masks.append(gt_depth_mask)
            
            pose = meta_data['extrinsic_mat']
            # 3x4 -> 4x4
            pose = np.concatenate([pose, np.array([[0,0,0,1]])], 0)
            pose = np.linalg.inv(pose) 
            pose = torch.from_numpy(pose[:3, :4])
            self.all_c2w.append(pose)

            fg_mask = torch.ones_like(depth_img)
            self.all_wl_fg_masks.append(fg_mask)
        

        overall_transform = read_overall_transform(os.path.join(op_root_dir, 'camera.xml'))
        point_cloud_transform = read_point_cloud_transform(os.path.join(op_root_dir, 'align.aln'))
        
        self.all_op_images, self.all_op_fg_masks = [], []
        
        for i, pose in enumerate(camera_matrices):
            transformed_pose = point_cloud_transform @ overall_transform @ pose
            c2w = torch.from_numpy(transformed_pose[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(op_root_dir, '{:04d}.png'.format(i))
            img = Image.open(img_path)
            img = img.resize(self.op_img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            fg_mask = torch.ones_like(img[..., -1])
            self.all_op_fg_masks.append(fg_mask)
            self.all_op_images.append(img[...,:3])
        
        self.all_c2w, self.all_depth_imgs, self.all_gt_depth_imgs, self.all_depth_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_depth_imgs, dim=0).float().to(self.rank), \
            torch.stack(self.all_gt_depth_imgs, dim=0).float().to(self.rank), \
            torch.stack(self.all_depth_masks, dim=0).to(self.rank)

        self.all_wl_images, self.all_wl_fg_masks, self.all_op_images, self.all_op_fg_masks = \
            torch.stack(self.all_wl_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_wl_fg_masks, dim=0).float().to(self.rank), \
            torch.stack(self.all_op_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_op_fg_masks, dim=0).to(self.rank)

        self.all_images   = self.all_wl_images
        self.all_fg_masks = self.all_wl_fg_masks

        if self.data_mix_status == 'OP_ONLY':
            self.set_data_status('OP')
        else:
            self.set_data_status('WL')

        self.print_status()       

class WLIOMDataset(Dataset, WLIOMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_c2w)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class WLIOMIterableDataset(IterableDataset, WLIOMDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('WLIOM')
class WLIOMDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = WLIOMIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = WLIOMDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = WLIOMDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = WLIOMDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1) 