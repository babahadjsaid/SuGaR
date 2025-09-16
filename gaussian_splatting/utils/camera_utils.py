#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
import os  # Add os import
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

WARNED = False

# PyTorch Dataset for lazy image loading
class CameraImageDataset(Dataset):
    def __init__(self, cam_infos, args, resolution_scale):
        self.cam_infos = cam_infos
        self.args = args
        self.resolution_scale = resolution_scale
        self.max_width = 1600
        self.transform = None  # Will be set per image

    def compute_resolution(self, orig_w, orig_h):
        args = self.args
        resolution_scale = self.resolution_scale
        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:
            if args.resolution == -1:
                if orig_w > self.max_width:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                              "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / self.max_width
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution
            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))
        return resolution

    def __len__(self):
        return len(self.cam_infos)

    def __getitem__(self, idx):
        cam_info = self.cam_infos[idx]
        # Assume cam_info.image_name is the path to the image file
        image_path = cam_info.image_path
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        resolution = self.compute_resolution(orig_w, orig_h)
        transform = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
        image_tensor = transform(image)
        # Only return RGB channels
        gt_image = image_tensor[:3, ...]
        
        # Create Camera object
        camera = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                        FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                        image=gt_image, gt_alpha_mask=None,  # Assuming no mask for now
                        image_name=cam_info.image_name, uid=idx, data_device=self.args.data_device)
        return camera

def loadCam(args, id, cam_info, resolution_scale):
    if cam_info.image is not None:
        orig_w, orig_h = cam_info.image.size
        image = cam_info.image
    else:
        # Load image from path
        image = Image.open(cam_info.image_path).convert('RGB')
        orig_w, orig_h = image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

# Returns a DataLoader for camera images
def cameraDataLoader_from_camInfos(cam_infos, resolution_scale, args, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False):
    dataset = CameraImageDataset(cam_infos, args, resolution_scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, collate_fn=lambda x: x[0] if batch_size == 1 else x)
    return dataloader

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
