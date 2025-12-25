import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import h5py
import cv2
from torchvision import transforms
from config import Config

class ShanghaiTechDataset(Dataset):
    """
    ShanghaiTech Crowd Counting Dataset (Part B) with pre-generated H5 density maps.
    """
    def __init__(self, root_path, phase='train', main_transform=None, fixed_size=None, augment=False):
        """
        Args:
            root_path (str): Path to the dataset root (e.g., '.../ShanghaiTech/part_B').
            phase (str): 'train' or 'test'.
            main_transform (callable, optional): Transform for the image (e.g., ToTensor, Normalize).
            fixed_size (tuple, optional): (H, W) to resize input images. If None, use original size.
            augment (bool): Whether to apply data augmentation (Random Crop, Flip).
        """
        random.seed(Config.RANDOM_SEED)
        self.root_path = root_path
        self.phase = phase
        self.main_transform = main_transform
        self.fixed_size = fixed_size
        self.augment = augment
        
        # Setup paths
        if phase == 'train':
            self.img_dir = os.path.join(root_path, 'train_data', 'images')
            self.gt_dir = os.path.join(root_path, 'train_data', 'ground-truth-h5')
        elif phase == 'test' or phase == 'val':
            self.img_dir = os.path.join(root_path, 'test_data', 'images')
            self.gt_dir = os.path.join(root_path, 'test_data', 'ground-truth-h5')
        else:
            raise ValueError(f"Unknown phase: {phase}")
            
        # Filter for images
        self.data_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        self.num_samples = len(self.data_files)
        
        print(f"[{phase.upper()}] Loaded {self.num_samples} samples from {self.img_dir}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        fname = self.data_files[index]
        img_path = os.path.join(self.img_dir, fname)
        
        # Load Image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Load Ground Truth Density Map (.h5)
        # Filename convention: IMG_1.jpg -> IMG_1.h5 (or similar)
        # Usually it's just replacing extension
        gt_name = fname.replace('.jpg', '.h5')
        gt_path = os.path.join(self.gt_dir, gt_name)
        
        density_map = None
        if os.path.exists(gt_path):
            try:
                with h5py.File(gt_path, 'r') as hf:
                    density_map = np.asarray(hf['density'])
            except Exception as e:
                print(f"Error loading GT H5 for {fname}: {e}")
        else:
            # Fallback: try with 'GT_' prefix if standard name fails, or vice versa
            # Some datasets name it GT_IMG_1.h5
            gt_name_alt = 'GT_' + fname.replace('.jpg', '.h5')
            gt_path_alt = os.path.join(self.gt_dir, gt_name_alt)
            if os.path.exists(gt_path_alt):
                try:
                    with h5py.File(gt_path_alt, 'r') as hf:
                        density_map = np.asarray(hf['density'])
                except Exception as e:
                    print(f"Error loading GT H5 for {fname}: {e}")
            else:
                print(f"GT H5 file not found for {fname}, generating empty density map.")

        if density_map is None:
             density_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        # Augmentation or Resize Logic
        if self.augment:
            # If augmenting, we use fixed_size as the crop size
            if self.fixed_size is not None:
                img, density_map = self.random_crop(img, density_map, self.fixed_size)
            
            img, density_map = self.horizontal_flip(img, density_map)
            
        elif self.fixed_size is not None:
            # Validation/Test: Resize to fixed size
            target_h, target_w = self.fixed_size
            if (target_h, target_w) != (orig_h, orig_w):
                # Resize Image
                img = img.resize((target_w, target_h), Image.BILINEAR)
                
                # Resize Density Map
                # We need to preserve the count (sum)
                original_count = density_map.sum()
                density_map = cv2.resize(density_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                new_count = density_map.sum()
                
                if new_count > 0:
                    density_map = density_map * (original_count / new_count)
        
        # Apply Transforms to Image
        if self.main_transform:
            img = self.main_transform(img)
        else:
            img = transforms.ToTensor()(img)
            
        # Convert Density Map to Tensor
        # Shape: (1, H, W)
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)
        
        return img, density_map

    def random_crop(self, img, density_map, crop_size):
        w, h = img.size
        th, tw = crop_size
        if w == tw and h == th:
            return img, density_map
        
        if w < tw or h < th:
            # If image is smaller than crop size, resize it first
            img = img.resize((tw, th), Image.BILINEAR)
            original_count = density_map.sum()
            density_map = cv2.resize(density_map, (tw, th), interpolation=cv2.INTER_LINEAR)
            new_count = density_map.sum()
            if new_count > 0:
                density_map = density_map * (original_count / new_count)
            return img, density_map

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        img = img.crop((j, i, j + tw, i + th))
        density_map = density_map[i:i+th, j:j+tw]
        
        return img, density_map

    def horizontal_flip(self, img, density_map):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            density_map = np.fliplr(density_map).copy()
        return img, density_map
