import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
import yaml
import random



class TuplesDataset(Dataset):
    """Tuples dataset."""

    def __init__(self,
                q_dir='input/2d3d/pairs/print/',
                p_dir='input/2d3d/pairs/stl/pairs/',
                mix=True,
                nnum=5,
                qsize=5000,
                transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.q_dir = q_dir
        self.p_dir = p_dir
        self.transform = transform
        self.img_cache = dict()
        self.qsize=qsize
        self.listdir_cache = dict()
        self.num_images = len(self.get_listdir(self.q_dir))
        self.nnum = min(nnum, self.num_images)
    
    def get_listdir(self, path):
        if path in self.listdir_cache:
            return self.listdir_cache[path]
        self.listdir_cache[path] = os.listdir(path)
        return self.listdir_cache[path]

    def __len__(self):
        return max(self.qsize, self.num_images)
    
    def _load_image(self, idx, query=False):
        path = self.p_dir if random.random() < 0.5 and not query else self.q_dir
        img_path = os.path.join(path, self.get_listdir(path)[idx])
        if os.path.isdir(img_path):
            img_path = os.path.join(img_path, random.choice(self.get_listdir(img_path)))
        return cv2.imread(img_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx%self.num_images

        p_indices = random.choices(list(set(range(self.num_images))-{idx}), k=self.nnum)
        
        q_img = self._load_image(idx, True)
        p_img = self._load_image(idx)
        output = [q_img, p_img]
        labels = torch.Tensor([-1, 1] + [0] * len(p_indices))
        for p_idx in p_indices:
            # p_img_path = os.path.join(self.p_dir, '%d.png' % p_idx)
            # output.append(cv2.imread(p_img_path))
            output.append(self._load_image(p_idx))
        
        
        if self.transform:
            output = [self.transform(image=output[i])['image'][[0]].unsqueeze_(0) for i in (range(len(output)))]
        output = torch.concatenate(output, dim=0)
        return output, labels


class TripletsDataset(Dataset):
    """Triplets dataset."""

    def __init__(self,
                q_dir='input/2d3d/pairs/print/',
                p_dir='input/2d3d/pairs/stl/pairs/',
                mix=True,
                nnum=5,
                qsize=5000,
                transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.q_dir = q_dir
        self.p_dir = p_dir
        self.transform = transform
        self.img_cache = dict()
        self.qsize=qsize
        self.listdir_cache = dict()
        self.num_images = len(self.get_listdir(self.q_dir))
        self.nnum = min(nnum, self.num_images)
    
    def get_listdir(self, path):
        if path in self.listdir_cache:
            return self.listdir_cache[path]
        self.listdir_cache[path] = os.listdir(path)
        return self.listdir_cache[path]

    def __len__(self):
        return max(self.qsize, self.num_images)
    
    def _load_image(self, idx, query=False):
        path = self.p_dir if random.random() < 0.5 and not query else self.q_dir
        img_path = os.path.join(path, self.get_listdir(path)[idx])
        if os.path.isdir(img_path):
            img_path = os.path.join(img_path, random.choice(self.get_listdir(img_path)))
        return cv2.imread(img_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx%self.num_images

        p_indices = random.choices(list(set(range(self.num_images))-{idx}), k=self.nnum)
        
        q_img = self._load_image(idx, True)
        p_img = self._load_image(idx)
        output = [q_img, p_img]
        labels = [idx, idx]
        for p_idx in p_indices:
            # p_img_path = os.path.join(self.p_dir, '%d.png' % p_idx)
            # output.append(cv2.imread(p_img_path))
            output.append(self._load_image(p_idx))
            labels.append(p_idx)
        labels = torch.Tensor(labels)
        
        
        if self.transform:
            output = [self.transform(image=output[i])['image'][[0]].unsqueeze_(0) for i in (range(len(output)))]
        output = torch.concatenate(output, dim=0)
        return output, labels



class ClassifyDataset(Dataset):
    """Classify dataset."""

    def __init__(self,
                q_dir='input/2d3d/pairs/print/',
                p_dir='input/2d3d/pairs/stl/pairs/',
                mix=True, nnum=5, qsize=500,
                transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.q_dir = q_dir
        self.p_dir = p_dir
        self.data = set()
        for dirname, _, filenames in list(os.walk(q_dir)) + list(os.walk(p_dir)):
            lbl = dirname.split('/')[-1]
            for filename in filenames:
                full_path = str(os.path.join(dirname, filename))
                if ('.png' not in full_path) and ('.jpg' not in full_path): 
                    continue
                self.data.add((os.path.join(dirname, filename), int(lbl)))
        self.data = list(self.data)
        
        self.transform = transform
        self.img_cache = dict()
        self.listdir_cache = dict()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.data[idx]
        
        image = cv2.imread(image_path)
        
        if self.transform:
            image = self.transform(image=image)['image'][[0]]
        
        if isinstance(image, np.ndarray): image = torch.tensor(image.copy())
        label = torch.tensor(label)
        
        
        sample = (image, label)
        return sample

class PairsDataset(Dataset):
    """Pairs dataset."""

    def __init__(self,
                q_dir='input/2d3d/pairs/print/',
                p_dir='input/2d3d/pairs/stl/pairs/',
                mix=True,
                transform=None):
        """
        Arguments:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.q_dir = q_dir
        self.p_dir = p_dir
        self.transform = transform
        self.img_cache = dict()
        self.num_images = len(os.listdir(self.q_dir))

    def __len__(self):
        return max(50000, self.num_images)
    
    def _load_image(self, idx, query=False):
        path = self.p_dir if random.random() < 0.5 and not query else self.q_dir
        img_path = os.path.join(path, os.listdir(path)[idx])
        if os.path.isdir(img_path):
            img_path = os.path.join(img_path, random.choice(os.listdir(img_path)))
        if img_path in self.img_cache: return self.img_cache[img_path]
        self.img_cache[img_path] = cv2.imread(img_path)
        return self.img_cache[img_path]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx % self.num_images
        
        src_img = self._load_image(idx, query=True)
        if random.random() < 0.1:       
            tgt_img = self._load_image(idx)
            labels = torch.tensor(1.0)
        else:
            p_idx = random.choice(range(self.num_images))
            tgt_img = self._load_image(p_idx)
            labels = torch.tensor(1.0 if p_idx == idx else 0.0)
        
        
        if self.transform:
            src_img = self.transform(image=src_img)['image'][[0]]
            tgt_img = self.transform(image=tgt_img)['image'][[0]]

        return src_img, tgt_img, labels