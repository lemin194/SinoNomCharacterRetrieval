import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import cv2
import vtkplotlib as vpl
from stl.mesh import Mesh
from tqdm import tqdm



class ClassifyDataset(Dataset):
    """Classify dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.df.iloc[idx].image
        label = self.df.iloc[idx].label
        
        image = cv2.imread(image_path)
        
        if self.transform:
            image = self.transform(image=image)['image'][[0], :, :]
        
        if isinstance(image, np.ndarray): image = torch.tensor(image.copy())
        label = torch.tensor(label)
        
        
        sample = (image, label)
        return sample