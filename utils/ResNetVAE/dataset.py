import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ResNetVAE_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file, visible_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_pickle(pickle_file)
        # self.root_dir = root_dir
        self.visible_transform = visible_transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        visible_img_name = self.df.iloc[idx, 0]
        visible_image = Image.open(visible_img_name).convert('RGB')

        sample = {'image': visible_image}

        if self.visible_transform:
            sample["image"] = self.visible_transform(sample["image"])




        return sample
