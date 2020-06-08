import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class AdaptiveFusionGatingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file, visible_transform=None, lwir_transform=None):
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
        self.lwir_transform = lwir_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        visible_img_name = self.df.iloc[idx, 0][0]
        visible_image = Image.open(visible_img_name).convert('RGB')

        lwir_img_name = self.df.iloc[idx, 1][0]
        lwir_immage = Image.open(lwir_img_name).convert('RGB')


        # image = torch.from_numpy(image)
        classification = (self.df.iloc[idx, 2])
        classification = np.array([classification])
        sample = {'visible_image': visible_image,
                  'lwir_image': lwir_immage,
                  'classification': classification}

        if self.visible_transform:
            sample["visible_image"] = self.visible_transform(sample["visible_image"])

        if self.lwir_transform:
            sample["lwir_image"] = self.lwir_transform(sample["lwir_image"])

        return sample
