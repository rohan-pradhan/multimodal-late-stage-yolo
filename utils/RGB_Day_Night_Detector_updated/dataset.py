import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class DayNightDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pickle_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_pickle(pickle_file)
        # self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        img_name = self.df.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')
        # image = torch.from_numpy(image)
        classification = (self.df.iloc[idx, 1])
        classification = np.array([classification])
        sample = {'image': image, 'classification': classification}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
