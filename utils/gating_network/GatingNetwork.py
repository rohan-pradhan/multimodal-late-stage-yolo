import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from utils.RGB_Day_Night_Detector_updated.dataset import DayNightDataset
import torchvision.transforms as transforms
from barbar import Bar
from math import floor, ceil
import matplotlib.pyplot as plt
import numpy as np

class GatingNet(nn.Module):
    def __init__(self):
        super(GatingNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)

