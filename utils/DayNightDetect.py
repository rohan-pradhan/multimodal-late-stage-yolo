import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        modules = list(resnet18.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        #self.fc1 = nn.Linear(512, 256)
        #self.fc2 = nn.Linear(256, 1)
        self.fc1 = nn.Linear(512, 5)
        self.fc2 = nn.Linear(5, 1)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x