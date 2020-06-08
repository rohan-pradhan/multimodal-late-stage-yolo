import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class AdaptiveFusionGatingNet(nn.Module):
    def __init__(self):
        super(AdaptiveFusionGatingNet, self).__init__()
        resnet18_visible = models.resnet18(pretrained=False)
        modules_visible = list(resnet18_visible.children())[:-1]
        self.visible_feature_extractor = nn.Sequential(*modules_visible)

        resnet18_lwir = models.resnet18(pretrained=False)
        modules_lwir = list(resnet18_lwir.children())[:-1]
        self.lwir_feature_extractor = nn.Sequential(*modules_lwir)

        #self.conv1 = nn.Conv2d(3, )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, visible, lwir):

        visible = self.visible_feature_extractor(visible)
        visible = visible.view(-1, 512)

        lwir = self.lwir_feature_extractor(lwir)
        lwir = lwir.view(-1, 512)
        x = torch.cat((visible, lwir), 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training)
        #return (x)
        x = self.fc5(x)

        return F.sigmoid(x)

