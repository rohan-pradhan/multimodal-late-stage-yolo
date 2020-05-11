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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
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

def train():
    net = Net()
    #net.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor()])
    dataset = DayNightDataset("./dataset.pkl", transform=transform)

    train_set_length = int(ceil(len(dataset) * 0.8))
    val_set_length = int(floor(len(dataset) * 0.2))

    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_length, val_set_length])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True, num_workers=4)

    for epoch in range(1):  # loop over the dataset multiple times
        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(Bar(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["image"]
            labels = data["classification"]
            inputs = inputs.cuda()
            labels = labels.cuda().float()
            # print (labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print (outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        print("Epoch loss: ", epoch+1, epoch_loss/len(train_set))
        print("Validation accuracy...")
        correct = 0
        for i, data in enumerate(Bar(valloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["image"]
            labels = data["classification"]
            inputs = inputs.cuda()
            labels = labels.cuda().float()
            outputs = net(inputs)
            # print (outputs)
            loss = criterion(outputs, labels)
            outputs = outputs.round()

            correct += (outputs == labels).float().sum()

        print(correct/len(val_set))

    torch.save(net.state_dict(), "day_detector.pt")

    print ("Finished training. ")

def test():
    net = Net()
    net.load_state_dict(torch.load("day_detector.pt"))
    net.cuda()
    net.eval()
    criterion = nn.BCELoss()

    transform = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor()])
    test_dataset = DayNightDataset("./test.pkl", transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    correct = 0
    for i, data in enumerate(Bar(testloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data["image"]
        # print(inputs[0].shape)
        labels = data["classification"]
        inputs = inputs.cuda()
        labels = labels.cuda().float()
        outputs = net(inputs)
        # print (outputs)
        loss = criterion(outputs, labels)
        outputs = outputs.round()
        # print(outputs)
        # print(outputs)

        correct += (outputs == labels).float().sum()

    print(correct / len(test_dataset))

if __name__ == "__main__":
    test()
    #train()

