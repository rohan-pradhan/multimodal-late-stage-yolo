import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
from torch import optim
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from ResNetVAE import ResNet_VAE
from dataset import ResNetVAE_Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 256     # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability

# training parameters
epochs = 20        # training epochs
batch_size = 50
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info


# save model
save_model_path = './results_MNIST'


def loss_function(recon_x, x):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    #MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return MSE


def train():
    net = ResNet_VAE()
    print(net)
    net.train().cuda()

    optimizer = optim.Adam(net.parameters(), lr=1)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    transform = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor()])
    train_set = ResNetVAE_Dataset("dataset.pkl", visible_transform=transform)


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)

    for epoch in range(100):  # loop over the dataset multiple times
        epoch_loss = 0.0
        running_loss = 0.0
        scheduler.step()
        for i, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            if data["image"].shape[0] <= 1:
                continue
            #print ("Got here")
            inputs = data["image"]
            inputs = inputs.cuda()
            #print (inputs)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print (outputs.shape, inputs.shape)
            # print (outputs)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        print("Epoch loss: ", epoch+1, epoch_loss/len(train_set), " | Learning Rate: ", scheduler.get_lr())
        torch.save(net.state_dict(), "test4.pt")



    print ("Finished training. ")

if __name__ == '__main__':
    train()