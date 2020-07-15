from utils.AdaptiveFusionGating import dataset, AdaptiveFusionGatingNet
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

dataset = dataset.AdaptiveFusionGatingDataset("train_FLIR_Gating.pkl", visible_transform=transform, lwir_transform=transform)
dataloader = DataLoader(dataset, shuffle=True ,pin_memory=True, batch_size=32)


GatingNet = AdaptiveFusionGatingNet.AdaptiveFusionGatingNet()
GatingNet.train().cuda()

optimizer = torch.optim.Adam((GatingNet.parameters()), lr=0.00001)
criterion = torch.nn.MSELoss()
#
for epoch in range(10):  # loop over the dataset multiple times
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # get the inputs; data is a list of [inputs, labels]
        visible_inputs = data["visible_image"]
        lwir_inputs = data["lwir_image"]
        labels = data["classification"]
        #inputs = inputs.cuda()
        visible_inputs = visible_inputs.cuda()
        lwir_inputs = lwir_inputs.cuda()
        labels = labels.cuda().float()
        # print (labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = GatingNet(visible_inputs, lwir_inputs)
        #print (outputs.shape)
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
    print("Epoch loss: ", epoch + 1, epoch_loss / len(dataloader))
    print("Validation accuracy...")
    torch.save(GatingNet.state_dict(), "FLIR_Gating.pt")


