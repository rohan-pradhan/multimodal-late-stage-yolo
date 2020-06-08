from utils.AdaptiveFusionGating import dataset, AdaptiveFusionGatingNet
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from PIL import Image

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

GatingNet = AdaptiveFusionGatingNet.AdaptiveFusionGatingNet()
GatingNet.load_state_dict(torch.load("GatingNet2.pt"))
GatingNet.eval().cuda()


image = Image.open("50197.jpg").convert('RGB')
tensor_to_predict = transform(image).cuda()
print (GatingNet(tensor_to_predict.unsqueeze(0)))
