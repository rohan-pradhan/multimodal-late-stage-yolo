import torch
from utils.AdaptiveFusionGating.AdaptiveFusionGatingNet import AdaptiveFusionGatingNet
from utils.VisualizeCNN.scorecam import ScoreCam, get_example_params, save_class_activation_images
from PIL import Image
import torchvision.transforms as transforms
import cv2
from utils.datasets import letterbox
import numpy as np


pretrained_model = AdaptiveFusionGatingNet()

pretrained_model.load_state_dict((torch.load("./GatingNet4.pt")))
pretrained_model.eval()
score_cam = ScoreCam(pretrained_model, target_layer=8)
vision_img = Image.open("./12072.jpg").convert('RGB')
lwir_img = Image.open("./12072L.jpg").convert('RGB')

#vision_img = Image.open("./34017V.jpg").convert('RGB')
#lwir_img = Image.open("./L.jpg").convert('RGB')

#vision_img = Image.open("./151-b.jpg").convert('RGB')
#lwir_img = Image.open("./151-bl.jpg").convert('RGB')

#vision_img = Image.open("./L.jpg").convert('RGB')
#lwir_img = Image.open("./545.jpg").convert('RGB')
#pil_image = Image.open("./1023.jpg").convert("BGR")


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
transform_2 = transforms.Compose([transforms.Resize((224, 224))])

vision_img = transform(vision_img)
lwir_img = transform(lwir_img)
#transformed_pil_image = transform_2(img)


vision_img = vision_img.unsqueeze(0)
lwir_img = lwir_img.unsqueeze(0)

print(float(pretrained_model(vision_img, lwir_img)[0][0]))


#prep_img = torch.autograd.Variable(img_vision_transformed_for_AFM, requires_grad=True)

#cam = score_cam.generate_cam(prep_img)

#save_class_activation_images(img2, cam, "./test2")







