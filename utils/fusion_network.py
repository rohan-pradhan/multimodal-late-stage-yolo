import argparse

import torch.nn as nn
import torch
import numpy
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

class FusionWeightsNet(nn.Module):
    def __init__(self):
        super(FusionWeightsNet, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stirde=4)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(12, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stirde=4)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(20, 30, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stirde=4)
        )
        self.fc1 = nn.Linear(10*8*30,1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x_vision, x_thermal):
        x = torch.cat((x_vision, x_thermal),1)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.reshape(x.size(0), -1) #flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x




def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

class fusion_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = preds[0]
        loss = 0
        bbox_loss = nn.SmoothL1Loss()
        # conf_loss = nn.NLLLoss()
        for pred in preds:
            # partial_loss = 0
            matched = False
            for target in targets:
                iou = bb_intersection_over_union(pred[0:4], target[0:4])
                if (iou > 0.5 and pred[6] == target[6]):
                    matched = True
                    reg_loss = bbox_loss(pred[0:4], target[0:4])
                    conf = torch.neg(torch.log((pred[4])))
                    loss += reg_loss + conf
                    break
            if not matched:
                loss += torch.neg(torch.log(torch.neg(pred[4])+1))
        return (loss)

def train_loop(thermal_source, vision_source, epochs=10, start_epochs=0):
    # adversarial training
    # sensor dropout

    print ("to complete")

    model_thermal = Darknet(opt.thermal_cfg,opt.img_size)
    model_vision = Darknet(opt.vision_cfg, opt.img_size)

    thermal_weights = opt.thermal_weights
    vision_weights = opt.vision_weights

    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    if thermal_weights.endswith('.pt'):  # pytorch format
        model_thermal.load_state_dict(torch.load(thermal_weights, map_location=device)['model'])

    if vision_weights.endswith('.pt'):
        model_vision.load_state_dict(torch.load(vision_weights, map_location=device)['model'])

    model_thermal.to(device).eval()
    model_vision.to(device).eval()

    dataset = LoadMultimodalImagesAndLabels(thermal_path=opt.thermal_source, vision_path=opt.vision_source)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             num_workers=1,
                                             shuffle=False,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    fusion_model = FusionWeightsNet()
    fusion_model.cuda()
    fusion_model.train()
    nb = len(dataloader)

    for epoch in range(start_epoch = start_epochs, epochs=epochs):
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (thermal_img, vision_img, targets, thermal_path, vision_path, _) in pbar:
            print(thermal_img, vision_img)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal_cfg', type=str, default='../cfg/yolov3-custom-thermal-1.cfg',
                        help='thermal cfg file path')
    parser.add_argument('--vision_cfg', type=str, default='../cfg/yolov3-spp.cfg', help='vison cfg file path')
    parser.add_argument('--thermal_data', type=str, default='../data/coco1.data', help='thermal coco.data file path')
    parser.add_argument('--vision_data', type=str, default='../data/coco.data', help='vison coco.data file path')
    parser.add_argument('--thermal_weights', type=str, default='../weights/best.pt', help='path to thermal weights file')
    parser.add_argument('--vision_weights', type=str, default='../weights/ultralytics68.pt',
                        help='path to vison weights file')
    parser.add_argument('--thermal_source', type=str, default='data/samples',
                        help='thermal files source')  # input file/folder, 0 for webcam
    parser.add_argument('--vision_source', type=str, default='data/samples', help='vison files source')
    parser.add_argument('--thermal_out', type=str, default='output/thermal', help='output folder')  # output folder
    parser.add_argument('--vision_out', type=str, default='output/vision', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)
