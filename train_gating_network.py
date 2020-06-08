from utils.gating_network import GatingNetwork
from utils.ObjGraphClique import GraphClique

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import argparse
import random
from utils.AdaptiveFusionModule.DayNightDetect import Net as DayNightDetectNet
from utils.AdaptiveFusionModule.FogDetect import Net as FogDetectNet

import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm


def custom_reward(preds, targets):
    number_of_preds = preds[0].shape[0]
    #print (targets.shape)
    preds = preds[0]
    number_of_targets = targets.shape[1]
    if not number_of_preds == number_of_targets:
        return 0
    else:
        if number_of_preds > 0:
            return torch.mean(preds[:,4])
        else:
            return 0



def custom_reward1(preds, targets):
    number_of_preds = preds[0].shape[0]
    #print (targets.shape)
    preds = preds[0]
    number_of_targets = targets.shape[1]
    if not number_of_preds == number_of_targets:
        return -1
    else:
        if number_of_preds > 0:
            number_of_cyclists_pred = (preds[:, 6]==2).sum(dim=0).cuda()
            number_of_cyclists_target = (targets[0,:, 1]==2).sum(dim=0).cuda()

            number_of_people_pred = (preds[:, 6] == 0).sum(dim=0).cuda()
            number_of_people_pred += (preds[:, 6] == 1).sum(dim=0).cuda()
            number_of_people_target = (targets[0,:, 1] == 0).sum(dim=0).cuda()
            number_of_people_target += (targets[0,:, 1] == 1).sum(dim=0).cuda()


            #rint (number_of_people_pred, number_of_people_target)

            if number_of_cyclists_pred == number_of_cyclists_target and number_of_people_pred == number_of_people_target:
                return torch.mean(preds[:,4])
            else:
                return 0
        else:
            return 0


hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/416 if img_size != 416)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00579,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       #'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       #'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       #'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'hsv_h': 0.0,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.0,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.0,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

def train(opt):
    GatingNet = GatingNetwork.GatingNet()
    GatingNet.train()
    GatingNet.cuda()

    print (GatingNet(torch.tensor([[1,2]]).cuda().float()))

    optimizer = torch.optim.Adam(GatingNet.parameters(), lr=0.05)


    RGB_DAY_DETECTOR_MODEL = DayNightDetectNet()
    RGB_DAY_DETECTOR_MODEL.load_state_dict(torch.load("utils/RGB_Day_Night_Detector_updated/day_detector.pt"))
    RGB_DAY_DETECTOR_MODEL.cuda()
    RGB_DAY_DETECTOR_MODEL.eval()

    FOG_DETECTOR_MODEL = FogDetectNet()
    FOG_DETECTOR_MODEL.load_state_dict(torch.load("utils/AFM_WEIGHTS/fog_detector.pt"))
    FOG_DETECTOR_MODEL.cuda()
    FOG_DETECTOR_MODEL.eval()


    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

    img_size = (416,416)  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    thermal_out, vision_out, thermal_source, vision_source, thermal_weights, vision_weights, half, view_img = opt.thermal_out, \
                                                                                                              opt.vision_out, \
                                                                                                              opt.thermal_source, \
                                                                                                              opt.vision_source, \
                                                                                                              opt.thermal_weights, \
                                                                                                              opt.vision_weights, \
                                                                                                              opt.half, \
                                                                                                              opt.view_img

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    # Initialize models
    model_thermal = Darknet(opt.thermal_cfg, img_size)
    model_vision = Darknet(opt.vision_cfg, img_size)

    model_thermal.arc = opt.arc
    model_vision.arc = opt.arc

    # Load weights
    if thermal_weights.endswith('.pt'):  # pytorch format
        model_thermal.load_state_dict(torch.load(thermal_weights, map_location=device)['model'])

    if vision_weights.endswith('.pt'):
        model_vision.load_state_dict(torch.load(vision_weights, map_location=device)['model'])

    # Eval mode
    model_thermal.to(device).eval()
    model_vision.to(device).eval()

    model_thermal.hyp = hyp
    model_vision.hyp = hyp

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model_thermal.half()
        model_vision.half()

    dataset = LoadMultimodalImagesAndLabels(thermal_path=thermal_source, vision_path=vision_source, img_size=416)

    dataloader = DataLoader(dataset, shuffle=True,
                            pin_memory=True)

    # Run inference
    t0 = time.time()
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    seen = 0

    counter = 0
    for batch_i, (img_thermal, img_vision, targets, thermal_path, vision_path, shapes) in enumerate(
            tqdm.tqdm(dataloader)):
        optimizer.zero_grad()
        img_thermal = img_thermal.to(device).float() / 255.0
        img_vision = img_vision.float() / 255.0
        targets = targets.to(device)
        _, _, height, width = img_thermal.shape

        img_vision_transformed_for_AFM = transform(torch.squeeze(img_vision))
        img_vision = img_vision.to(device)


        img_vision_transformed_for_AFM = img_vision_transformed_for_AFM.cuda().unsqueeze(0)

        day_value = RGB_DAY_DETECTOR_MODEL(img_vision_transformed_for_AFM)[0][0].detach()
        fog_value = FOG_DETECTOR_MODEL(img_vision_transformed_for_AFM)[0][0].detach()


        if img_vision.ndimension() == 3:
            img_vision = img_vision.unsqueeze(0)

        if img_thermal.ndimension() == 3:
            img_thermal = img_thermal.unsqueeze(0)

        pred_thermal = model_thermal(img_thermal)[0]
        pred_vision = model_vision(img_vision)[0]

        if opt.half:
            pred_thermal = pred_thermal.float()
            pred_vision = pred_vision.float()

        # Apply NMS
        pred_thermal = non_max_suppression(pred_thermal, opt.conf_thres, opt.nms_thres)
        pred_vision = non_max_suppression(pred_vision, opt.conf_thres, opt.nms_thres)

        try:
            np_thermal_pred = pred_thermal[0].cpu().detach().numpy()
        except:
            np_thermal_pred = []
        try:
            np_vision_pred = pred_vision[0].cpu().detach().numpy()
        except:
            np_vision_pred = []

        if len(np_vision_pred) == 0 and len(np_thermal_pred) == 0 and targets.shape[1] == 0:
            continue

        gating_input = torch.autograd.Variable(torch.tensor([[day_value, fog_value]]).float(), requires_grad=True ).cuda()

        gating_output = GatingNet(gating_input)
        print (gating_output, day_value, fog_value)


        thermal_multiplier = float(gating_output[0][0].item())
        vision_multiplier = float(gating_output[0][1].item())

        thermal_values = np.arange(0.0, 1.05, 0.05)
       #print (thermal_values.shape)
        best_thermal_val = None
        best_vision_val = None
        ratio_dict = np.empty(21)
        for index, thermal_val in np.ndenumerate(thermal_values):
            vision_val = 1 -thermal_val
            output = [GraphClique.fusion_graph(np_thermal_pred, np_vision_pred, thermal_val, vision_val)]

            #print (output, targets)
            active_loss = custom_reward(output, targets)
           #print (loss)
            ratio_dict[index] = active_loss
            #print (loss)

        if np.max(ratio_dict) == 0:
            best_thermal_val = 0.5
            best_vision_val = 0.5
        else:
            best_thermal_val = 0.05 * (np.argmax(ratio_dict))
            best_vision_val = 1 - best_thermal_val



    torch.save(GatingNet.state_dict(), "GatingNet.pt")
    print(counter)
    print (GatingNet(torch.Tensor([[0, 0]]).cuda().float()))
    print (GatingNet(torch.Tensor([[1,0]]).cuda().float()))
    print (GatingNet(torch.Tensor([[0,1]]).cuda().float()))
    print (GatingNet(torch.Tensor([[1,1]]).cuda().float()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal_cfg', type=str, default='cfg/yolov3-custom-thermal-1.cfg',
                        help='thermal cfg file path')
    parser.add_argument('--vision_cfg', type=str, default='cfg/yolov3-spp.cfg', help='vison cfg file path')
    parser.add_argument('--thermal_data', type=str, default='data_files/data/coco1.data', help='thermal coco.data file path')
    parser.add_argument('--vision_data', type=str, default='data_files/data/coco.data', help='vison coco.data file path')
    parser.add_argument('--thermal_weights', type=str, default='models/yolov3-weights/best.pt', help='path to thermal weights file')
    parser.add_argument('--vision_weights', type=str, default='models/yolov3-weights/ultralytics68.pt',
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
    parser.add_argument('--dt_path')
    parser.add_argument('--gt_path')
    parser.add_argument("--thermal_multiplier")
    parser.add_argument("--vision_multiplier")
    parser.add_argument("--day_night", default=0)
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # defaultpw, uCE, uBCE



    opt = parser.parse_args()
    #print(opt)


    print(train(opt))

        #python test_multimodal.py --vision_source D:/FLIR/val/RGB_adjusted --thermal_source D:/FLIR/val/thermal_8_bit_adjusted
#python ./train_gating_network.py --vision_source D:/KAIST_Dataset_FOG/val/images/visible_all --thermal_source D:/KAIST_Dataset_FOG/val/images/lwir_all --thermal_cfg cfg/yolov3-kaist-thermal.cfg --vision_cfg cfg/yolov3-kaist-visible.cfg --thermal_data D:/KAIST_Dataset_FOG/data/kaist_thermal.data --vision_data D:/KAIST_Dataset_FOG/data/kaist_visible.data --thermal_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/lwir/best.pt --vision_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/visible/best.pt
