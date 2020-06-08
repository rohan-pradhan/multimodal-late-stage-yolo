from utils.AdaptiveFusionGating.AdaptiveFusionGatingNet import AdaptiveFusionGatingNet
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
import cv2
from statistics import mean
import pandas as pd

def plot_bbox(thermal_img_path, vision_img_path, preds, targets):
    t_img = thermal_img_path[0].cpu().numpy().transpose(1,2,0)*255
    v_img = vision_img_path[0].cpu().numpy().transpose(1,2,0)*255
    cv2.imwrite("./t_img.jpg", t_img)
    cv2.imwrite("./v_img.jpg", v_img)

    t_img = cv2.imread("./t_img.jpg")
    v_img = cv2.imread("./v_img.jpg")

    #print (t_img.shape)
    #print (preds, targets)
    number_of_preds = preds[0].shape[0]
    number_of_targets = targets.shape[1]
    if number_of_preds > 0:
        preds = preds[0][:, :4].cuda()  # REMOVE CONFIDENCES AND ONLY KEEP BOUNDING BOX #SHOULD I KEEP CLASS?
        for pred in preds:
            x1 = pred[0]
            y1 = pred[1]
            x2 = pred[2]
            y2 = pred[3]
           # print (x1, y1, x2, y2)
           # print (type(t_img))
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            color = (0, 0, 255)
            cv2.rectangle(t_img, pt1, pt2, color, 2)
            cv2.rectangle(v_img, pt1, pt2, color, 2)

    if number_of_targets > 0:
        targets = targets.squeeze(0)[:, 2:]
        for target in targets:
            x1 = target[0]
            y1 = target[1]
            x2 = target[2]
            y2 = target[3]
            #print(x1, y1, x2, y2)
            #print(type(t_img))
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            color = (0, 255, 0)
            cv2.rectangle(t_img, pt1, pt2, color, 2)
            cv2.rectangle(v_img, pt1, pt2, color, 2)

    return t_img, v_img




def match(threshold, truths, priors):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        (priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    # conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    # conf[best_truth_overlap < threshold] = 0
    return matches


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    # print (A,B)
    # print ("boxa: ", box_a, "boxb: ", box_b)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def filter_double_matches(ious):
    #unique_indicies, counts_of_indicies = torch.unique(matched_ious[:, 0], return_counts=True)
    #repeated_indicies_index = (counts_of_indicies > 1).nonzero()
    #repeated_indicies = unique_indicies[repeated_indicies_index]

    for index, pred in enumerate(ious):
        pred = torch.where(pred == torch.max(pred), torch.max(pred), torch.Tensor([0]).cuda())
        ious[index] = pred

    return ious





def custom_reward_updated(preds, targets):
    number_of_preds = preds[0].shape[0]
    number_of_targets = targets.shape[1]
    target_ones = None

    if number_of_targets > 0:
        target_ones = torch.ones(number_of_targets).cuda()

    #print (number_of_preds, number_of_targets)


    if number_of_preds > 0 and number_of_targets > 0:
        preds = preds[0][:, :4].cuda()  # REMOVE CONFIDENCES AND ONLY KEEP BOUNDING BOX #SHOULD I KEEP CLASS?
        targets = targets.squeeze(0)[:, 2:]  # REMOVE CONFIDENCES AND ONLY KEEP BOUNDING BOX #SHOULD I KEEP CLASS?
        ious = jaccard(preds,
                       targets).cuda()  # ious of objects shown in 2D tensor. size of tensor: [n_preds, n_targets]

        indicies = (ious > 0.3).nonzero().cuda()  # filter and find indices of IOUs that can be considered "good match"
        matched_ious = ious[indicies[:, 0], indicies[:, 1]].cuda()  # retrieve filtered IOU values by index
        ious = filter_double_matches(ious)

        number_of_preds = max(number_of_preds, matched_ious.shape[0])

        if number_of_preds > number_of_targets:
            target_ones = torch.ones(number_of_targets).cuda()
            number_of_extra_preds = number_of_preds - number_of_targets
            target_zeros = torch.zeros(number_of_extra_preds).cuda()
            target_ones = torch.cat([
                target_ones,
                target_zeros
            ])

        #print ("IOUs: ", ious)


        number_of_matched_preds = matched_ious.shape[0]

        number_of_non_matched_preds = abs(number_of_preds - matched_ious.shape[0])
        #print ("preds :", number_of_preds, "number_of_targets: ", number_of_targets, "number_of_matched: ", matched_ious.shape[0], "number of non macthed: ", number_of_non_matched_preds)

        if (number_of_preds == number_of_targets):

            matched_ious = torch.cat([matched_ious,
                                  torch.zeros(number_of_non_matched_preds).cuda()])
        elif (number_of_preds>number_of_targets):
            matched_ious = torch.cat([matched_ious,
                                      torch.ones(number_of_non_matched_preds).cuda()])

       # pad_number = target_ones.shape[0] - matched_ious.shape[0]
        pad_number = target_ones.shape[0] - matched_ious.shape[0]
        if pad_number >=0 :
            '''
            e.g. 
            n_targets = 5 
            n_matched_predictions = 3 
                2 targets were not predicted 
            matched_predictions = [0.53, 0.32, 0.69] 
            target_ones = [1, 1, 1, 1, 1] 



            '''
            matched_ious = torch.cat([matched_ious,
                                      torch.zeros(pad_number).cuda()])
        elif pad_number < 0:
            pad_number = abs(pad_number)
            target_ones = torch.cat([target_ones,
                                     torch.zeros(pad_number).cuda()])
        #print ("target_ones:", target_ones)
        #print ("matched_ious :", matched_ious)
        return torch.mean(torch.abs(target_ones - matched_ious))

        # return match(0.5, preds, targets)
    elif number_of_preds == 0 and number_of_targets > 0:
        return (torch.Tensor([1.0])).cuda()

    elif number_of_preds == 0 and number_of_targets == 0:
        #print ("Nothing to return")
        return None



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
       # 'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       # 'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       # 'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'hsv_h': 0.0,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.0,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.0,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)


def train(opt):
    #GatingNet = AdaptiveFusionGatingNet()
    #GatingNet.train()
    #GatingNet.cuda()

    # print (GatingNet(torch.tensor([[1,2]]).cuda().float()))

    #optimizer = torch.optim.Adam(GatingNet.parameters(), lr=0.0001)
    # criterion = torch.nn.MSELoss()
    #criterion = torch.nn.MSELoss()

    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])

    column_names = ["visible_filename", "lwir_filename", "label"]

    dataframe = pd.DataFrame(columns=column_names)

    img_size = (416, 416)  # (320, 192) or (416, 256) or (608, 352) for (height, width)
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

    fog_net = FogDetectNet().eval().cuda()
    night_net = DayNightDetectNet().eval().cuda()

    fog_net.load_state_dict(torch.load("./utils/AFM_WEIGHTS/fog_detector.pt"))
    night_net.load_state_dict(torch.load("./utils/AFM_WEIGHTS/day_detector.pt"))

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

    fog_night = []
    no_fog_night = []
    fog_day = []
    no_fog_day = []
    for epoch in range(1):
        cumulative_loss = 0
        counter = 0
        print("Epoch: ", epoch)
        temp_counter = 0
        for batch_i, (img_thermal, img_vision, targets, thermal_path, vision_path, shapes) in enumerate(
                tqdm.tqdm(dataloader)):
            if targets.shape[1] == 0:
                continue
            temp_counter += 1

            #print (thermal_path)

            img = cv2.imread(vision_path[0])
            img = img/255.0
            img = img.transpose(2,0,1)
            #print(img.shape)



            img = torch.from_numpy(img).float()

            #optimizer.zero_grad()
            img_thermal = img_thermal.to(device).float() / 255.0
            img_vision = img_vision.float() / 255.0
            targets = targets.to(device)
            _, _, height, width = img_thermal.shape

            img_vision_transformed_for_AFM = transform((img))
            img_vision = img_vision.to(device)

            img_vision_transformed_for_AFM = img_vision_transformed_for_AFM.cuda().unsqueeze(0)

            #night_value = round(night_net(img_vision_transformed_for_AFM)[0].item())
            #fog_value = round(fog_net(img_vision_transformed_for_AFM)[0].item())



            #print (vision_path)
            #print("NIGHT VALUE: ", night_value, " | FOG VALUE: ", fog_value)
            #print(" ------")

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


            thermal_values = np.arange(0.0, 1.05, 0.25)
            ratio_dict = np.ones(5)

            for index, thermal_val in np.ndenumerate(thermal_values):

                vision_val = 1 - thermal_val
                output = [GraphClique.fusion_graph(np_thermal_pred, np_vision_pred, thermal_val, vision_val)]

                #plotted_img_thermal, plotted_img_vision = plot_bbox(img_thermal, img_vision, output, targets)
                #combined_img = cv2.vconcat([plotted_img_thermal, plotted_img_vision])
                #cv2.imshow("window", combined_img)
                #cv2.waitKey(10)
                #print (output)

                alpha = 0.95
                beta = 0.05
                conf_loss = 0

                if output[0].shape[0] > 0:
                    conf_loss = 1 - torch.mean(output[0][:,4])
                localized_loss = custom_reward_updated(output, targets)

                if localized_loss == 1:
                    #print ("No matches")
                    loss = localized_loss
                else:
                    loss = alpha*localized_loss + beta*conf_loss
                if loss is not None:
                    ratio_dict[index] = loss.item()

            if np.min(ratio_dict) < 1:
                best_thermal_val = 0.25 * (np.argmin(ratio_dict))
                best_vision_val = 1 - best_thermal_val

            else:
                best_thermal_val = 0.5
                best_vision_val = 0.5

                #print("-------")
                #print("path: ", vision_path)
                #print("active learning loss: ", loss)
                #print("vision val: ", best_vision_val)
                #print("thermal val: ", best_thermal_val)

                # if fog_value == 1 and night_value ==1:
                #     fog_night.append(best_vision_val)
                #     #print ("fog_night: ", mean(fog_night), len(fog_night))
                # elif fog_value == 0 and night_value ==1:
                #     no_fog_night.append(best_vision_val)
                #     #print ("no_fog_night: ", mean(no_fog_night), len(no_fog_night))
                #
                # elif fog_value == 1 and night_value == 0:
                #     fog_day.append(best_vision_val)
                #     #print ("fog_day: ", mean(fog_day), len(fog_day))
                #
                # elif fog_value==0 and night_value == 0:
                #     no_fog_day.append(best_vision_val)
                #     #print ("no fog day: ", mean(no_fog_day), len(no_fog_day))

            dataframe = dataframe.append({"visible_filename": vision_path,
                                              "lwir_filename": thermal_path,
                                              "label": best_vision_val}, ignore_index=True)



                #print (best_vision_val)
                #print ("------")

            target = torch.Tensor([[best_vision_val]]).cuda().float()


            if temp_counter % 1000 == 0:
                dataframe = dataframe.sample(frac=1)
                # dataframe = dataframe[:100]
                dataframe = dataframe.reset_index(drop=True)

                dataframe.to_pickle("train_dataset.pkl")



        dataframe = dataframe.sample(frac=1)
                # dataframe = dataframe[:100]
        dataframe = dataframe.reset_index(drop=True)

        dataframe.to_pickle("train_dataset.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal_cfg', type=str, default='cfg/yolov3-custom-thermal-1.cfg',
                        help='thermal cfg file path')
    parser.add_argument('--vision_cfg', type=str, default='cfg/yolov3-spp.cfg', help='vison cfg file path')
    parser.add_argument('--thermal_data', type=str, default='data_files/data/coco1.data',
                        help='thermal coco.data file path')
    parser.add_argument('--vision_data', type=str, default='data_files/data/coco.data',
                        help='vison coco.data file path')
    parser.add_argument('--thermal_weights', type=str, default='models/yolov3-weights/best.pt',
                        help='path to thermal weights file')
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
    # print(opt)

    print(train(opt))

    # python test_multimodal.py --vision_source D:/FLIR/val/RGB_adjusted --thermal_source D:/FLIR/val/thermal_8_bit_adjusted
#python ./debug_graph.py --vision_source D:/KAIST_Dataset_FOG/train/images/visible_gating --thermal_source D:/KAIST_Dataset_FOG/train/images/lwir_gating --thermal_cfg cfg/yolov3-kaist-thermal.cfg --vision_cfg cfg/yolov3-kaist-visible.cfg --thermal_data D:/KAIST_Dataset_FOG/data/kaist_thermal.data --vision_data D:/KAIST_Dataset_FOG/data/kaist_visible.data --thermal_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/lwir/best.pt --vision_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/visible_not_trained_with_fog/best.pt
