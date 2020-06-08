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

# def custom_reward_updated(preds, targets):
#     number_of_preds = preds[0].shape[0]
#     number_of_targets = targets.shape[1]
#     if number_of_preds > 0 and number_of_targets > 1:
#         preds = preds[0][:, :4].cuda() # REMOVE CONFIDENCES AND ONLY KEEP BOUNDING BOX #SHOULD I KEEP CLASS?
#         targets = targets.squeeze(0)[:, 2:] #REMOVE CONFIDENCES AND ONLY KEEP BOUNDING BOX #SHOULD I KEEP CLASS?
#         ious = jaccard(preds,targets).cuda() #ious of objects shown in 2D tensor. size of tensor: [n_preds, n_targets]
#         indicies = (ious > 0.3).nonzero().cuda() #filter and find indices of IOUs that can be considered "good match"
#         matched_ious = ious[indicies[:,0], indicies[:,1]].cuda() #retrieve filtered IOU values by index
#         target_ones = torch.ones(number_of_targets).cuda()
#         pad_number = target_ones.shape[0] - matched_ious.shape[0]
#         if pad_number > 0:
#             '''
#             e.g.
#             n_targets = 5
#             n_matched_predictions = 3
#                 2 targets were not predicted
#             matched_predictions = [0.53, 0.32, 0.69]
#             target_ones = [1, 1, 1, 1, 1]
#
#
#
#             '''
#             matched_ious = torch.cat([matched_ious,
#                                     torch.zeros(pad_number).cuda()])
#         elif pad_number < 0:
#             pad_number = abs(pad_number)
#             target_ones = torch.cat([target_ones,
#                                       torch.zeros(pad_number).cuda()])
#         return torch.mean(torch.abs(target_ones-matched_ious))
#
#
#
#         #return match(0.5, preds, targets)
#     else:
#         return None
#
#
#
#
# def custom_reward(preds, targets):
#     number_of_preds = preds[0].shape[0]
#     print (preds)
#
#     #print (targets.shape)
#     preds = preds[0]
#     number_of_targets = targets.shape[1]
#     if not number_of_preds == number_of_targets:
#         return 0
#     else:
#         if number_of_preds > 0:
#             return torch.mean(preds[:,4])
#         else:
#             return 0
#
# def custom_loss(preds, targets):
#     number_of_preds = preds[0].shape[0]
#     if number_of_preds > 0:
#         preds = preds[0]
#         #print (type(preds), preds.shape)
#         #print ("preds ", preds[:,0:4])
#         targets = targets[:,:, 2:]
#         #print ("targets", targets)
#         number_of_targets = targets.shape[1]
#         if number_of_targets > 0:
#             return (float(number_of_preds/number_of_targets))
#         else:
#             return None
#
#
#
# def custom_reward1(preds, targets):
#     number_of_preds = preds[0].shape[0]
#     #print (targets.shape)
#     preds = preds[0]
#     number_of_targets = targets.shape[1]
#     if not number_of_preds == number_of_targets:
#         return -1
#     else:
#         if number_of_preds > 0:
#             number_of_cyclists_pred = (preds[:, 6]==2).sum(dim=0).cuda()
#             number_of_cyclists_target = (targets[0,:, 1]==2).sum(dim=0).cuda()
#
#             number_of_people_pred = (preds[:, 6] == 0).sum(dim=0).cuda()
#             number_of_people_pred += (preds[:, 6] == 1).sum(dim=0).cuda()
#             number_of_people_target = (targets[0,:, 1] == 0).sum(dim=0).cuda()
#             number_of_people_target += (targets[0,:, 1] == 1).sum(dim=0).cuda()
#
#
#             #rint (number_of_people_pred, number_of_people_target)
#
#             if number_of_cyclists_pred == number_of_cyclists_target and number_of_people_pred == number_of_people_target:
#                 return torch.mean(preds[:,4])
#             else:
#                 return 0
#         else:
#             return 0

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

        number_of_matched_preds = matched_ious.shape[0]

        number_of_non_matched_preds = abs(number_of_preds - matched_ious.shape[0])

        if (number_of_preds == number_of_targets):

            matched_ious = torch.cat([matched_ious,
                                  torch.zeros(number_of_non_matched_preds).cuda()])
        elif (number_of_preds>number_of_targets):
            matched_ious = torch.cat([matched_ious,
                                      torch.ones(number_of_non_matched_preds).cuda()])

        pad_number = target_ones.shape[0] - matched_ious.shape[0]
        #pad_number = target_ones.shape[0] - number_of_preds
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

        return torch.mean(torch.abs(target_ones - matched_ious))

        # return match(0.5, preds, targets)
    elif number_of_preds == 0 and number_of_targets > 0:
        return torch.mean(torch.Tensor([1.0])).cuda()

    elif number_of_preds == 0 and number_of_targets == 0:

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
    GatingNet = AdaptiveFusionGatingNet()
    GatingNet.train()
    GatingNet.cuda()

    #print (GatingNet(torch.tensor([[1,2]]).cuda().float()))

    optimizer = torch.optim.Adam(GatingNet.parameters())
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()


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
    for epoch in range(50):
        cumulative_loss = 0
        counter =0
        print ("Epoch: ", epoch)
        for batch_i, (img_thermal, img_vision, targets, thermal_path, vision_path, shapes) in enumerate(
                tqdm.tqdm(dataloader)):
            if targets.shape[1] == 0:
                continue
            optimizer.zero_grad()

            img = cv2.imread(vision_path[0])
            img = img / 255.0
            img = img.transpose(2, 0, 1)
            # print(img.shape)

            img = torch.from_numpy(img).float()


            img_thermal = img_thermal.to(device).float() / 255.0
            img_vision = img_vision.float() / 255.0
            targets = targets.to(device)
            _, _, height, width = img_thermal.shape

            img_vision_transformed_for_AFM = transform((img))
            img_vision = img_vision.to(device)


            img_vision_transformed_for_AFM = img_vision_transformed_for_AFM.cuda().unsqueeze(0)


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

            gating_output = GatingNet(img_vision_transformed_for_AFM)
            #plt.imshow(img_vision_transformed_for_AFM.numpy())
            #plt.pause(10)


            thermal_values = np.arange(0.0, 1.05, 0.25)
           #print (thermal_values.shape)
            best_thermal_val = None
            best_vision_val = None
            ratio_dict = np.empty(5)
            for index, thermal_val in np.ndenumerate(thermal_values):
                vision_val = 1 - thermal_val
                output = [GraphClique.fusion_graph(np_thermal_pred, np_vision_pred, thermal_val, vision_val)]

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

                print("path: ", vision_path)


                best_vision_val = round(best_vision_val)
                target = torch.Tensor([[best_vision_val]]).cuda().float()
                print("Network output: ", gating_output)
                gating_output = gating_output.float()
                print("Active learning target: ", target)
                full_loss = criterion(gating_output, target)
                full_loss.backward()
                optimizer.step()
                cumulative_loss += full_loss.item()
                counter +=1

                #print("Network loss: ", full_loss)


                # if best_vision_val > 0.5:
                #     optimizer.zero_grad()
                #     gating_output = GatingNet(img_vision_transformed_for_AFM)
                #     loss = criterion(gating_output, target)
                #     loss.backward()
                #     optimizer.step()
                #     print ("Double step.")

        print ("average loss: ", cumulative_loss/counter)
        torch.save(GatingNet.state_dict(), "GatingNet.pt")





            #print (np_thermal_pred)








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
#python ./train_adaptive_combined_gating_network.py --vision_source D:/KAIST_Dataset_FOG/val/images/visible_all --thermal_source D:/KAIST_Dataset_FOG/val/images/lwir_all --thermal_cfg cfg/yolov3-kaist-thermal.cfg --vision_cfg cfg/yolov3-kaist-visible.cfg --thermal_data D:/KAIST_Dataset_FOG/data/kaist_thermal.data --vision_data D:/KAIST_Dataset_FOG/data/kaist_visible.data --thermal_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/lwir/best.pt --vision_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/visible_not_trained_with_fog/best.pt
