import argparse

import cv2

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.AdaptiveFusionGating.AdaptiveFusionGatingNet import AdaptiveFusionGatingNet
#from utils.RGB_Day_Night_Detector_updated.DayNightDetect import Net
import networkx as nx
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from utils.ObjGraphClique import GraphClique



CONST_NIGHT_ONLY = -1
CONST_DAY_ONLY = 1
CONST_BOTH = 0


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


def fusion_graph(thermal_detections, vision_detections, day_night, thermal_mult, vision_mult, IOU_match=0.5 ):

    t_G = nx.Graph()
    v_G = nx.Graph()
    counter = 0
    #thermal_mult = float(thermal_mult)
    #vision_mult = float(vision_mult)

    #thermal_multiplier = thermal_mult if day_night == 1 else vision_mult
    #vision_multiplier = vision_mult if day_night == 1 else thermal_mult
    # thermal_multiplier = thermal_mult
    # vision_multiplier = vision_mult

    #thermal_multiplier = 1
    #vision_multiplier = 0
    #

    thermal_multiplier = 0.9 if day_night == 1 else 0.1
    vision_multiplier = 0.1 if day_night == 1 else 0.9

    for det in thermal_detections:
        thermal_box = [det[0],
                       det[1],
                       det[2],
                       det[3],
                       det[4],
                       det[5],
                       det[6],
                       1]

        t_G.add_node(counter, data=thermal_box)
        counter += 1
    for det in vision_detections:
        vision_box = [det[0],
                      det[1],
                      det[2],
                      det[3],
                      det[4],
                      det[5],
                      det[6],
                      0]
        v_G.add_node(counter, data=vision_box)
        counter +=1


    G = nx.compose(t_G, v_G)
    for v_node in v_G.nodes(data=True):
        v_data = v_node[1]['data'][0:4]

        for t_node in t_G.nodes(data=True):
            t_data = t_node[1]['data'][0:4]
            iou = bb_intersection_over_union(v_data, t_data)

            if (iou >= IOU_match): #checks if there is good amount overlap between bounding boxes -> this is a hyperparameter
                G.add_edge(v_node[0], t_node[0], weight=iou)

    list_to_remove = []
    for node in G.nodes():
        edge_list = G.edges(node)
        max_weight = 0
        index = -1
        counter = 0

        for x, y in (edge_list):
            weight = G.get_edge_data(x, y)['weight']
            if weight > max_weight:
                max_weight = weight
                index = counter
            counter += 1
        counter = 0
        for x, y in (edge_list):
            if counter == index:
                counter += 1
                continue
            else:
                list_to_remove.append((x, y))

    G.remove_edges_from(list_to_remove)

    detections_to_return = []
    nodes_visited = []
    for node in G.nodes(data=True):
        if (node[0] in nodes_visited):
            continue

        if (len(G.edges(node[0])) == 0):
            det = node[1]['data']

            if det[7] == 1:

                det[4] = det[4]*thermal_multiplier

            else:

                det[4] = det[4]*vision_multiplier

            det = det[0:7]

            if det[4] > 0.4: detections_to_return.append(det)
        else:

            det_a = node[1]['data']
            det_a_thermal = None
            if det_a[7] == 1:
                det_a_thermal = True
                #det a is thermal
            else:
                det_a_thermal = False
                #det a is vision

            adj_node_index = list(G.edges(node[0]))[0][1]

            adj_node = G.nodes[adj_node_index]

            det_b = adj_node['data']

            if (det_a_thermal):
                #det_a is a thermal image

                det_c = [(thermal_multiplier*det_a[0] + vision_multiplier*det_b[0]),
                         (thermal_multiplier*det_a[1] + vision_multiplier*det_b[1]),
                         (thermal_multiplier*det_a[2] + vision_multiplier*det_b[2]),
                         (thermal_multiplier*det_a[3] + vision_multiplier*det_b[3]),
                         (thermal_multiplier*det_a[4] + vision_multiplier*det_b[4]),
                         (thermal_multiplier*det_a[5] + vision_multiplier*det_b[5]),
                         det_a[6]]

            elif (not det_a_thermal):
                #det_a is a vision image
                det_c = [(vision_multiplier*det_a[0] + thermal_multiplier*det_b[0]),
                         (vision_multiplier*det_a[1] + thermal_multiplier*det_b[1]),
                         (vision_multiplier*det_a[2] + thermal_multiplier*det_b[2]),
                         (vision_multiplier*det_a[3] + thermal_multiplier*det_b[3]),
                         (vision_multiplier*det_a[4] + thermal_multiplier*det_b[4]),
                         (vision_multiplier*det_a[5] + thermal_multiplier*det_b[5]),
                         det_a[6]]



            nodes_visited.append(adj_node_index)
            if det_c[4] >= 0.4: detections_to_return.append(det_c)

    return torch.FloatTensor(detections_to_return)


def test(dt_path,
         gt_path,
         model_arch):
    model_arch = model_arch
    print("Testing model with ", model_arch)

    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])



    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
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

    if os.path.exists(dt_path):
        shutil.rmtree(dt_path)
    os.mkdir(dt_path)

    if os.path.exists(gt_path):
        shutil.rmtree(gt_path)
    os.mkdir(gt_path)

    # Initialize model
    model_thermal = Darknet(opt.thermal_cfg, img_size)
    model_vision = Darknet(opt.vision_cfg, img_size)

    # Load weights
    # attempt_download(weights)
    if thermal_weights.endswith('.pt'):  # pytorch format
        model_thermal.load_state_dict(torch.load(thermal_weights, map_location=device)['model'])

    if vision_weights.endswith('.pt'):
        model_vision.load_state_dict(torch.load(vision_weights, map_location=device)['model'])


    # Eval mode
    model_thermal.to(device).eval()
    model_vision.to(device).eval()


    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model_thermal.half()
        model_vision.half()

    # Set Dataloader
    vid_path, vid_writer = None, None

    save_img = True


    GatingNet = AdaptiveFusionGatingNet()
    GatingNet.eval()
    GatingNet.cuda()
    GatingNet.load_state_dict(torch.load("./GatingNet4.pt"))





    dataset = LoadMultimodalImagesAndLabels(thermal_path=thermal_source, vision_path=vision_source, img_size=416)


    dataloader = DataLoader(dataset,
                            pin_memory=True)

    # Get classes and colors
    thermal_classes = load_classes(parse_data_cfg(opt.thermal_data)['names'])
    vision_classes = load_classes(parse_data_cfg(opt.vision_data)['names'])
    thermal_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(thermal_classes))]
    vision_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(vision_classes))]


    # Run inference
    t0 = time.time()
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    seen = 0

    counter = 0
    for batch_i, (img_thermal, img_vision, targets, thermal_path, vision_path, shapes) in enumerate(tqdm.tqdm(dataloader)):
        #print (targets)
        img_thermal = img_thermal.to(device).float() / 255.0
        img_vision = img_vision.float() / 255.0
        plt.imshow(np.moveaxis(img_vision[0].numpy(), 0,2))
        plt.pause(10)

        targets = targets.to(device)
        _, _, height, width = img_thermal.shape

        img_vision_AFM = Image.open(vision_path[0]).convert('RGB')
        img_lwir_AFM = Image.open(thermal_path[0]).convert('RGB')

        img_vision_AFM = transform(img_vision_AFM)
        img_lwir_AFM = transform(img_lwir_AFM)

        img_vision = img_vision.to(device)

        img_vision_AFM = img_vision_AFM.cuda().unsqueeze(0)
        img_lwir_AFM = img_lwir_AFM.cuda().unsqueeze(0)



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
            np_thermal_pred = pred_thermal[0].cpu().numpy()
        except:
            np_thermal_pred = []
        try:
            np_vision_pred = pred_vision[0].cpu().numpy()
        except:
            np_vision_pred = []


        if (model_arch == "thermal_only"):
            output = [GraphClique.fusion_graph(np_thermal_pred, np_vision_pred, 1, 0)]

        elif (model_arch == "vision_only"):
            output = [GraphClique.fusion_graph(np_thermal_pred, np_vision_pred, 0.0, 1)]

        elif (model_arch == "average"):
            output = [GraphClique.fusion_graph(np_thermal_pred, np_vision_pred, 0.5, 0.5)]

        elif (model_arch == "adaptive"):

            gating_output = GatingNet(img_vision_AFM, img_lwir_AFM)


            vision_multiplier = float(gating_output[0][0].item())
            thermal_multiplier = 1. - vision_multiplier

            output = [GraphClique.fusion_graph(np_thermal_pred, np_vision_pred, thermal_multiplier, vision_multiplier)]

        base_name = os.path.basename(thermal_path[0]).split(".")[0] + ".txt"
        full_gt_path = os.path.join(gt_path, base_name)
        full_dt_path = os.path.join(dt_path, base_name)
        output = output[0].squeeze()
        #print(np_vision_pred)

        targets = targets.squeeze()
        #print (full_gt_path, full_dt_path)
        if targets.dim() == 1:
            targets.unsqueeze_(0)
        if output.dim() ==1:
            output.unsqueeze_(0)
        with open(full_dt_path, 'a+') as f:
            for gt in output:
                if len(gt) ==0:
                    continue
                if gt[6] == 0:
                    category = "person"
                elif gt[6] == 1:
                     category = "people"
                elif gt[6] == 2:
                    category = "cyclist"
                x1 = str(gt[0].item())
                y1 = str(gt[1].item())
                x2 = str(gt[2].item())
                y2 = str(gt[3].item())
                conf = str(gt[4].item())
                to_write = [category, conf, x1, y1, x2, y2, "\n"]
                str_to_write = " ".join(to_write)
                f.write(str_to_write)

        with open(full_gt_path, 'a+') as f:
            for gt in targets:
                if gt[1] == 0:
                    category = "person"
                elif gt[1]==1:
                    category = "people"
                elif gt[1]==2:
                    category = "cyclist"
                x1 = str(gt[2].item())
                y1 = str(gt[3].item())
                x2 = str(gt[4].item())
                y2 = str(gt[5].item())
                to_write = [category, x1, y1, x2, y2,"\n"]
                str_to_write = " ".join(to_write)
                f.write(str_to_write)
    print (counter)


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
    parser.add_argument("--model_arch", default="adaptive")
    opt = parser.parse_args()
    #print(opt)

    with torch.no_grad():
        print(test(opt.dt_path, opt.gt_path, opt.model_arch))
        #python test_multimodal.py --vision_source D:/FLIR/val/RGB_adjusted --thermal_source D:/FLIR/val/thermal_8_bit_adjusted

