import argparse

import cv2

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from utils.DayNightDetect import Net
import networkx as nx
import torchvision.transforms as transforms


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # # #print("box a", boxA[0], "boxb", boxB[0])
    xA = max(boxA[0], boxB[0])
    # #print("gotem")
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


def fusion_graph(thermal_detections, vision_detections, day_night, IOU_match=0.5):
    #t0 = time.time()
    t_G = nx.Graph()
    v_G = nx.Graph()
    counter = 0
    thermal_multiplier = 0.75 if day_night == 1 else 0.25
    vision_multiplier = 0.25 if day_night == 1 else 0.75
    #print("Thermal Multiplier: ", thermal_multiplier, "| Vision Multiplier: ", vision_multiplier)

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

        if vision_box[6] == 2:
            vision_box[6] = 1

            v_G.add_node(counter, data=vision_box)
            counter += 1
        elif vision_box[6] == 0:
            v_G.add_node(counter, data=vision_box)
            counter += 1

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
        # # #print(node)
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
            # det_c = [(det_a[0] + det_b[0]) / 2,
            #          (det_a[1] + det_b[1]) / 2,
            #          (det_a[2] + det_b[2]) / 2,
            #          (det_a[3] + det_b[3]) / 2,
            #          (det_a[4] + det_b[4]) / 2,
            #          (det_a[5] + det_b[5]) / 2,
            #          det_a[6]]
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
            if det_c[4] > 0.4: detections_to_return.append(det_c)

    return torch.FloatTensor(detections_to_return)


def test(save_txt=False,
         save_img=False,
         dataloader = None):
    RGB_DAY_DETECTOR_MODEL = Net()
    RGB_DAY_DETECTOR_MODEL.load_state_dict(torch.load("../RGB_Day_Night_detector/day_detector.pt"))
    RGB_DAY_DETECTOR_MODEL.cuda()
    RGB_DAY_DETECTOR_MODEL.eval()
    transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),transforms.ToTensor()])



    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    thermal_out, vision_out, thermal_source, vision_source, thermal_weights, vision_weights, half, view_img = opt.thermal_out, \
                                                                                                              opt.vision_out, \
                                                                                                              opt.thermal_source, \
                                                                                                              opt.vision_source, \
                                                                                                              opt.thermal_weights, \
                                                                                                              opt.vision_weights, \
                                                                                                              opt.half, \
                                                                                                              opt.view_img
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)

    thermal_out_targets = thermal_out+"_targets"

    if os.path.exists(thermal_out):
        shutil.rmtree(thermal_out)  # delete output folder
    os.makedirs(thermal_out)  # make new output folder

    if os.path.exists(thermal_out_targets):
        shutil.rmtree(thermal_out_targets)  # delete output folder
    os.makedirs(thermal_out_targets)  # make new output folder

    if os.path.exists(vision_out):
        shutil.rmtree(vision_out)  # delete output folder
    os.makedirs(vision_out)  # make new output folder

    # Initialize model
    model_thermal = Darknet(opt.thermal_cfg, img_size)
    model_vision = Darknet(opt.vision_cfg, img_size)

    # Load weights
    # attempt_download(weights)
    if thermal_weights.endswith('.pt'):  # pytorch format
        model_thermal.load_state_dict(torch.load(thermal_weights, map_location=device)['model'])

    if vision_weights.endswith('.pt'):
        model_vision.load_state_dict(torch.load(vision_weights, map_location=device)['model'])

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    #     modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model_thermal.to(device).eval()
    model_vision.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=10)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        #print(onnx.helper.#printable_graph(model.graph))  # # #printa human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model_thermal.half()
        model_vision.half()

    # Set Dataloader
    vid_path, vid_writer = None, None

    save_img = True



    if dataloader is None:
        dataset = LoadMultimodalImagesAndLabels(thermal_path=thermal_source, vision_path=vision_source, img_size=416)
        batch_size = 1
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

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
    for batch_i, (img_thermal, img_vision, targets, thermal_path, vision_path, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img_thermal = img_thermal.to(device).float() / 255.0
        img_vision = img_vision.to(device).float() / 255.0
        targets = targets.to(device)
        _, _, height, width = img_thermal.shape

        img_vision_transformed_for_rgb_day_detector = transform(img_vision)
        img_vision_transformed_for_rgb_day_detector = img_vision_transformed_for_rgb_day_detector.cuda()
        img_vision_transformed_for_rgb_day_detector = img_vision_transformed_for_rgb_day_detector.unsqueeze(0)
        day_night = int(RGB_DAY_DETECTOR_MODEL(img_vision_transformed_for_rgb_day_detector).round()[0][0].item())

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


        output = [fusion_graph(np_thermal_pred, np_vision_pred, day_night)]




        # Process detections
        for si, pred in enumerate(output):  # detections per image
            labels = targets[targets[:,0] == si, 1:]
            nl = len(targets)
            tcls = targets[:,0].tolist() if nl else []
            seen +=1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(d[6])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})

                # Clip boxes to image bounds
                clip_coords(pred, (height, width))

                # Assign all predictions as incorrect
                correct = [0] * len(pred)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= width
                    tbox[:, [1, 3]] *= height

                    # Search for correct predictions
                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                        # Break if all targets already located in image
                        if len(detected) == nl:
                            break

                        # Continue if predicted class not among image classes
                        if pcls.item() not in tcls:
                            continue

                        # Best iou, index between pred and targets
                        m = (pcls == tcls_tensor).nonzero().view(-1)
                        iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                        # If iou > threshold and class is correct mark as correct
                        if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                            correct[i] = 1
                            detected.append(m[bi])

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%10.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

        # Print results per class
        if verbose and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

        return (mp, mr, map,  mf1, *(loss / len(dataloader)).tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal_cfg', type=str, default='cfg/yolov3-custom-thermal-1.cfg',
                        help='thermal cfg file path')
    parser.add_argument('--vision_cfg', type=str, default='cfg/yolov3-spp.cfg', help='vison cfg file path')
    parser.add_argument('--thermal_data', type=str, default='data/coco1.data', help='thermal coco.data file path')
    parser.add_argument('--vision_data', type=str, default='data/coco.data', help='vison coco.data file path')
    parser.add_argument('--thermal_weights', type=str, default='weights/best.pt', help='path to thermal weights file')
    parser.add_argument('--vision_weights', type=str, default='weights/ultralytics68.pt',
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
    #print(opt)

    with torch.no_grad():
        print(test())
