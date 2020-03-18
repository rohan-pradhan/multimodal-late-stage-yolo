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


def test(save_txt=False, save_img=False):
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
    dataset = LoadMultimodalImages(thermal_path=thermal_source,
                                   vision_path=vision_source,
                                   img_size=416,
                                   half=opt.half)

    dataset = LoadMultimodalImagesAndLabels(thermal_path=thermal_source,
                                            vision_path=vision_source,
                                            img_size=416)
    # dataset_thermal = LoadImages(thermal_source, img_size=img_size, half=half)
    # dataset_vision = LoadImages(vision_source, img_size=img_size, half=half)

    # Get classes and colors
    thermal_classes = load_classes(parse_data_cfg(opt.thermal_data)['names'])
    vision_classes = load_classes(parse_data_cfg(opt.vision_data)['names'])
    thermal_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(thermal_classes))]
    vision_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(vision_classes))]

    # zipped_dataset = zip(dataset_thermal, dataset_vision)

    # #print(zipped_dataset)
    # x = [[1],[2], [3]]
    # y = [[4],[5], [6]]
    # zipa = zip(x,y)
    # for x, y in zipa:
    #   # #print(x)
    #  # #print(y)

    # for thermal, vison in zipped_dataset:
    # #print("thermal")
    # # #print(thermal)
    # #print("vison")
    # # #print(vison)

    # #print(dataset_thermal[0])

    # Run inference
    t0 = time.time()
    # img_thermal, img_vision, targets, thermal_path, vision_path, _, _ in dataset:
    jdict, stats, ap, ap_class = [], [], [], []
    seen = 0
    for img_thermal, img_vision, targets, thermal_path, vision_path,img0s_thermal, img0s_vision,ratio_pad_tuple in dataset:
        t0 = time.time()
        imgs0_vision_cp_targets = np.copy(img0s_vision)
        # plt.imshow(img0s_vision)
        # plt.pause(1)
        # = targets
        #img_vision = np.swapaxes(np.swapaxes(img_vision, 0,2), 0,1)
        print(img_vision.shape)
        print(img0s_vision.shape)

        print (ratio_pad_tuple[1])

        img_thermal = torch.from_numpy(img_thermal).to(device)
        height, width, _ = img0s_thermal.shape

        tbox = xywh2xyxy(targets[:, 2:])
        for det in tbox:
            det[[0,2]] *=640
            det[[1,3]] *= 512
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            print(x1, x2, y1, y2)
            cv2.rectangle(imgs0_vision_cp_targets, (x1,y1), (x2,y2), (0,255,0), 6)
       # plt.imshow(imgs0_vision_cp_targets)
       # plt.pause(10)
        #break
        #print(tbox)
        #tbox[:, :4] = scale_coords(img_vision.shape[1:], tbox[:, :4], img0s_thermal.shape, ratio_pad_tuple[1])

        #print(tbox)

        #tbox[:, [0, 2]] *= width
        #tbox[:, [1, 3]] *= height

        #for det in tbox:






        img_vision = torch.from_numpy(img_vision).to(device)
        # img_vision_1 = img_vision.cpu()
        img_vision_1 = Image.open(vision_path)
        img_vision_transformed_for_rgb_day_detector = transform(img_vision_1)
        #print(img_vision_transformed_for_rgb_day_detector.shape)
        img_vision_transformed_for_rgb_day_detector = img_vision_transformed_for_rgb_day_detector.cuda()
        img_vision_transformed_for_rgb_day_detector = img_vision_transformed_for_rgb_day_detector.unsqueeze(0)
        day_night = int(RGB_DAY_DETECTOR_MODEL(img_vision_transformed_for_rgb_day_detector).round()[0][0].item())
        # #print("Day Night Value: ", day_night)


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


        pred_thermal = [fusion_graph(np_thermal_pred, np_vision_pred, day_night)]




        # Process detections
        for i, det in enumerate(pred_thermal):  # detections per image

            nl = len(targets)
            tcls = targets[:,0].tolist() if nl else []
            seen +=1

            if len(det)==0:

                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            clip_coords(det, (height, width))
            correct = [0] * len(det)

            thermal_save_path = str(Path(thermal_out) / Path(thermal_path).name)
            thermal_targets_save_path = str(Path(thermal_out_targets) / Path(thermal_path).name)
            print("thermal save path: ", thermal_save_path)
            print ("Det 1: ", det)
            det[:, :4] = scale_coords(img_vision.shape[2:], det[:, :4], img0s_thermal.shape)
            print("Det 2: ",  det)

            for *xyxy, conf, _, cls in det:

                 # Add bbox to image
                label = '%s %.2f' % (thermal_classes[int(cls)], conf)
                plot_one_box(xyxy, img0s_vision, label=label, color=thermal_colors[int(cls)])
                print ("Saved image...")


            if nl:
                detected = []
                tcls_tensor = targets[:,0]
                #print (targets)
                tbox = xywh2xyxy(targets[:, 2:])

                tbox[:, [0,2]] *= width
                tbox[:, [1,3]] *= height
                #print("Tbox: ", tbox)
                #print ("detections: ",  det[:, [0,1,2,3]])



                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(det):
                    if len(detected) == nl:
                        break

                    if pcls.item() not in tcls:
                        continue

                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)
                   # print (iou)

                    #If iou > threshold and class is correct mark as correct
                    if iou > 0.5 and m[bi] not in detected:
                        correct[i] = 1
                        detected.append(m[bi])

                stats.append((correct, det[:,4].cpu(), det[:,6], tcls))
                cv2.imwrite(thermal_save_path, img0s_vision)
                cv2.imwrite(thermal_targets_save_path, imgs0_vision_cp_targets)
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=2)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))



    #         p, s, im0 = thermal_path, '', img0s_thermal
    #
    #         thermal_save_path = str(Path(thermal_out) / Path(p).name)
    #         s += '%gx%g ' % img_thermal.shape[2:]  # # #printstring
    #         if det is not None and len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_coords(img_thermal.shape[2:], det[:, :4], im0.shape).round()
    #
    #             # # #printresults
    #             for c in det[:, -1].unique():
    #                 n = (det[:, -1] == c).sum()  # detections per class
    #                 s += '%g %ss, ' % (n, thermal_classes[int(c)])  # add to string
    #
    #             # Write results
    #             for *xyxy, conf, _, cls in det:
    #                 if save_txt:  # Write to file
    #                     with open(thermal_save_path + '.txt', 'a') as file:
    #                         file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
    #
    #                 if save_img or view_img:  # Add bbox to image
    #                     label = '%s %.2f' % (thermal_classes[int(cls)], conf)
    #                     plot_one_box(xyxy, img0s_vision, label=label, color=thermal_colors[int(cls)])
    #
    #         #print('%sDone. (%.3fs)' % (s, time.time() - t))
    #         #
    #         for i, det in enumerate(pred_vision):  # detections per image
    #             # if webcam:  # batch_size >= 1
    #             #     p, s, im0 = path[i], '%g: ' % i, im0s[i]
    #             # else:
    #             p, s, im0 = vision_path, '', imgs0_vision_og
    #             # #print("TYPE ", type(imgs0_vision_og))
    #
    #             vision_save_path = str(Path(vision_out) / Path(p).name)
    #             s += '%gx%g ' % img_vision.shape[2:]  # # #printstring
    #             if det is not None and len(det):
    #                 # Rescale boxes from img_size to im0 size
    #                 det[:, :4] = scale_coords(img_vision.shape[2:], det[:, :4], im0.shape).round()
    #
    #                 # # #printresults
    #                 for c in det[:, -1].unique():
    #                     n = (det[:, -1] == c).sum()  # detections per class
    #                     s += '%g %ss, ' % (n, vision_classes[int(c)])  # add to string
    #
    #                 # Write results
    #                 for *xyxy, conf, _, cls in det:
    #                     if save_txt:  # Write to file
    #                         with open(vision_save_path + '.txt', 'a') as file:
    #                             file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
    #
    #                     if save_img or view_img:  # Add bbox to image
    #                         label = '%s %.2f' % (vision_classes[int(cls)], conf)
    #                         plot_one_box(xyxy, imgs0_vision_og, label=label, color=vision_colors[int(cls)])
    #
    #             #print('%sDone. (%.3fs)' % (s, time.time() - t))
    #
    #                 # Stream results
    #                 # if view_img:
    #                 #     cv2.imshow(p, im0)
    #
    #         # results (image with detections)
    #         if save_img:
    #
    #             cv2.imwrite(thermal_save_path, img0s_vision)
    #             cv2.imwrite(vision_save_path, imgs0_vision_og)
    # #             else:
    # #                 if vid_path != save_path:  # new video
    # #                     vid_path = save_path
    # #                     if isinstance(vid_writer, cv2.VideoWriter):
    # #                         vid_writer.release()  # release previous video writer
    # #
    # #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
    # #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
    # #                 vid_writer.write(im0)
    # #
    # # if save_txt or save_img:
    # #     #print('Results saved to %s' % os.getcwd() + os.sep + out)
    # #     if platform == 'darwin':  # MacOS
    # #         os.system('open ' + out + ' ' + save_path)
    # #
    # #print('Done. (%.3fs)' % (time.time() - t0))


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
        test()
