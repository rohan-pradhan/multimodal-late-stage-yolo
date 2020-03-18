import argparse

import torch.nn as nn
import torch
import numpy
from torch import optim

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import networkx as nx


class FusionWeightsNet(nn.Module):
    def __init__(self):
        super(FusionWeightsNet, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(12, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(20, 30, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4)
        )
        self.fc1 = nn.Linear(900,800)
        self.fc2 = nn.Linear(800, 2)

    def forward(self, x_vision, x_thermal):
        x = torch.cat((x_vision, x_thermal),1)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.reshape(x.size(0), -1) #flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x)

# class fusion_loss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, preds, targets):
#         preds = preds[0]
#         print("hello world")
#         print ("Preds: ", preds, " Targets: ", targets)
#         loss = 0
#         bbox_loss = nn.SmoothL1Loss()
#         # conf_loss = nn.NLLLoss()
#         for pred in preds:
#             # partial_loss = 0
#             matched = False
#             for target in targets:
#                 iou = bb_intersection_over_union(pred[0:4], target[0:4])
#                 if (iou > 0.5 and pred[6] == target[6]):
#                     matched = True
#                     reg_loss = bbox_loss(pred[0:4], target[0:4])
#                     conf = torch.neg(torch.log((pred[4])))
#                     loss += reg_loss + conf
#                     break
#             if not matched:
#                 loss += torch.neg(torch.log(torch.neg(pred[4])+1))
#         print (loss)
#         return (loss)




def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
   # print (boxA, boxB)
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



def fusion_graph(thermal_detections, vision_detections, thermal_conf, vision_conf):
    t0 = time.time()
    t_G = nx.Graph()
    v_G = nx.Graph()
    counter = 0
    #print ("Thermal Detections length: ", len(thermal_detections))
    #print ("Vision Detections length: ", len(vision_detections))
    for det in thermal_detections:
        #print(det)
        thermal_box = [det[0],
                       det[1],
                       det[2],
                       det[3],
                       det[4],
                       det[5],
                       det[6],
                       True]
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
                      False]
        #print(vision_box)
        if vision_box[6] == 0 or vision_box[6] == 1:
            #print("true")
            v_G.add_node(counter, data=vision_box)
            counter += 1
    #print("Thermal nodes: ", t_G.nodes())
    #print("Vision nodes: ", v_G.nodes())
    G = nx.compose(t_G, v_G)
    for v_node in v_G.nodes(data=True):
        v_data = v_node[1]['data'][0:4]
        # print(type(v_node[0]))
        for t_node in t_G.nodes(data=True):
            t_data = t_node[1]['data'][0:4]
            iou = bb_intersection_over_union(v_data, t_data)
            if (iou > 0.35): #checks if there is good amount overlap between bounding boxes -> this is a hyperparameter
                G.add_edge(v_node[0], t_node[0], weight=iou)

    #Removes edges from graph to ensure that each node only is connected to one other node from the other modality
    list_to_remove = []
    for node in G.nodes():
        edge_list = G.edges(node)
        edge_dict = {}
        max_weight = 0
        index = -1
        counter = 0
        # print (node)
        for x, y in (edge_list):
            weight = G.get_edge_data(x, y)['weight']
            if weight > max_weight:
                max_weight = weight
                index = counter
            #print(G.get_edge_data(x, y))
            counter += 1
        counter = 0
        for x, y in (edge_list):
            if counter == index:
                counter += 1
                continue
            else:
                list_to_remove.append((x, y))
                # G.remove_edge(x,y)
                #print("Edge removed", x, y)
                counter += 1
    #print("list to remove: ", list_to_remove)
    G.remove_edges_from(list_to_remove)

    detections_to_return = []
    nodes_visited = []
    for node in G.nodes(data=True):
        if (node[0] in nodes_visited):
            continue

        if (len(G.edges(node[0])) == 0):
            det = node[1]['data']
            if det[7] == True:
                det[4] = det[4]*thermal_conf
            else:
                det[4] = det[4]*vision_conf
            det = [det[0],
                   det[1],
                   det[2],
                   det[3],
                   det[4],
                   det[6]]
            detections_to_return.append(det)
        else:
            det_a = node[1]['data']
            #print (type(G.edges(node[0])))
            #print ((list(G.edges(node[0]))[0][1]))
            adj_node_index = list(G.edges(node[0]))[0][1]
            #print ("adj node index: ", adj_node_index)
            adj_node = G.nodes[adj_node_index]
            #print (adj_node)
            det_b = adj_node['data']
            x = 0
            y = 0

            if det_a[7] == True:
                x = thermal_conf
                y = vision_conf
            else:
                x = vision_conf
                y = thermal_conf

            det_c = [(x*det_a[0]+y*det_b[0]),
                     (x*det_a[1] + y*det_b[1]),
                     (x*det_a[2] + y*det_b[2]),
                     (x*det_a[3] + y*det_b[3]),
                     (x*det_a[4] + y*det_b[4]),
                     (det_a[6])]


            nodes_visited.append(adj_node_index)
            detections_to_return.append(det_c)
       # print(G.edges(node[0]))
    #print ("DETECTIONS TO RETURN :", len(detections_to_return))
    #print("Graph time: ", time.time() - t0)
    return torch.FloatTensor(detections_to_return)


class fusion_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds = preds[0]
        targets = targets[0]
        #print ("Preds: ", preds, " Targets: ", targets)
        loss =
        bbox_loss = nn.SmoothL1Loss()
        # conf_loss = nn.NLLLoss()
        for pred in preds:

            # partial_loss = 0
            matched = False
            for target in targets:
                iou = bb_intersection_over_union(pred[0:4].numpy(), target[0:4].numpy())
                if (iou > 0.5 and pred[6] == target[6]):
                    matched = True
                    reg_loss = bbox_loss(pred[0:4], target[0:4])
                    conf = torch.neg(torch.log((pred[4])))
                    loss += reg_loss + conf
                    break
            if not matched:
                loss += torch.neg(torch.log(torch.neg(pred[4])+1))
        return torch.tensor(loss, dtype=torch.float)


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
                                             collate_fn=None)
    fusion_model = FusionWeightsNet()
    fusion_model.cuda()
    fusion_model.train()
    nb = len(dataloader)
    optimizer = optim.SGD(fusion_model.parameters(), lr=0.1)
    loss_fn = fusion_loss()

    for epoch in range(start_epochs, epochs):
        epoch_loss = 0
        count = 0
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (thermal_img, vision_img, targets, thermal_path, vision_path, _) in pbar:
            count +=1
            thermal_pred = model_thermal(thermal_img.to(device=device, dtype=torch.float))[0]
            vision_pred = model_vision(vision_img.to(device=device, dtype=torch.float))[0]

            thermal_pred = non_max_suppression(thermal_pred, opt.conf_thres, opt.nms_thres)
            vision_pred = non_max_suppression(vision_pred, opt.conf_thres, opt.nms_thres)

            input_tensor = torch.cat((thermal_img, vision_img), 1)
            outputs = fusion_model(x_vision= vision_img.cuda(), x_thermal = thermal_img.cuda())
            # print(outputs[0].grad)
            # print("outputs: ", outputs)
            thermal_pred_input = thermal_pred[0].cpu().clone().numpy() if thermal_pred[0] is not None else []
            vision_pred_input = vision_pred[0].cpu().clone().numpy() if vision_pred[0] is not None else []
            fused_preds = [fusion_graph(thermal_pred_input, vision_pred_input, outputs[0][0], outputs[0][1])]
            loss = loss_fn(fused_preds, targets)
            epoch_loss += loss
            loss.retain_grad()
            loss = torch.autograd.Variable(loss, requires_grad=True)
            print(loss.is_leaf)
            print(loss.grad)
            loss.backward()

            optimizer.step()
            # print("Loss: ", loss)

        print ("Average epoch loss: ", float(epoch_loss/count))













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
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.1, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    train_loop(opt.thermal_source, vision_source=opt.vision_source)
