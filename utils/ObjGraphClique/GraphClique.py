import networkx as nx
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

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


def fusion_graph(thermal_detections, vision_detections, thermal_mult, vision_mult, IOU_match=0.5 ):

    t_G = nx.Graph()
    v_G = nx.Graph()
    counter = 0
    thermal_multiplier = float(thermal_mult)
    vision_multiplier = float(vision_mult)

    #thermal_multiplier = thermal_mult if day_night == 1 else vision_mult
    #vision_multiplier = vision_mult if day_night == 1 else thermal_mult
    # thermal_multiplier = thermal_mult
    # vision_multiplier = vision_mult

    #thermal_multiplier = 1
    #vision_multiplier = 0
    #

    #thermal_multiplier = 0.9 if day_night == 1 else 0.1
    #vision_multiplier = 0.1 if day_night == 1 else 0.9

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
