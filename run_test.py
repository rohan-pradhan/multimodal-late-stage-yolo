import os
import numpy as np
import shutil


og_results_folder = "C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/results"

test_strings = [
"python ./test_multimodal.py --vision_source D:/KAIST_Dataset_FOG/val/images/visible_all --thermal_source D:/KAIST_Dataset_FOG/val/images/lwir_all --dt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/detections --gt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/groundtruths --thermal_cfg cfg/yolov3-kaist-thermal.cfg --vision_cfg cfg/yolov3-kaist-visible.cfg --thermal_data D:/KAIST_Dataset/data/kaist_thermal.data --vision_data D:/KAIST_Dataset/data/kaist_visible.data --thermal_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/lwir/best.pt --vision_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/visible/best.pt --model_arch ",
"python ./test_multimodal.py --vision_source D:/KAIST_Dataset_FOG/val/images/visible --thermal_source D:/KAIST_Dataset_FOG/val/images/lwir --dt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/detections --gt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/groundtruths --thermal_cfg cfg/yolov3-kaist-thermal.cfg --vision_cfg cfg/yolov3-kaist-visible.cfg --thermal_data D:/KAIST_Dataset/data/kaist_thermal.data --vision_data D:/KAIST_Dataset/data/kaist_visible.data --thermal_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/lwir/best.pt --vision_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/visible/best.pt --model_arch ",
"python ./test_multimodal.py --vision_source D:/KAIST_Dataset_FOG/val/images/visible_fog --thermal_source D:/KAIST_Dataset_FOG/val/images/lwir_fog --dt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/detections --gt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/groundtruths --thermal_cfg cfg/yolov3-kaist-thermal.cfg --vision_cfg cfg/yolov3-kaist-visible.cfg --thermal_data D:/KAIST_Dataset/data/kaist_thermal.data --vision_data D:/KAIST_Dataset/data/kaist_visible.data --thermal_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/lwir/best.pt --vision_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/visible/best.pt --model_arch ",
]

models = [
    "vision_only",
    "thermal_only",
    "average",
    "adaptive"
]

for model in models:
    print ("Testing model arch: ", model)
    for index, run_test_string in enumerate(test_strings):
        run_test_string += model
        print (run_test_string)
        os.system(run_test_string)
        #
        run_map_string = "python C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/pascalvoc.py -np"
        #run_map_string += "--model_arch "
        #run_map_string += model
        os.system(run_map_string)

        new_results_folder = og_results_folder + "_" + model + "_" + str(index)
        if not os.path.exists(new_results_folder):
            os.rename(og_results_folder, new_results_folder)
        if os.path.exists(og_results_folder):
            shutil.rmtree(og_results_folder)
        os.mkdir(og_results_folder)


