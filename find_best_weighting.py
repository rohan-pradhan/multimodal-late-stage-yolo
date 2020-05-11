import os
import numpy as np
import shutil


thermal_values = np.arange(0.0, 1.05, 0.05)
thermal_values = thermal_values.tolist()
og_results_folder = "C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/results"
day_night_test = [0]
for y in day_night_test:
    #for x in thermal_values:
    for x in day_night_test:

        run_test_string = "python ./test_multimodal.py --vision_source D:/KAIST_Dataset/val/images/visible --thermal_source D:/KAIST_Dataset/val/images/lwir --dt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/detections --gt_path C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/groundtruths --thermal_cfg cfg/yolov3-kaist-thermal.cfg --vision_cfg cfg/yolov3-kaist-visible.cfg --thermal_data D:/KAIST_Dataset/data/kaist_thermal.data --vision_data D:/KAIST_Dataset/data/kaist_visible.data --thermal_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/lwir/best.pt --vision_weights D:/Thesis/saved_models/yolov3/KAIST_pedestrian/visible/best.pt"
        thermal_value = str(x)
        vision_value = str(1.0-x)
        day_night_value = str(y)
        print ("Running expeirment with: thermal value = ", thermal_value, " | vision value = ", vision_value, " | testing on: ", day_night_value)
        print ("---------------------")
        print (" ")
        run_test_string += " --thermal_multiplier "
        run_test_string += thermal_value
        run_test_string += " --vision_multiplier "
        run_test_string += vision_value
        run_test_string += " --day_night "
        run_test_string += day_night_value
        print (run_test_string)
        os.system(run_test_string)
        #
        run_map_string = "python C:/Users/Rohan/Documents/Development/Thesis/Experimentation/Object-Detection-Metrics/pascalvoc.py -np"

        os.system(run_map_string)

        new_results_folder = og_results_folder + "_thermal_val_" + str(int(x*100)) + "_day_val_" + day_night_value
        if not os.path.exists(new_results_folder):
            os.rename(og_results_folder, new_results_folder)
        if os.path.exists(og_results_folder):
            shutil.rmtree(og_results_folder)
        os.mkdir(og_results_folder)


