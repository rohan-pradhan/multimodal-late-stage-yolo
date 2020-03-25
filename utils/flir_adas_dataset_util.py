import os
import cv2
import json
import shutil
from PIL import Image
import numpy as np
from collections import defaultdict

class FLIR_ADAS:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train_dir = os.path.join(root_dir, "train")
        self.val_dir = os.path.join(root_dir, "val")

        self.train_rgb_dir = os.path.join(self.train_dir, "RGB")
        self.train_thermal_dir = os.path.join(self.train_dir, "thermal_8_bit")

        self.val_rgb_dir = os.path.join(self.val_dir, "RGB")
        self.val_thermal_dir = os.path.join(self.val_dir, "thermal_8_bit")

        self.train_thermal_dir_adjusted = os.path.join(self.train_dir, "thermal_8_bit_adjusted")
        self.val_thermal_dir_adjusted = os.path.join(self.val_dir, "thermal_8_bit_adjusted")

        self.train_rgb_dir_adjusted = os.path.join(self.train_dir, "RGB_adjusted")
        self.val_rgb_dir_adjusted = os.path.join(self.val_dir, "RGB_adjusted")

        #os.mkdir(self.train_thermal_dir_adjusted)
        #os.mkdir(self.val_thermal_dir_adjusted)

        #os.mkdir(self.train_rgb_dir_adjusted)
        #os.mkdir(self.val_rgb_dir_adjusted)

        self.train_annotations_json_path = os.path.join(self.train_dir, "thermal_annotations.json")
        self.val_annotations_json_path = os.path.join(self.val_dir, "thermal_annotations.json")

        self.train_annotations_data = self.load_json_annotations(self.train_annotations_json_path)
        self.val_annotations_data = self.load_json_annotations(self.val_annotations_json_path)

        self.train_labels_dir = os.path.join(self.train_dir, "yolo_labels")
        self.val_labels_dir = os.path.join(self.val_dir, "yolo_labels")

        os.mkdir(self.train_labels_dir)
        os.mkdir(self.val_labels_dir)

        self.homography = np.array([[3.93814117e-01, -5.02045010e-03, -5.58470980e+01],
                                    [3.91106718e-03, 3.97512997e-01, -5.92255578e+01],
                                    [-2.91237496e-06, -4.49315649e-06, 1.00000000e+00]], )

        self.img_size = (640, 512)

    def load_json_annotations(self, path_to_json_file):
        print ("Loading JSON annotations...", end='')
        with open(path_to_json_file) as f:
            data = json.load(f)
        data = data["annotations"]
        print (" Done.")
        return data

    def update_file_names(self):
        print ("Updating file names...", end='')
        def change_flir(dir_path):
            for file in os.listdir(dir_path):
                new_file = file.split(".")
                new_file[0] = str(int(new_file[0].split("_")[1])-1)
                new_file = new_file[0] + "." + new_file[1]
                full_old_file = os.path.join(dir_path, file)
                full_new_file = os.path.join(dir_path, new_file)
                os.rename(full_old_file, full_new_file)
        change_flir(self.train_rgb_dir)
        change_flir(self.train_thermal_dir)
        change_flir(self.val_rgb_dir)
        change_flir(self.val_thermal_dir)
        print(" Done.")

    def remove_wrong_sizes(self):
        print("Copying files into new directories...", end='')
        print (self.train_thermal_dir_adjusted)
        def copy_files(old_dir, new_dir):
            for file in os.listdir(old_dir):
                old_path = os.path.join(old_dir, file)
                new_path = os.path.join(new_dir, file)
                shutil.copy(old_path, new_path)

        copy_files(self.train_thermal_dir, self.train_thermal_dir_adjusted)
        copy_files(self.val_thermal_dir, self.val_thermal_dir_adjusted)
        copy_files(self.train_rgb_dir, self.train_rgb_dir_adjusted)
        copy_files(self.val_rgb_dir, self.val_rgb_dir_adjusted)
        print(" Done.")

        print("Removing files of wrong size... ", end='')
        def remove_files(rgb_dir, thermal_dir):
            for file in os.listdir(rgb_dir):
                full_path_rgb = os.path.join(rgb_dir, file)
                full_path_thermal = os.path.join(thermal_dir, file).split(".")[0] +".jpeg"
                if not os.path.isfile(full_path_rgb):
                    continue
                width = 0
                height = 0
                with Image.open(full_path_rgb) as im:
                    width, height = im.size
                print (width, height)
                if width != 1800 or height != 1600:
                    try:
                        os.remove(full_path_rgb)
                    except:
                        print("Could not remove: ", full_path_rgb)
                    try:
                        os.remove(full_path_thermal)
                    except:
                        print("Could not remove: ", full_path_thermal)
        remove_files(self.train_rgb_dir_adjusted, self.train_thermal_dir_adjusted)
        remove_files(self.val_rgb_dir_adjusted, self.val_thermal_dir_adjusted)
        print(" Done. ")


    def remove_unmatched_thermal_files(self):
        print("Removing unmatched thermal files...", end='')
        def remove_files(rgb_dir, thermal_dir):
            for file in os.listdir(thermal_dir):
                #print(file)
                full_path_thermal = os.path.join(thermal_dir, file)
                full_path_rgb = os.path.join(rgb_dir, file).split(".")[0] + ".jpg"

                if not os.path.isfile(full_path_rgb):
                    #print (full_path_thermal)
                    os.remove(full_path_thermal)

        remove_files(self.train_rgb_dir_adjusted, self.train_thermal_dir_adjusted)
        remove_files(self.val_rgb_dir_adjusted, self.val_thermal_dir_adjusted)
        print(" Done.")


    def register_images(self):
        print("Registering RGB Images...", end='')
        def register_image(dir):
            for file in os.listdir(dir):
                if file.endswith(".jpg"):
                    full_path = os.path.join(dir, file)
                    image = cv2.imread(full_path)
                    try:
                        projected_img = cv2.warpPerspective(image, self.homography, self.img_size)
                        cv2.imwrite(full_path, projected_img)
                    except:
                        continue
        print("Training images... ", end='')
        print("Val images...", end='')
        register_image(self.train_rgb_dir_adjusted)
        register_image(self.val_rgb_dir_adjusted)
        print(" Done.")

    def create_label_files(self):
        #1 -> 0
        #3 -> 1
        #Delete all other numbers

        def change_category_id(category_id):
            if (category_id == 1):
                category_id = 0
            elif (category_id == 3):
                category_id = 1
            else:
                category_id = -1
            return category_id

        def convert_bbox(bbox):
            x_top_left = bbox[0]
            y_top_left = bbox[1]
            width = bbox[2]
            height = bbox[3]

            x_center = x_top_left + (width/2.0)
            y_center = y_top_left + (height/2.0)
            bbox = [x_center, y_center, width, height]

            img_width = float(self.img_size[0])
            img_height = float(self.img_size[1])

            bbox[0] /= img_width
            bbox[2] /= img_width
            bbox[1] /= img_height
            bbox[3] /= img_height

            return bbox

        def create_labels(data, directory):
            annotations_list = defaultdict(list)
            for annotation in data:
                image_id = annotation['image_id']
                category_id  = change_category_id(annotation['category_id'])
                if category_id == -1:
                    continue
                bbox = convert_bbox(annotation['bbox'])
                tuple_to_add = (category_id, bbox)
                annotations_list[image_id].append(tuple_to_add)

            for key in annotations_list:
                file_name = str(key)+".txt"
                file_path = os.path.join(directory, file_name)

                with open(file_path, 'a+') as f:
                    for tuple in annotations_list[key]:
                        category_id = str(tuple[0])
                        x_center = str(tuple[1][0])
                        y_center = str(tuple[1][1])
                        width = str(tuple[1][2])
                        height = str(tuple[1][3])
                        str_to_write = " ".join([category_id, x_center, y_center, width, height, "\n"])
                        f.write(str_to_write)

        create_labels(self.train_annotations_data, self.train_labels_dir)
        create_labels(self.val_annotations_data, self.val_labels_dir)













flir = FLIR_ADAS(root_dir="D:/FLIR")
#flir.update_file_names()
#flir.remove_wrong_sizes()
#flir.remove_unmatched_thermal_files()
#flir.register_images()
flir.create_label_files()
