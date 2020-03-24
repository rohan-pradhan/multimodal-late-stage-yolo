import os
import cv2
import json
import shutil
from PIL import Image
import numpy as np

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

        os.mkdir(self.train_thermal_dir_adjusted)
        os.mkdir(self.val_thermal_dir_adjusted)

        os.mkdir(self.train_rgb_dir_adjusted)
        os.mkdir(self.val_rgb_dir_adjusted)

        self.train_annotations_json_path = os.path.join(self.train_dir, "thermal_annotations.json")
        self.val_annotations_json_path = os.path.join(self.val_dir, "thermal_annotations.json")

        self.train_annotations_data = self.load_json_annotations(self.train_annotations_json_path)
        self.val_annotations_data = self.load_json_annotations(self.val_annotations_json_path)

        self.train_labels_dir = os.path.join(self.train_dir, "labels")
        self.val_labels_dir = os.path.join(self.val_dir, "labels")

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







flir = FLIR_ADAS(root_dir="D:/FLIR")
#flir.update_file_names()
flir.remove_wrong_sizes()
flir.remove_unmatched_thermal_files()
flir.register_images()
