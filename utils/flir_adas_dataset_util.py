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

        self.make_dir_safe(self.train_thermal_dir_adjusted)
        self.make_dir_safe(self.val_thermal_dir_adjusted)

        self.make_dir_safe(self.train_rgb_dir_adjusted)
        self.make_dir_safe(self.val_rgb_dir_adjusted)

        self.train_annotations_json_path = os.path.join(self.train_dir, "thermal_annotations.json")
        self.val_annotations_json_path = os.path.join(self.val_dir, "thermal_annotations.json")

        self.train_annotations_data = self.load_json_annotations(self.train_annotations_json_path)
        self.val_annotations_data = self.load_json_annotations(self.val_annotations_json_path)

        self.train_labels_dir = os.path.join(self.train_dir, "yolo_labels")
        self.val_labels_dir = os.path.join(self.val_dir, "yolo_labels")

        self.make_dir_safe(self.train_labels_dir)
        self.make_dir_safe(self.val_labels_dir)

        self.homography = np.array([[3.93814117e-01, -5.02045010e-03, -5.58470980e+01],
                                    [3.91106718e-03, 3.97512997e-01, -5.92255578e+01],
                                    [-2.91237496e-06, -4.49315649e-06, 1.00000000e+00]], )

        self.img_size = (640, 512)

    def make_dir_safe(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

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
                full_path_thermal = os.path.join(thermal_dir, file)
                full_path_rgb = os.path.join(rgb_dir, file).split(".")[0] + ".jpg"

                if not os.path.isfile(full_path_rgb):
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
        print("Creating label files...", end='')
        #1 -> 0
        #3 -> 1
        #Delete all other numbers
        # 0 = person
        # 1 = car
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

        def create_labels(data, directory, offset=0):
            annotations_list = defaultdict(list)
            for annotation in data:
                image_id = annotation['image_id'] + offset
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
        create_labels(self.val_annotations_data, self.val_labels_dir, offset=8862)
        print(" Done.")

    def remove_extra_label_fils(self):
        print("Removing exta label files...", end='')
        def remove_files(label_dir, rgb_dir, thermal_dir):
            for label_file in os.listdir(label_dir):
                label_path = os.path.join(label_dir, label_file)
                rgb_file = label_file.split(".")[0] + ".jpg"
                thermal_file =  label_file.split(".")[0] + ".jpeg"
                rgb_path = os.path.join(rgb_dir, rgb_file)
                thermal_path = os.path.join(thermal_dir, thermal_file)

                if not os.path.isfile(rgb_path) or not os.path.isfile(thermal_path):
                    os.remove(label_path)
        remove_files(self.train_labels_dir, self.train_rgb_dir_adjusted, self.train_thermal_dir_adjusted)
        remove_files(self.val_labels_dir, self.val_rgb_dir_adjusted, self.val_thermal_dir_adjusted)
        print(" Done.")
    def plot_annotations(self):
        print("Plotting annotations...", end='')
        train_rgb_annotated_path = os.path.join(self.train_dir, "rgb_annotated")
        train_thermal_annotated_path = os.path.join(self.train_dir, "thermal_annotated")
        val_rgb_annotated_path = os.path.join(self.val_dir, "rgb_annotated")
        val_thermal_annotated_path = os.path.join(self.val_dir, "thermal_annotated")

        self.make_dir_safe(train_thermal_annotated_path)
        self.make_dir_safe(train_rgb_annotated_path)
        self.make_dir_safe(val_thermal_annotated_path)
        self.make_dir_safe(val_rgb_annotated_path)

        def convertxywh2xyxy(x_center, y_center, w, h):

            x1 = (x_center - w/2.0)*self.img_size[0]
            y1 = (y_center - h/2.0)*self.img_size[1]
            x2 = (x_center + w/2.0)*self.img_size[0]
            y2 = (y_center + h/2.0)*self.img_size[1]
            return x1, y1, x2, y2

        def plot_box(img, annotation):
            category_id = annotation[0]
            color = (255, 0, 0) if category_id == 0 else (0, 255, 0)
            x_center = float(annotation[1])
            y_center = float(annotation[2])
            w = float(annotation[3])
            h = float(annotation[4])
            x1, y1, x2, y2 = convertxywh2xyxy(x_center, y_center, w, h)
            c1,c2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, c1, c2, color)
            return img

        def get_annotations_from_file(label_file):
            raw_lines = []
            with open(label_file) as f:
                for line in f:
                    raw_lines.append(line)
            lines = []
            for line in raw_lines:
                line_to_append = line.split(" ")
                lines.append(line_to_append)
            return lines

        def get_img_paths(label_file):
            rgb_file = label_file.replace(".txt", ".jpg")
            thermal_file = label_file.replace(".txt", ".jpeg")
            return rgb_file, thermal_file

        def plot_on_imgs(labels_dir, rgb_input, thermal_input, rgb_output, thermal_output):
            for label_file in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_file)
                labels = get_annotations_from_file(label_path)
                rgb_file, thermal_file = get_img_paths(label_file)
                rgb_path = os.path.join(rgb_input, rgb_file)
                thermal_path = os.path.join(thermal_input, thermal_file)
                rgb_img = cv2.imread(rgb_path)
                thermal_img = cv2.imread(thermal_path)
                for label in labels:
                    rgb_img = plot_box(rgb_img, label)
                    thermal_img = plot_box(thermal_img, label)
                rgb_save_path = os.path.join(rgb_output, rgb_file)
                thermal_save_path = os.path.join(thermal_output, thermal_file)
                cv2.imwrite(rgb_save_path, rgb_img)
                cv2.imwrite(thermal_save_path, thermal_img)

        plot_on_imgs(self.train_labels_dir,
                     self.train_rgb_dir_adjusted,
                     self.train_thermal_dir_adjusted,
                     train_rgb_annotated_path,
                     train_thermal_annotated_path)

        plot_on_imgs(self.val_labels_dir,
                     self.val_rgb_dir_adjusted,
                     self.val_thermal_dir_adjusted,
                     val_rgb_annotated_path,
                     val_thermal_annotated_path)

        print(" Done.")





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    args = parser.parse_args()

    flir = FLIR_ADAS(root_dir=args.root_dir)
    flir.update_file_names()
    flir.remove_wrong_sizes()
    flir.remove_unmatched_thermal_files()
    flir.register_images()
    flir.create_label_files()
    flir.remove_extra_label_fils()
    flir.plot_annotations()