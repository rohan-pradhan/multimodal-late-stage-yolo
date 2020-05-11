import os
import shutil
from random import sample




class KAIST:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.annotation_dir = os.path.join(self.root_dir, "annotations")
        self.images_dir = os.path.join(self.root_dir, "images" )
        self.data_dir = os.path.join(self.root_dir, "data")

        os.mkdir(self.data_dir)

        self.train_out_dir = os.path.join(self.root_dir, "train")
        self.train_images_dir = os.path.join(self.train_out_dir, "images")
        self.train_images_lwir_dir = os.path.join(self.train_images_dir, "lwir")
        self.train_images_visible_dir = os.path.join(self.train_images_dir, "visible")
        self.train_labels_dir = os.path.join(self.train_out_dir, "labels")
        #
        os.mkdir(self.train_out_dir)
        os.mkdir(self.train_images_dir)
        os.mkdir(self.train_images_lwir_dir)
        os.mkdir(self.train_images_visible_dir)
        os.mkdir(self.train_labels_dir)

        self.val_out_dir = os.path.join(self.root_dir, "val")
        self.val_images_dir = os.path.join(self.val_out_dir, "images")
        self.val_images_lwir_dir = os.path.join(self.val_images_dir, "lwir")
        self.val_images_visible_dir = os.path.join(self.val_images_dir, "visible")
        self.val_labels_dir = os.path.join(self.val_out_dir, "labels")

        os.mkdir(self.val_out_dir)
        os.mkdir(self.val_images_dir)
        os.mkdir(self.val_images_lwir_dir)
        os.mkdir(self.val_images_visible_dir)
        os.mkdir(self.val_labels_dir)

        self.test_out_dir = os.path.join(self.root_dir, "test")
        self.test_images_dir = os.path.join(self.test_out_dir, "images")
        self.test_images_lwir_dir = os.path.join(self.test_images_dir, "lwir")
        self.test_images_visible_dir = os.path.join(self.test_images_dir, "visible")
        self.test_labels_dir = os.path.join(self.test_out_dir, "labels")
        #
        os.mkdir(self.test_out_dir)
        os.mkdir(self.test_images_dir)
        os.mkdir(self.test_images_lwir_dir)
        os.mkdir(self.test_images_visible_dir)
        os.mkdir(self.test_labels_dir)

        self.img_height = 512.0
        self.img_width = 640.0

        self.height_normalization = 1 / self.img_height
        self.width_normalization = 1 / self.img_width

        self.counter = 0



    def transform_single_label_yolo(self, label):
        print(label)
        if "person" in label[0]:
            obj_cls = 0
        elif "people" in label[0]:
            obj_cls = 1
        elif "cyclist" in label[0]:
            obj_cls = 2

        x_top_left = float(label[1])
        y_top_left = float(label[2])
        w_unnormalized = float(label[3])
        h_unnormalized = float(label[4])

        x = (x_top_left + w_unnormalized / 2.0) * self.width_normalization
        y = (y_top_left + h_unnormalized / 2.0) * self.height_normalization
        w = w_unnormalized * self.width_normalization
        h = h_unnormalized * self.height_normalization

        array_to_return = [str(obj_cls), str(x), str(y), str(w), str(h), "\n"]
        updated_label_str = " ".join(array_to_return)
        return (updated_label_str)


    def transform_label_file(self, label_file_path):
        updated_labels = []
        with open(label_file_path) as f:
            labels = f.readlines()
        labels = [x.strip() for x in labels]
        labels = labels[1:]
        print(label_file_path)
        for label in labels:
            label = label.split(" ")
            updated_labels.append(self.transform_single_label_yolo(label))

        return updated_labels

    def update_and_move_files(self, label_file_path, thermal_file_path, rgb_file_path, train_val_test):
        updated_labels = self.transform_label_file(label_file_path)
        updated_label_file_name = str(self.counter) + ".txt"

        updated_visible_img_file_name = str(self.counter) + ".jpg"
        updated_thermal_img_file_name = str(self.counter) + ".jpg"

        updated_label_file_path = None
        updated_visible_img_path = None
        updated_thermal_img_path = None



        if (train_val_test == 0): # train
            updated_label_file_path = os.path.join(self.train_labels_dir, updated_label_file_name)
            updated_thermal_img_path = os.path.join(self.train_images_lwir_dir, updated_thermal_img_file_name)
            updated_visible_img_path = os.path.join(self.train_images_visible_dir, updated_visible_img_file_name)

        elif (train_val_test == 1): #val
            updated_label_file_path = os.path.join(self.val_labels_dir, updated_label_file_name)
            updated_thermal_img_path = os.path.join(self.val_images_lwir_dir, updated_thermal_img_file_name)
            updated_visible_img_path = os.path.join(self.val_images_visible_dir, updated_visible_img_file_name)

        elif (train_val_test == 2): #test
            updated_label_file_path = os.path.join(self.test_labels_dir, updated_label_file_name)
            updated_thermal_img_path = os.path.join(self.test_images_lwir_dir, updated_thermal_img_file_name)
            updated_visible_img_path = os.path.join(self.test_images_visible_dir, updated_visible_img_file_name)


        print (updated_label_file_path)
        print (updated_thermal_img_path)
        print (updated_visible_img_path)

        shutil.move(thermal_file_path, updated_thermal_img_path)
        shutil.move(rgb_file_path, updated_visible_img_path)

        with open(updated_label_file_path, 'a+') as f:
            for label in updated_labels:
                f.write(label)
        self.counter += 1

    def transform_dir(self):
        annotations_file_mask = ["annotations"]
        train_file_mask = ["set00", "set01", "set02", "set03", "set04", "set05"]
        test_file_mask = ["set06", "set07", "set08", "set09", "set10", "set11"]

        for folder in train_file_mask:
            folder_path = os.path.join(self.root_dir, folder)  #root_dir/set00
            for sub_folder in os.listdir(folder_path):
                if sub_folder.startswith('.'):
                    continue
                sub_folder_path = os.path.join(folder_path, sub_folder) #root_dir/set00/v00
                thermal_folder_path = os.path.join(sub_folder_path, "lwir")
                for file in os.listdir(thermal_folder_path):
                    if file.startswith('.'):
                        continue
                    thermal_file_path = os.path.join(thermal_folder_path, file)
                    visible_file_path = thermal_file_path.replace("lwir", "visible")
                    annotation_file_name = file.replace("jpg", "txt")
                    annotation_file_path = os.path.join(self.root_dir, "annotations", folder, sub_folder, annotation_file_name)
                    self.update_and_move_files(annotation_file_path, thermal_file_path, visible_file_path, 0)
            shutil.rmtree(folder_path)



        for folder in test_file_mask:
            folder_path = os.path.join(self.root_dir, folder)  #root_dir/set00
            for sub_folder in os.listdir(folder_path):
                if sub_folder.startswith('.'):
                    continue
                sub_folder_path = os.path.join(folder_path, sub_folder) #root_dir/set00/v00
                thermal_folder_path = os.path.join(sub_folder_path, "lwir")
                for file in os.listdir(thermal_folder_path):
                    if file.startswith('.'):
                        continue
                    thermal_file_path = os.path.join(thermal_folder_path, file)
                    visible_file_path = thermal_file_path.replace("lwir", "visible")
                    annotation_file_name = file.replace("jpg", "txt")
                    annotation_file_path = os.path.join(self.root_dir, "annotations", folder, sub_folder, annotation_file_name)
                    self.update_and_move_files(annotation_file_path, thermal_file_path, visible_file_path, 2)
            shutil.rmtree(folder_path)


    def generate_val_set(self, percentage=0.08):
        list_of_train_labels = os.listdir(self.train_labels_dir)
        number_of_val_samples = int(len(list_of_train_labels)*percentage)
        list_of_val_labels = sample(list_of_train_labels, number_of_val_samples)

        for val_label in list_of_val_labels:
            train_label_path = os.path.join(self.train_labels_dir, val_label)
            img_file = val_label.replace("txt", "jpg")
            train_visible_path = os.path.join(self.train_images_visible_dir, img_file)
            train_thermal_path = os.path.join(self.train_images_lwir_dir, img_file)

            val_label_path = os.path.join(self.val_labels_dir, val_label)
            val_visible_path = os.path.join(self.val_images_visible_dir, img_file)
            val_thermal_path = os.path.join(self.val_images_lwir_dir, img_file)

            shutil.move(train_label_path, val_label_path)
            shutil.move(train_visible_path, val_visible_path)
            shutil.move(train_thermal_path, val_thermal_path)


    def generate_text_files(self):
        self.train_text_file_visible_path = os.path.join(self.root_dir, "train_visible.txt")
        self.train_text_file_thermal_path = os.path.join(self.root_dir, "train_thermal.txt")

        self.val_text_file_visible_path = os.path.join(self.root_dir, "val_visible.txt")
        self.val_text_file_thermal_path = os.path.join(self.root_dir, "val_thermal.txt")

        self.test_text_file_visible_path = os.path.join(self.root_dir, "test_visible.txt")
        self.test_text_file_thermal_path = os.path.join(self.root_dir, "test_thermal.txt")


        def generate_text_file(dir, output_file):
            with open(output_file, 'a+') as f:
                for img in os.listdir(dir):
                    img_path = os.path.join(dir, img)
                    print(img_path)
                    f.write(img_path)
                    f.write("\n")

        generate_text_file(self.train_images_lwir_dir, self.train_text_file_thermal_path)
        generate_text_file(self.train_images_visible_dir, self.train_text_file_visible_path)

        generate_text_file(self.val_images_lwir_dir, self.val_text_file_thermal_path)
        generate_text_file(self.val_images_visible_dir, self.val_text_file_visible_path)

        generate_text_file(self.test_images_lwir_dir, self.test_text_file_thermal_path)
        generate_text_file(self.test_images_visible_dir, self.test_text_file_visible_path)

    def create_names_file(self):
        self.names_file_path = os.path.join(self.data_dir, "kaist.names")
        with open(self.names_file_path, 'a+') as f:
            f.write("person\n")
            f.write("people\n")
            f.write("cyclist")
  '''
  TODO:
   - FIX DATA FILE CREATION. IT IS CURRENTLY PUTTING THE TRAIN / TEST DIR PATH. NEEDS TO GO TO TRAIN / TEST TEXT FILE.

  '''

    def create_data_files(self):
        self.visible_data_file_path = os.path.join(self.data_dir, "kaist_visible.data")
        self.thermal_data_file_path = os.path.join(self.data_dir, "kaist_thermal.data")

        def create_data_file(path, train_dir, test_dir):
            with open(path, 'a+') as f:
                f.write("classes=3\n")
                f.write("train=")
                f.write(train_dir)
                f.write("\n")
                f.write("valid=")
                f.write(test_dir)
                f.write("\n")
                f.write("names=")
                f.write(self.names_file_path)


        #Create data file for thermal
        create_data_file(self.thermal_data_file_path, self.train_images_lwir_dir, self.val_images_lwir_dir)

        #Create data file for visible
        create_data_file(self.visible_data_file_path, self.train_images_visible_dir, self.val_images_visible_dir)

    def run_util(self):
        self.transform_dir()
        self.generate_val_set()
        self.generate_text_files()
        self.create_names_file()
        self.create_data_files()








kaist = KAIST("D:/KAIST_Dataset")
kaist.run_util()




