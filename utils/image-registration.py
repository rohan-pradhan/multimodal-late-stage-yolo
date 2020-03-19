import os
from PIL import Image
from collections import defaultdict
import shutil
import argparse
import glob
import random

class ImageRegistration:
    """
    A class used to organize file structure to faciliate easier multimodal image registration

    ...
    Attributes
    -----------
    rgb_dir : str
        a valid path to the rgb images to be organized by size
    thermal_dir : str
        a valid path to the thermal images to be organized by size
    outpuut_rgb_dir : str
        a generated path to an output directory where the sampled rgb images by size will be moved
    output_thermal_dir : str
        a generated path to an output directory where the sampled (matched to RGB) thermal images will be saved

    Methods
    --------
    arrange_files_by_imagesize()
        Creates a python dictionary list of all the RGB images and sorts them by size.
        Each key to the dictionary represents a unique RGB image size from the provided input RGB directory.

    sample_and_move_files()
        Takes the image map dictionary and samples a maximum of 10 files from each image size.
        Checks if the corresponding thermal image file exists.
        Creates directory for each iamge size and moves the sampled rgb files into the respective directory.
        Also moves the respective thermal file for each sampled rgb file.

    """
    def __init__(self, rgb_dir, thermal_dir):
        if os.path.isdir(rgb_dir):
            self.rgb_dir = rgb_dir
        else:
            print("Enter valid RGB dir...")
        if os.path.isdir(thermal_dir):
            self.thermal_dir = thermal_dir
        else:
            print("Enter valid thermal dir...")

        self.output_rgb_dir = os.path.join(self.rgb_dir, "sorted_RGB")
        self.output_thermal_dir = os.path.join(self.thermal_dir, "sorted_Thermal")
        self.img_map = defaultdict(list)

    def arrange_files_by_image_size(self):
        for file in os.listdir(self.rgb_dir):
            full_path = (os.path.join(self.rgb_dir, file))
            if not os.path.isfile(full_path):
                continue
            image = Image.open(full_path)
            width, height = image.size
            key = str(width) + "_" + str(height)
            self.img_map[key].append(full_path)

    def sample_and_move_files(self, num_samples=10):

        for key in self.img_map.keys():
            # Make folders to move sampled iamges of each size for both thermal and rgb
            full_path_rgb = (os.path.join(self.rgb_dir, key))
            full_path_thermal = (os.path.join(self.thermal_dir, key))

            #Removes directories if they already exist
            if (os.path.isdir(full_path_rgb)):
                shutil.rmtree(full_path_rgb)
            if (os.path.isdir(full_path_thermal)):
                shutil.rmtree(full_path_thermal)
                
            os.mkdir(full_path_rgb)
            os.mkdir(full_path_thermal)

            #Sample files for image size

            def check_matching_thermal_file(rgb_file):
                base_file = os.path.basename(rgb_file)
                rgb_file_path = os.path.join(self.rgb_dir, base_file)
                thermal_file_path = os.path.join(self.thermal_dir, base_file)

                rgb_file_path = rgb_file_path.split(".")[0] + ".*"
                thermal_file_path = thermal_file_path.split(".")[0] + ".*"

                rgb_file_path = glob.glob(rgb_file_path)[0]
                thermal_file_path = glob.glob(thermal_file_path)[0]

                if rgb_file_path and thermal_file_path:
                    return (True, rgb_file_path, thermal_file_path)
                else:
                    return (False, rgb_file_path, thermal_file_path)

            sampled_file_list = []

            for x in range(0,num_samples):
                sample = random.choice(self.img_map[key])
                matching, rgb_path, thermal_path = check_matching_thermal_file(sample)
                while (not matching):
                    sample = random.choice(self.img_map[key])
                    matching, rgb_path, thermal_path = check_matching_thermal_file(sample)

                sampled_file_list.append((rgb_path, thermal_path))

            print (sampled_file_list)

            #Move sampled files to respective sorted folder
            for file in sampled_file_list:
                rgb_file = file[0]
                thermal_file = file[1]

                rgb_file_only = os.path.basename(rgb_file)
                thermal_file_only = os.path.basename(thermal_file)

                source_file_path_rgb= os.path.join(self.rgb_dir, rgb_file_only)
                source_file_path_thermal = os.path.join(self.thermal_dir, thermal_file_only)

                dest_file_path_rgb = os.path.join(full_path_rgb, rgb_file_only)
                dest_file_path_thermal = os.path.join(full_path_thermal, thermal_file_only)

                shutil.copyfile(source_file_path_rgb, dest_file_path_rgb)
                shutil.copyfile(source_file_path_thermal, dest_file_path_thermal)








parser = argparse.ArgumentParser()
parser.add_argument("--rgb_dir")
parser.add_argument("--thermal_dir")
args = parser.parse_args()

image_registration = ImageRegistration(rgb_dir=args.rgb_dir, thermal_dir=args.thermal_dir)
image_registration.arrange_files_by_image_size()
image_registration.sample_and_move_files()




