import os
from PIL import Image
from collections import defaultdict
import shutil
import argparse
import matlab.engine

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
    def __init__(self, root_dir, rgb_dir, thermal_dir):
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

    def sample_and_move_files(self):
        '''
        TODO
            - FIX VARIABLE NAMES TO CLASS STRUCTURE
            - CODE SAMPLING METHOD
            - CODE MATCHING METHOD TO CHECK IF THERMAL FILE EXISTS FOR SAMPLED RGB FILE
            - ALTER COPY FUNCTION BELOW
        '''
        for key in self.img_map.keys():
            full_path = (os.path.join(self.rgb_dir, key))
            os.mkdir(full_path)
            for file in self.img_map[key]:
                file_only = os.path.basename(file)
                source_file_path = (os.path.join(self.rgb, file))
                dest_file_path = os.path.join(full_path, file_only)

                shutil.copyfile(source_file_path, dest_file_path)





parser = argparse.ArgumentParser()
parser.add_argument("img_dir")
args = parser.parse_args()

arrange_files_by_image_size(args.img_dir)




