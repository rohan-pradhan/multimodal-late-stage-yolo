import os
import cv2
import PIL
from collections import defaultdict
import shutil
import argparse

def arrange_files_by_image_size(img_dir):
    img_map = defaultdict(list)
    if os.path.isdir(img_dir):
        for file in os.listdir(img_dir):
            full_path = os.path.normpath(os.path.join(img_dir, file))
            image = PIL.Image.open(full_path)
            width, height = image.size
            key = string(width) + "_" + string(height)
            img_map[key].append(full_path)

        for key in img_map.keys():
            full_path = os.path.normpath(os.path.join(img_dir, key))
            os.mkdir(full_path)
            for file in img_map[key]:
                source_file_path = os.path.normpath(os.path.join(img_dir, file))
                dest_file_dir = os.path.normpath(os.path.join(img_dir, key))
                dest_file_path = os.path.normpaht(os.path.join(dest_file_dir, file))
                shutil.copyfile(source_file_path, dest_file_path)

parser = argparse.ArgumentParser()
parser.add_argument("img_dir")
args = parser.parse_args()
print (args.echo)

arrange_files_by_image_size(args.img_dir)




