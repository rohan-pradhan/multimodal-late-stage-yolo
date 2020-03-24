import scipy.io
import os
from PIL import Image
from collections import defaultdict



def arrange_files_by_image_size(rgb_dir):
    img_map = defaultdict(int)
    for file in os.listdir(rgb_dir):
        full_path = (os.path.join(rgb_dir, file))
        if not os.path.isfile(full_path):
            continue
        image = Image.open(full_path)
        width, height = image.size
        key = str(width) + "_" + str(height)
        img_map[key]+=1
    return img_map


print(arrange_files_by_image_size("D:/RGB"))