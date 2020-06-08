from PIL import Image, ImageFilter
import os
import numpy as np
import cv2

input_dir = "D:/KAIST_Dataset_FOG/test/images/visible"
output_dir = "D:/KAIST_Dataset_FOG/test/images/visible_dropout"


for file in os.listdir(input_dir):
    img = np.zeros((512, 640, 3))
    output_file = os.path.join(output_dir, file)
    cv2.imwrite(output_file, img)