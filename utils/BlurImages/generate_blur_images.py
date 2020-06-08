from PIL import Image, ImageFilter
import os

input_dir = "D:/KAIST_Dataset_FOG/train/images/lwir"
output_dir = "D:/KAIST_Dataset_FOG/train/images/lwir_focus_high"


for file in os.listdir(input_dir):
    full_file = os.path.join(input_dir, file)
    img = Image.open(full_file)
    img = img.filter(ImageFilter.GaussianBlur(10))
    output_file = os.path.join(output_dir, file)
    img.save(output_file)