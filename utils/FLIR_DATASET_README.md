# FLIR ADAS Dataset 
## Original File structure 
The files are split into three folders, train, val, and video. Each folder 
contains RGB and thermal files. There is also a JSON file for each split of the dataset that 
contains the annoations for the files.  

### File naming 
The images are indexed FLIR-(#image number).jpg for RGB files, and FLIR-(#image-number).jpeg for thermal files. There are thermal files that do **not** have a corresponding RGB image. 
The images are always indexed from 1 for the training set, and start at 8863 for the validation set. 

### Annotation Structure 
The annotations are seperated by image-id. The image ID represents what image the corresponding annotation matches to. The annotation iamge ids start at 0 rather 1 for training and 8863 for the validation set. 
There are some images with no annotations (i.e. there are no objects in that image).
In that case, the annotation file just skips that image-id. 

##FLIR ADAS Data Set Util 
The FLIR ADAS Data Set Util automatically configures the dataset to be compatible with this library. 
It invovles aligning the images, creating corresponding label files, removing unmatched files, and fixing the annotation file. 
The utility does the following steps listed below: 

### Updates file name 
The recognized file naming convention for this library is that files will e indexed by number -1. 
This allows for the file name to match the corresponding image-id in the annotations file. 

```angular2
FLIR-0001.jpg -> 0.jpg 
FLIR-0978.jpg -> 977.jpg 
FLIR-1049.jpeg -> 1048.jpeg 
FLIR-1049.jpeg -> 1048.jpeg 
```

### Remove files of wrong image sizes
The original FLIR ADAS dataset contains RGB images of different sizes and perspectives. The validation set and video set only contain images of 1800 by 1600 pixels. 
The training set contains images of 4 different image sizes, the majority of images are 1800 by 1600. The other sized RGB images are removed. 
Two directories are created for thermal images. One directory keeps all the thermal image files. The other directory removes the corresponding thermal images that were lost due to the removal of the off-sized RGB images.

### Remove un-matched thermal files 
Remove thermal files from the directory that is kept in sync with the RGB files to remove any thermal files that do not have a corresponding RGB file. 

### Run homography image alignment on RGB images
The remaining RGB files are registered to the thermal files using a pre-set 8 point homography. 

### Create label files
The annotation files are parsed and created into individual files in YOLO format. 
The annotation files are then converted into the correct format (top right to center coordinates). 
The annotation files labels are corrected (pedestrians, bikes, cars)

#TODO 
    - Add label file generation for different models (SSD, Faster R-CNN, etc).  
    - Automatically download the files from the dropbox 
