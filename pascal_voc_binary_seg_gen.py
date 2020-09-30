import os
import numpy as np
import operator
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import shutil
from PIL import Image

# color patterns for various masks
class_colors = {'aeroplane':[128,0,0],
                'bicycle':[0,128,0],
                'bird':[128,128,0],
                'boat':[0,0,128],
                'bottle':[128,0,128],
                'bus':[0,128,128],
                'car':[128,128,128],
                'cat':[64,0,0],
                'chair':[192,0,0],
                'cow':[64,128,0],
                'diningtable':[192,128,0],
                'dog':[64,0,128],
                'horse':[192,0,128],
                'motorbike':[64,128,128],
                'person':[192,128,128],
                'pottedplant':[0,64,0],
                'sheep':[128,64,0],
                'sofa':[0,192,0],
                'train':[128,192,0],
                'tvmonitor':[0,64,128]}

# read text file and return a list 
def read_file(file_path):
    with open(file_path, "r") as f:
        file_list = f.readlines()
    file_list = [l.strip("\n") for l in file_list]
    return file_list

def generate_masks(train_or_val):
    # list containing all train or validation images
    data_list = read_file(os.path.join(voc_imgsets, f"{train_or_val}.txt"))
    # for every mask in the list
    for fn in data_list:
        # read the mask data into a numpy array
        data = plt.imread(os.path.join(masks_root, f"{fn}.png"))
        # get the corresponding image
        img = os.path.join(imgs_root, f"{fn}.jpg")
        h,w,_ = data.shape
        # get the corresponding xml annotation file
        img_annotation = os.path.join(annotations_root, f"{fn}.xml")
        # parse the xml file
        tree = ET.parse(img_annotation)
        root = tree.getroot()
        img_classes = []
        # find all the semantic classes that can be found in the mask
        for obj in root.findall('object'):
            text_list = list(obj.itertext())
            img_classes.extend([i for i in text_list if i in class_colors.keys()])
        # img_class is a list containing all the semantic classes for the mask
        img_classes = list(set(img_classes))
        
        # for each class
        for _class in img_classes:
            # create a numpy array and fill it with zeros
            output_mask = np.zeros_like(data, dtype=np.uint8)
            # get the color that corresponds to that class
            _class_color = class_colors[_class]
            if not os.path.exists(os.path.join("pascal_segmentation", f"{_class}", f"{train_or_val}")):
                os.makedirs(os.path.join("pascal_segmentation", f"{_class}", f"{train_or_val}", "images"))
                os.makedirs(os.path.join("pascal_segmentation", f"{_class}", f"{train_or_val}", "masks"))
            
            for i in range(h):
                for j in range(w):
                    # check each pixel and convert it to a list
                    pixel_color = list((data[i,j]*255).astype(np.uint8))
                    # check pixel_color is equal to class_color
                    # make the pixel that corresponds to the class_color white
                    if operator.eq(pixel_color, _class_color):
                        output_mask[i,j] = [255, 255, 255]
            mask = Image.fromarray(output_mask)
            mask.save(os.path.join("pascal_segmentation", f"{_class}", f"{train_or_val}", "masks", f"{fn}.jpg"))
            shutil.copy2(img, os.path.join("pascal_segmentation", f"{_class}", f"{train_or_val}", "images"))
                     
os.makedirs("pascal_segmentation", exist_ok=True)
imgs_root = os.path.join("VOCdevkit", "VOC2012", "JPEGImages")
masks_root = os.path.join("VOCdevkit", "VOC2012", "SegmentationClass")
voc_imgsets = os.path.join("VOCdevkit", "VOC2012", "ImageSets", "Segmentation")
annotations_root = os.path.join("VOCdevkit", "VOC2012", "Annotations")

if __name__ == "__main__":
    generate_masks("train")
    generate_masks("val")