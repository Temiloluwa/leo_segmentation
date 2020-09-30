import os
import shutil

# initialize all classes as train classes
train_classes = ['aeroplane',
            'bicycle', 
            'bird', 
            'boat', 
            'bottle',
            'bus',
            'car', 
            'cat',
            'chair',
            'cow', 
            'diningtable', 
            'dog',
            'horse',
            'motorbike', 
            'person',
            'pottedplant', 
            'sheep', 
            'sofa', 
            'train', 
            'tvmonitor']

def read_file(file_path):
    with open(file_path, "r") as f:
        file_list = f.readlines()
    file_list = [l.strip("\n") for l in file_list]
    return file_list

def generate_data(test_fold=0):
    # select validation classes
    val_classes = train_classes[test_fold*4: test_fold*4+5]
    # remove validation classes from train classes
    for i in val_classes:
        train_classes.remove(i)
    train_list = read_file(os.path.join(voc_imgsets, "train.txt"))
    val_list = read_file(os.path.join(voc_imgsets, "val.txt"))

    def transfer_data(train_or_val, classes, train_or_val_list):
        for _class in classes:
            dest_img_dir = os.path.join(f"pascal_5i_fold_{test_fold}", f"{train_or_val}", "images", f"{_class}")
            dest_mask_dir = os.path.join(f"pascal_5i_fold_{test_fold}", f"{train_or_val}", "masks", f"{_class}")
            src_img_dir = os.path.join("pascal_segmentation", f"{_class}", f"{train_or_val}", "images")
            src_mask_dir = os.path.join("pascal_segmentation", f"{_class}", f"{train_or_val}", "masks")

            if not os.path.exists(dest_mask_dir):
                os.makedirs(dest_img_dir)
                os.makedirs(dest_mask_dir)
            data_root = os.path.join(src_img_dir)
            for fn in os.listdir(data_root):
                # copy images and masks are are only present the train or validation list
                if fn.split(".")[0] in train_or_val_list:
                    shutil.copy2(os.path.join(src_img_dir, fn), os.path.join(dest_img_dir, fn))
                    shutil.copy2(os.path.join(src_mask_dir, fn), os.path.join(dest_mask_dir, fn))
    
    transfer_data("train", train_classes, train_list)
    transfer_data("val", val_classes, val_list)

voc_imgsets = os.path.join("VOCdevkit", "VOC2012", "ImageSets", "Segmentation")

if __name__ == "__main__":
    generate_data(test_fold=3)