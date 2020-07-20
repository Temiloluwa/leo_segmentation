import os
import shutil

class_names = ['aeroplane',
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

def gen_img_or_mask(image_or_mask, select_class, working_fold):
    print(f"Working on {image_or_mask} for class {class_names[select_class]}")
    destination_path = os.path.join(os.getcwd(), f"data/grouped_by_classes/{class_names[select_class]}/{image_or_mask}")
    os.makedirs(destination_path, exist_ok=True)
    source_path = os.path.join(os.getcwd(), "data", "original_pascalvoc5i" ,"pascal-5", f"{working_fold}", f"{image_or_mask}", "classa") 

    source_data_list = os.listdir(source_path)
    for filename in source_data_list:
        if filename in images_or_mask_list:
            source = os.path.join(source_path, filename)
            destination = os.path.join(destination_path,filename)
            dest = shutil.copy(source, destination)
    

for i in range(20):
    select_class = i
    path_to_img_list = os.path.join(os.getcwd(),"class_names", f"class{select_class+1}.txt")
    with open(path_to_img_list, "r") as f:
        images_list = f.readlines()

    images_or_mask_list = [i.rstrip("\n")+".jpg" for i in images_list]
    working_fold = int(i//5)
    gen_img_or_mask("images", select_class, working_fold)
    gen_img_or_mask("masks", select_class, working_fold)
