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
    num = 0
    source_data_list = os.listdir(source_path)
    for filename in source_data_list:
        if filename in images_or_mask_list:
            source = os.path.join(source_path, filename)
            destination = os.path.join(destination_path,filename)
            dest = shutil.copy(source, destination)
            num += 1
    return num
    
    
num_files_per_class = {}
num_transfered_per_class_img = {}
num_transfered_per_class_mask = {}
for i in range(20):
    select_class = i
    path_to_val_list = os.path.join(os.getcwd(),"class_names", f"{class_names[i]}_val.txt")
    path_to_train_list = os.path.join(os.getcwd(),"class_names", f"{class_names[i]}_train.txt")
    images_or_mask_list = []
    with open(path_to_val_list, "r") as f:
        images_list = f.readlines()
        for single_line in images_list:
            line_data=single_line.replace('\n',' ').split(' ')
            line_data.remove('')
            if line_data[1]=='1':
                images_or_mask_list.append(line_data[0])
                
    print("num images", len(images_or_mask_list))
    with open(path_to_train_list, "r") as f:
        images_list = f.readlines()
        for single_line in images_list:
            line_data=single_line.replace('\n',' ').split(' ')
            line_data.remove('')
            if line_data[1]=='1':
                images_or_mask_list.append(line_data[0])
    images_or_mask_list = [ i+".jpg" for i in images_or_mask_list]
    print("num images extended", len(images_or_mask_list))
    num_files_per_class[class_names[i]]= len(images_or_mask_list)

    working_fold = int(i//5)
    num_transfered_img = gen_img_or_mask("images", select_class, working_fold)
    num_transfered_per_class_img[class_names[i]] = num_transfered_img
    num_transfered_mask = gen_img_or_mask("masks", select_class, working_fold)
    num_transfered_per_class_mask[class_names[i]] = num_transfered_mask

print(f"num_transferred images: { num_transfered_per_class_img.items()}")
print(f"num_transferred masks: { num_transfered_per_class_mask.items()}")
print(f"Total images",sum([v for k,v in num_transfered_per_class_img.items()]))
print(f"Total masks",sum([v for k,v in num_transfered_per_class_mask.items()]))
