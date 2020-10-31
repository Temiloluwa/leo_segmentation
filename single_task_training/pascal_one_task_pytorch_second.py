import numpy as np
import torch
import torch.optim as optim
import torchvision
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import pprint
from pathlib import Path
from itertools import cycle, islice
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image
import time
from one_shot_network import Res_Deeplab

pp = pprint.PrettyPrinter(width=41, compact=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "leo_segmentation", "data", "pascal_5i")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]
CLASSES_DICT = {i:CLASSES[i] for i in range(len(CLASSES))}
INPUT_SIZE = (384, 512)
BATCH_SIZE = 32
CHOSEN_CLASSES_TRAIN = ["bicycle"]
CHOSEN_CLASSES_VAL = ["car"]
NUM_VALIDATION_EXAMPLES = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
PSUEDO CODE FOR CNET DATALOADER
1. Creates a longlist of all training data instances [img name, class index]
2. Train images are all images except the fold to be used or validataion
2. Groups all training data instances by class
3. for each batch based on batch size:
    releases a pair of query,support images
4. for validation select all images from the selected fold:
    - make 3 copies of the list
    - shuffle the concatenation of these three list
    - then select 1000 images
    - select support images for each of the 1000 images
"""
def rotate(img):
    w, h = img.size
    if h > w:
        img = img.transpose(method=Image.ROTATE_270)
    return img


def read_file(filename):
    with open(filename, "r") as f:
        temp_list = f.readlines()
        temp_list = [i.strip("\n").split("__") for i in temp_list]
        temp_list = [(i[0], int(i[1])) for i in temp_list]
        return temp_list


query_mask_transformer = transforms.Resize((49, 65), interpolation=Image.NEAREST)
support_mask_transformer = transforms.Resize((INPUT_SIZE[0]//8, INPUT_SIZE[1]//8), interpolation=Image.NEAREST)
img_transformer = transforms.Resize(INPUT_SIZE, interpolation=Image.BILINEAR)
to_array = lambda x: np.array(x)

class DataGenerator:
    index = 0
    max_batch_class_length = None
    shuffle = True
    def __init__(self, fold, chosen_classes=CHOSEN_CLASSES_TRAIN, mode="train"):
        self.chosen_classes = chosen_classes
        self.mode = mode
        self.class_to_img_mapping(fold)
        #self.val_classes = CLASSES[fold*5:(fold + 1)*5]
        #self.train_classes = list(set(CLASSES) - set(self.val_classes))

    def class_to_img_mapping(self, fold):
        self.class_img_mapping = defaultdict(list)
        self.class_counts = defaultdict(int)
        train_folds = [0, 1, 2, 3]
        train_folds.remove(fold)
        
        if self.mode == "train":
            for fold_number in train_folds:
                textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'train', f'split{fold_number}_train.txt')
                train_files_list = read_file(textfile_path)
                for fname, class_idx in train_files_list:
                    self.class_img_mapping[CLASSES_DICT[class_idx-1]].append(fname)
                    self.class_counts[CLASSES_DICT[class_idx-1]] += 1

        if self.mode == "val" or self.mode != "train":
            textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'val', f'split{fold}_val.txt')
            val_files_list = read_file(textfile_path)
            for fname, class_idx in val_files_list:
                self.class_img_mapping[CLASSES_DICT[class_idx-1]].append(fname)
                self.class_counts[CLASSES_DICT[class_idx-1]] += 1
        return self

    
    def __iter__(self):
        if not self.shuffle:
            return self
            
        self.idx_chosen_classes = {}
        for _class in self.chosen_classes:
            _class_length = self.class_counts[_class]
            temp = list(range(_class_length))
            np.random.shuffle(temp)
            self.idx_chosen_classes[_class] = temp
            
        for _class in self.chosen_classes:
            _class_length = self.class_counts[_class]
            if self.max_batch_class_length is None:
                self.__len__()

            if _class_length != self.max_batch_class_length:
                _class_idx = self.idx_chosen_classes[_class]
                _class_idx = list(islice(cycle(_class_idx), self.max_batch_class_length))
                # cycle the indices until they have the same length has max batch length
                self.idx_chosen_classes[_class] = _class_idx

        for _class in self.chosen_classes:
            assert len(self.idx_chosen_classes[_class]) == self.max_batch_class_length, "all classes should be of the same length"
            assert len(set(self.idx_chosen_classes[_class])) == self.class_counts[_class], "error occured generating indices"
        
        if self.mode == "val" or self.mode != "train":
            assert len(self.idx_chosen_classes[_class]) == NUM_VALIDATION_EXAMPLES, f"num examples should be {NUM_VALIDATION_EXAMPLES}"

        return self

    def get_one_class_batch(self, chosen_class):
        idx = self.index
        fname_list = self.class_img_mapping[chosen_class]
        indices = self.idx_chosen_classes[chosen_class]
        query_imgs = [] 
        support_imgs = []
        query_masks = []
        support_masks = []

        for i in range(BATCH_SIZE):
            query_fn = fname_list[indices[idx]]
            end_loop = False 
            while not end_loop:
                support_fn = np.random.choice(fname_list, size=1)[0]
                if support_fn != query_fn:
                    end_loop = True
            idx += 1

            query_img_path = os.path.join(DATA_DIR, 'JPEGImages', f'{query_fn}.jpg')
            query_imgs.append(to_array(img_transformer(rotate(Image.open(query_img_path)))))
            
            support_img_path = os.path.join(DATA_DIR, 'JPEGImages', f'{support_fn}.jpg')
            support_imgs.append(to_array(img_transformer(rotate(Image.open(support_img_path)))))
            
            query_mask_path = os.path.join(DATA_DIR, 'Binary_map_aug', self.mode, f'{CLASSES.index(chosen_class) + 1}', f'{query_fn}.png')
            query_masks.append(to_array(query_mask_transformer(rotate(Image.open(query_mask_path)))))
            
            support_mask_path = os.path.join(DATA_DIR, 'Binary_map_aug', self.mode, f'{CLASSES.index(chosen_class) + 1}', f'{support_fn}.png')
            support_masks.append(to_array(support_mask_transformer(rotate(Image.open(support_mask_path)))))
        
        return query_imgs, support_imgs, query_masks, support_masks


    def __next__(self):
        query_imgs = [] 
        support_imgs = []
        query_masks = []
        support_masks = []
        for _class in self.chosen_classes:
            batch = self.get_one_class_batch(_class)
            query_imgs.extend(batch[0])
            support_imgs.extend(batch[1])
            query_masks.extend(batch[2])
            support_masks.extend(batch[3])
        
        query_imgs = np.array(query_imgs)
        query_imgs = (query_imgs/255).astype("float32")
        support_imgs = np.array(support_imgs)
        support_imgs = (support_imgs/255).astype("float32")
        query_masks = np.array(query_masks)/255
        support_masks = np.array(support_masks)/255
        if self.index + BATCH_SIZE >= self.max_batch_class_length:
            raise StopIteration

        self.index += BATCH_SIZE

        query_imgs = torch.from_numpy(query_imgs).permute((0, 3, 1, 2)).to(device)
        support_imgs = torch.from_numpy(support_imgs).permute((0, 3, 1, 2)).to(device)
        query_masks = torch.from_numpy(query_masks).to(device)
        support_masks = np.expand_dims(support_masks, axis=1).astype("float32")
        support_masks = torch.from_numpy(support_masks).to(device)

        return query_imgs, support_imgs, query_masks, support_masks

    def __len__(self):
        self.max_batch_class_length = np.max([self.class_counts[_class] for _class in self.chosen_classes])
        if self.mode == "val" or self.mode != "train":
            self.max_batch_class_length = NUM_VALIDATION_EXAMPLES
        return self.max_batch_class_length//BATCH_SIZE


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def load_resnet50_param(model, stop_layer='layer4'):
    resnet50 = torchvision.models.resnet50(pretrained=True)
    saved_state_dict = resnet50.state_dict()
    new_params = model.state_dict().copy()

    for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer
        i_parts = i.split('.')
        if not i_parts[0] == stop_layer:
            new_params['.'.join(i_parts)] = saved_state_dict[i]
        else:
            break
    model.load_state_dict(new_params)
    model.train()
    return model


def get_10x_lr_params(model):
    """
    get layers for optimization
    """

    b = []
    b.append(model.module.layer5.parameters())
    b.append(model.module.layer55.parameters())
    b.append(model.module.layer6_0.parameters())
    b.append(model.module.layer6_1.parameters())
    b.append(model.module.layer6_2.parameters())
    b.append(model.module.layer6_3.parameters())
    b.append(model.module.layer6_4.parameters())
    b.append(model.module.layer7.parameters())
    b.append(model.module.layer9.parameters())
    b.append(model.module.residule1.parameters())
    b.append(model.module.residule2.parameters())
    b.append(model.module.residule3.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def optim_or_not(model, yes):
    for param in model.parameters():
        if yes:
            param.requires_grad = True
        else:
            param.requires_grad = False

def turn_off(model):
    optim_or_not(model.module.conv1, False)
    optim_or_not(model.module.layer1, False)
    optim_or_not(model.module.layer2, False)
    optim_or_not(model.module.layer3, False)

def calc_iou_per_class(pred, y):
    pred = torch_to_numpy(pred)
    target = torch_to_numpy(y)
    ious = []
    for i in range(len(pred)):
        target = np.expand_dims(y[i], 0).astype(int)  
        pred = np.argmax(pred, axis=1).astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        ious.append(iou)
    return np.mean(ious)


model = Res_Deeplab(num_classes=2)
model = load_resnet50_param(model, stop_layer='layer4')
model = nn.DataParallel(model,[0])
turn_off(model)

chosen_classes = ["bicycle"]
train_data_generator = DataGenerator(fold=1, chosen_classes=chosen_classes)
#val_data_generator = DataGenerator(fold=1, chosen_classes=CHOSEN_CLASSES_VAL, mode="val")

num_batches = len(train_data_generator)
train_test_split = 0.7
train_split = int(0.7 * num_batches)
epochs = 60
freq = 2
lr = 1e-3
training_stats = []
epoch_times = []
optimizer = optim.Adam([{'params': get_10x_lr_params(model), 'lr': 10 *lr}], lr)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(1, epochs + 1):
    start_time = time.time()
    train_batch_loss = []
    val_batch_loss = []

    train_data_generator.index = 0
    for i, (query_imgs, support_imgs, query_masks, support_masks) in enumerate(tqdm(train_data_generator)):
        if i <= train_split:
            optimizer.zero_grad()
            pred = model(query_imgs, support_imgs, support_masks)
            loss = loss_fn(pred, query_masks.long())
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())
        else:
            pred = model(query_imgs, support_imgs, support_masks)
            loss = loss_fn(pred, query_masks.long())
            val_batch_loss.append(loss.item())

    train_loss = float(np.mean(train_batch_loss))
    val_loss = float(np.mean(val_batch_loss))

    train_data_generator.index = 0
    train_data_generator.shuffle = False
    for i, (query_imgs, support_imgs, query_masks, support_masks) in enumerate(tqdm(train_data_generator)):
        if i <= train_split:
            pred = model(query_imgs, support_imgs, support_masks)
            train_iou = calc_iou_per_class(pred, query_masks)
        else:
            pred = model(query_imgs, support_imgs, support_masks)
            val_iou = calc_iou_per_class(pred, query_masks)
    
    end_time = time.time()
    epoch_time = (end_time - start_time)/60
    
    if epoch % freq == 1:
        print(f"Epoch:{epoch:03d}- Epoch Time:{epoch_time:3f} train loss:{train_loss:3f}, train_iou:{train_iou:3f}, val loss:{val_loss:3f}, val_iou:{val_iou:3f}")
    epoch_times.append(epoch_time)
   
    training_stats.append({
      "epoch":epoch,
      "train loss":train_loss,
      "val loss":val_loss,
      "epoch time": epoch_time,
    })
  
