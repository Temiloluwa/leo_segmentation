import os
import numpy as np
from torchvision import transforms
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "leo_segmentation", "data", "pascal_5i")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CLASSES = ["aeroplane", 
            "bicycle", 
            "bird", 
            "boat", 
            "bottle", 
            "bus", 
            "car",
            "cat", 
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse", 
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa", 
            "train",
            "tvmonitor"]
CLASSES_DICT = {i:CLASSES[i] for i in range(len(CLASSES))}
BATCH_SIZE = 32
NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH = 3, 384, 512
img_transformer = transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.BILINEAR)
mask_transformer = transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=Image.NEAREST)


def rotate(img):
    h, w = img.size
    if h > w:
        img = img.transpose(method=Image.ROTATE_270)
    return img


class DataGenerator:
    index = 0

    def __init__(self, fold, mode="train", shuffle=True):
        self.mode = mode
        self.val_classes = CLASSES[fold*5:(fold + 1)*5]
        self.train_classes = list(set(CLASSES) - set(self.val_classes))
        self.genenerate_class_paths(fold)
        self.shuffle = shuffle
        

    def genenerate_class_paths(self, fold):
        def read_file(filename):
            with open(filename, "r") as f:
                temp_list = f.readlines()
                temp_list = [i.strip("\n").split("__") for i in temp_list]
                temp_list = [(i[0], int(i[1])) for i in temp_list]
                return temp_list

        self.train_paths = []
        self.train_class_names = []
        self.val_paths = []
        self.val_class_names = []
        train_folds = [0, 1, 2, 3]
        train_folds.remove(fold)
        
        for fold_number in train_folds:
            textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'train', f'split{fold_number}_train.txt')
            train_files_list = read_file(textfile_path)
            for fname, class_idx in train_files_list:
                self.train_paths.append(fname)
                self.train_class_names.append(CLASSES_DICT[class_idx-1])

        textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'val', f'split{fold}_val.txt')
        val_files_list = read_file(textfile_path)
        for fname, class_idx in val_files_list:
            self.val_paths.append(fname)
            self.val_class_names.append(CLASSES_DICT[class_idx-1])
        
        self.train_indices = list(range(len(self.train_paths)))

        assert len(self.train_paths) == len(self.train_class_names)
        assert len(self.val_paths) == len(self.val_class_names)
        assert set(self.train_class_names) == set(self.train_classes)
        assert set(self.val_class_names) == set(self.val_classes)

        return self

    
    def __iter__(self):
        if self.mode == "train":
            if self.shuffle:
                np.random.shuffle(self.train_indices)
            self.paths = [self.train_paths[i] for i in self.train_indices]
            self.class_names = [self.train_class_names[i] for i in self.train_indices]
        else:
            self.paths = self.val_paths
            self.class_names = self.val_class_names

        self.length = len(self.paths)
        self.img_paths = [os.path.join(DATA_DIR, 'JPEGImages', f'{i}.jpg') for i in self.paths]
        self.mask_paths = [os.path.join(DATA_DIR, 'Binary_map_aug', self.mode,\
                     f'{CLASSES.index(self.class_names[i]) + 1}', f'{self.paths[i]}.png') for i in range(len(self.paths))]
        return self


    def __next__(self):
        if self.index >= self.length:
            raise StopIteration

        end_point = self.index+BATCH_SIZE if self.index + BATCH_SIZE <= self.length else self.length
        img_paths = self.img_paths[self.index: end_point]
        mask_paths = self.mask_paths[self.index: end_point]
        class_names = self.class_names[self.index: end_point]
        imgs = [np.array(img_transformer(rotate(Image.open(i)))) for i in img_paths]
        masks = [np.array(mask_transformer(rotate(Image.open(i)))) for i in mask_paths]

        imgs = (np.array(imgs)/255).astype("float32")
        masks = (np.array(masks)/255).astype("uint8")

        self.index += BATCH_SIZE
        return imgs, masks, class_names

    def setmode(self, mode):
        self.mode = mode
        self.index = 0

    def __len__(self):
        if self.length % BATCH_SIZE == 0:
            _len = self.length//BATCH_SIZE
        else:
            _len = self.length//BATCH_SIZE + 1
        return _len