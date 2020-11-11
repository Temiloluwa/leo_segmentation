import os
import numpy as np
from torchvision import transforms
from PIL import Image
from easydict import EasyDict as edict
from itertools import cycle, islice
from collections import defaultdict

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

CONFIG = {
    "classes": CLASSES,
    "batch_size": 32,
    "num_channels": 3,
    "img_height": 384,
    "img_width": 512,
    "n_query": 1,
    "n_support": 1,
    "train_batch_size": 15,
    "val_batch_size": 1000
}

config = edict(CONFIG)
img_transformer = transforms.Resize((config.img_height, config.img_width), interpolation=Image.BILINEAR)
mask_transformer = transforms.Resize((config.img_height, config.img_width), interpolation=Image.NEAREST)


def rotate(img):
    h, w = img.size
    if h > w:
        img = img.transpose(method=Image.ROTATE_270)
    return img


class Transform_image:
    """Performs data preprocessing steps on input images
    Args:
        img_width (int): Input image width
        img_height (int): Input image height
    """
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im):
        """Implements the data preprocessing
        Args:
            im (PIL.image): PIL image
        Returns:
            im (np.ndarray): numpy array containing image data
        """
        w, h = im.size
        if h > w:
            im = im.transpose(method=Image.ROTATE_270)
        
        im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im)/255.0

        return im


class Transform_mask:
    """Performs data preprocessing steps on input masks
    Args:
        img_width (int): Input image width
        img_height (int): Input image height
    """
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im):
        """Implements the data preprocessing
        Args:
            im (PIL.image): PIL image
        Returns:
            im (np.ndarray): numpy array containing image data
        """
        w, h = im.size

        if h > w:
            im = im.transpose(method=Image.ROTATE_270)

        # Nearest upsampling
        im = im.resize((self.img_width, self.img_height), resample=0)
        im = np.array(im)/255
        im = im.astype("uint8")
        return im


class DataGenerator:
    index = 0

    def __init__(self, fold, mode="train", shuffle=True):
        self.mode = mode
        self.val_classes = CLASSES[fold*5:(fold + 1)*5]
        self.train_classes = list(set(CLASSES) - set(self.val_classes))
        self.genenerate_class_paths(fold)
        self.shuffle = shuffle
        

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


class FewShotDataGenerator:
    """Sample task data Few shot data 
    """
    def __init__(self, fold, mode="train", config=config):
        self.config = config
        self.fold = fold
        self.mode = mode
        self.classes = config.classes
        self.classes_dict = {i:self.classes[i] for i in range(len(self.classes))}
        self.genenerate_class_paths(fold)
        self.transform_image = Transform_image(config.img_width, config.img_height)
        self.transform_support_mask = Transform_mask(config.img_width, config.img_height)
        self.transform_query_mask = Transform_mask(config.img_width, config.img_height)
    

    def genenerate_class_paths(self, fold):
        def read_file(filename):
            with open(filename, "r") as f:
                temp_list = f.readlines()
                temp_list = [i.strip("\n").split("__") for i in temp_list]
                temp_list = [(i[0], int(i[1])) for i in temp_list]
                return temp_list

        self.train_paths = []
        self.train_path_map = defaultdict(list)
        self.val_paths = []
        self.val_path_map = defaultdict(list)
        self.path_val_map = {}
        train_folds = [0, 1, 2, 3]
        train_folds.remove(fold)
        
        for fold_number in train_folds:
            textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'train', f'split{fold_number}_train.txt')
            train_files_list = read_file(textfile_path)
            for fname, class_idx in train_files_list:
                self.train_paths.append(fname)
                self.train_path_map[self.classes_dict[class_idx-1]].append(fname)

        textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'val', f'split{fold}_val.txt')
        val_files_list = read_file(textfile_path)
        for fname, class_idx in val_files_list:
            self.val_paths.append(fname)
            self.val_path_map[self.classes_dict[class_idx-1]].append(fname)
            self.path_val_map[fname] = self.classes_dict[class_idx-1] 
        
        self.train_classes = list(self.train_path_map.keys())
        self.val_classes = list(self.val_path_map.keys())

    def __len__(self):
        return 15 if self.mode == "train" else 1000

    def __getitem__(self, idx):
        mode = self.mode
        n_query = self.config.n_query
        n_support = self.config.n_support
        train_batch_size = self.config.train_batch_size
    
        if train_batch_size > len(self.train_classes):
            raise ValueError("number of tasks must be less than the number \
                             of available train classes")
    
        support_imgs = []
        query_imgs = []
        support_masks = []
        query_masks = []
        classes_selected = []
        support_fnames = []
        query_fnames = []

        if self.mode == "train":
            classes = self.train_classes.copy()
            for i in range(train_batch_size):
                selected_class = (np.random.choice(classes, 1, replace=False))[0]
                fname_list = self.train_path_map[selected_class]
                classes_selected.append(selected_class)
                classes.remove(selected_class)
                img_paths = list(np.random.choice(fname_list, n_query + n_support, replace=False))

                support_img_paths = img_paths[:n_support]
                support_mask_paths = img_paths[:n_support]
                query_img_paths = img_paths[n_support:]
                query_mask_paths = img_paths[n_support:]
                support_fns = img_paths[:n_support]
                query_fns = img_paths[n_support:]

                support_img_paths = [os.path.join(DATA_DIR, 'JPEGImages', f'{img}.jpg') for img in support_img_paths]
                support_mask_paths = [os.path.join(DATA_DIR, 'Binary_map_aug', mode,\
                            f'{self.classes.index(selected_class) + 1}', f'{mask}.png') for mask in support_mask_paths]
                query_img_paths = [os.path.join(DATA_DIR, 'JPEGImages', f'{img}.jpg') for img in query_img_paths]
                query_mask_paths = [os.path.join(DATA_DIR, 'Binary_map_aug', mode,\
                            f'{self.classes.index(selected_class) + 1}', f'{mask}.png') for mask in query_mask_paths]
                support_fns = [f'{selected_class}-{fn}' for fn in support_fns]
                query_fns = [f'{selected_class}-{fn}' for fn in query_fns]

                support_imgs.append(np.array([self.transform_image(Image.open(i)) for i in  support_img_paths]))
                query_imgs.append(np.array([self.transform_image(Image.open(i)) for i in  query_img_paths]))
                support_masks.append(np.array([self.transform_support_mask(Image.open(i)) for i in support_mask_paths]))
                query_masks.append(np.array([self.transform_query_mask(Image.open(i)) for i in query_mask_paths]))
                support_fnames.extend(support_fns)
                query_fnames.extend(query_fns)
        
            return np.array(support_imgs), np.array(query_imgs), np.array(support_masks), np.array(query_masks), support_fnames, query_fnames
        else:
            validation_paths = list(islice(cycle(self.val_paths), self.config.val_batch_size))
            for query_path in validation_paths:
                selected_class = self.path_val_map[query_path]
                fname_list = self.val_path_map[selected_class].copy()
                fname_list.remove(query_path)
                suport_paths = list(np.random.choice(fname_list, n_support, replace=False))
            
                support_img_paths = [os.path.join(DATA_DIR, 'JPEGImages', f'{img}.jpg') for img in suport_paths]
                support_mask_paths = [os.path.join(DATA_DIR, 'Binary_map_aug', mode,\
                            f'{self.classes.index(selected_class) + 1}', f'{mask}.png') for mask in suport_paths]
                query_img_paths = [os.path.join(DATA_DIR, 'JPEGImages', f'{query_path}.jpg')]
                query_mask_paths = [os.path.join(DATA_DIR, 'Binary_map_aug', mode,\
                            f'{self.classes.index(selected_class) + 1}', f'{query_path}.png')]
                support_fns = [f'{selected_class}-{fn}' for fn in suport_paths]
                query_fns = [f'{selected_class}-{query_path}']

                yield (np.array([self.transform_image(Image.open(i)) for i in  support_img_paths]),
                np.array([self.transform_image(Image.open(i)) for i in  query_img_paths]),
                np.array([self.transform_support_mask(Image.open(i)) for i in support_mask_paths]),
                np.array([self.transform_query_mask(Image.open(i)) for i in query_mask_paths]),
                support_fns,
                query_fns)
                
                
    def get_batch_data(self):
        return self.__getitem__(0)
