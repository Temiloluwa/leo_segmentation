import os
import collections
import pandas as pd
import numpy as np
import random
from .utils import meta_classes_selector, print_to_string_io, \
    train_logger, val_logger, load_config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "leo_segmentation", "data", "pascal_5i")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]
CLASSES_DICT = {i:CLASSES[i] for i in range(len(CLASSES))}
NUM_VALIDATION_EXAMPLES = 1000


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

class Datagenerator(Dataset):
    """Sample task data for Meta-train, Meta-val and Meta-train tasks

    Args:
        dataset (str): dataset name
        mode (str): Meta-train, Meta-val or Meta-test
    """
    def __init__(self, dataset, mode="meta_train"):
        self._dataset = dataset
        self.mode = mode
        self.class_img_mapping,\
        self.class_counts = class_to_img_mapping(fold)
        img_dims = config.data_params.img_dims
        self.transform_image = Transform_image(img_dims.width, img_dims.height)
        self.transform_mask = Transform_mask(img_dims.width, img_dims.height)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        _config = config.data_params
        dataset_root_path = os.path.join(os.path.dirname(__file__),
                                         config.data_path, self._dataset)
        classes = self.classes_dict[self.mode]
        num_classes = _config.num_classes
        n_train_per_class = _config.n_train_per_class[self.mode]
        n_val_per_class = _config.n_val_per_class[self.mode]
        batch_size = _config.num_tasks[self.mode]
        img_datasets = datasets.ImageFolder(root=os.path.join(
            dataset_root_path, "images"))

        if batch_size > len(classes):
            raise ValueError("number of tasks must be less than the number \
                             of available classes")

        def data_path_assertions(data_path, img_or_mask):
            """ Make assertions over selected paths"""
            temp = data_path.split(os.sep)
            _img_or_mask, _selected_class = temp[-3], temp[-2]
            assert _img_or_mask == img_or_mask,\
                    "wrong data type (image or mask)"
            # assert _selected_class == selected_class,\
            # "wrong class (selected class)"
        
        def data_path_assertions(data_path, img_or_mask):
            temp = data_path.split(os.sep)
            _img_or_mask, _selected_class = temp[-3], temp[-2]
            assert _img_or_mask == img_or_mask, "wrong data type (image or mask)"
            #assert _selected_class == selected_class, "wrong class (selected class)"

        tr_imgs = []
        tr_masks = []
        val_imgs = []
        val_masks = []
        classes_selected = []

        for i in range(batch_size):
            selected_class = (np.random.choice(classes, num_classes,
                              replace=False))[0]
            classes_selected.append(selected_class)
            classes.remove(selected_class)
            tr_img_paths = []
            tr_masks_paths = []
            val_img_paths = []
            val_masks_paths = []

            # Sample image paths belonging to classes
            img_paths = [i[0] for i in img_datasets.imgs
                         if selected_class in i[0]]
            random.shuffle(img_paths)
            if self.mode == "meta_train":
                img_paths = list(np.random.choice(img_paths,
                                 n_train_per_class + n_val_per_class,
                                 replace=False))
            for img_path in img_paths:
                data_path_assertions(img_path, "images")

            # Sample mask paths and convert them to the correct extensions
            mask_paths = [i.replace("images", "masks") for i in img_paths]
            mask_paths = [i.replace("jpg", "png") if not os.path.exists(i)
                          else i for i in mask_paths]
            # Create a list in the case only one image path is created
            img_paths = [img_paths] if type(img_paths) == str else img_paths
            mask_paths = [mask_paths] if type(mask_paths) == str else mask_paths
            
            # Divide sample paths to train and val splits
            tr_img_paths.extend(img_paths[:n_train_per_class])
            tr_masks_paths.extend(mask_paths[:n_train_per_class])
            val_img_paths.extend(img_paths[n_train_per_class:])
            val_masks_paths.extend(mask_paths[n_train_per_class:])
            
            # Store np.arrays for train and val images for all data types
            # Store only paths of val images for Meta-val and Meta-test
            tr_imgs.append(np.array([self.transform_image(Image.open(i))
                                     for i in tr_img_paths]))
            tr_masks.append(np.array([self.transform_mask(Image.open(i))
                                     for i in tr_masks_paths]))
            if self.mode in ["meta_val", "meta_test"]:
                val_imgs.append(val_img_paths)
                val_masks.append(val_masks_paths)
            else:
                val_imgs.append(np.array([self.transform_image(Image.open(i))
                                          for i in val_img_paths]))
                val_masks.append(np.array([self.transform_mask(Image.open(i))
                                           for i in val_masks_paths]))

        assert len(classes_selected) == len(set(classes_selected)),\
               "classes are not unique"
        total_tr_img_paths = tr_imgs + tr_masks
        total_vl_img_paths = val_imgs + val_masks
        if self.mode == "meta_train":
            tr_data, tr_data_masks, val_data, val_masks = np.array(tr_imgs),\
                                                        np.array(tr_masks),\
                                                        np.array(val_imgs),\
                                                        np.array(val_masks)
            return tr_data, tr_data_masks, val_data, val_masks,\
                classes_selected, total_tr_img_paths, total_vl_img_paths
        else:
            tr_data, tr_data_masks = np.array(tr_imgs), np.array(tr_masks)
            return tr_data, tr_data_masks, val_imgs, val_masks,\
                classes_selected, total_tr_img_paths, total_vl_img_paths

    def get_batch_data(self):
        return self.__getitem__(0)


class TrainingStats:
    """ Stores train statistics data """
    def __init__(self):
        self._meta_train_stats = []
        self._meta_val_stats = []
        self._meta_test_stats = []
        self._meta_train_ious = []
        self._meta_val_ious = []
        self._meta_test_ious = []
        
    def set_episode(self, episode):
        self.episode = episode
    
    def set_mode(self, mode):
        self.mode = mode

    def set_mode(self, mode):
        self.mode = mode

    def set_batch(self, batch):
        self.batch = batch

    def update_stats(self, **kwargs):
        self.total_val_loss = kwargs["total_val_loss"]
        self.mean_iou_dict = kwargs["mean_iou_dict"] 
        self.mean_iou_dict["episode"] = self.episode

        _stats = {
            "mode": self.mode,
            "episode": self.episode,
            "total_val_loss": self.total_val_loss,
        }

        if self.mode == "meta_train":
            self._meta_train_stats.append(_stats)
            self._meta_train_ious.append(self.mean_iou_dict)
        elif self.mode == "meta_val":
            self._meta_val_stats.append(_stats)
            self._meta_val_ious.append(self.mean_iou_dict)
        else:
            self._meta_test_stats.append(_stats)
            self._meta_test_ious.append(self.mean_iou_dict)
        
        mean_iou_dict = self.mean_iou_dict.copy()
        mean_iou_dict.pop("episode")
        average_iou = np.mean([v for _, v in mean_iou_dict.items()])
        mean_iou_string = print_to_string_io(mean_iou_dict, True)
        msg = f"mode: {self.mode}, episode: {self.episode: 03d}, "\
            + f"total_val_loss: {self.total_val_loss:2f}, "\
            + f"\nval_mean_iou:{mean_iou_string} "\
            + f"Average of all ious:{average_iou}"
        self.stats_msg = msg
        if self.mode == "meta_train":
            train_logger.debug(self.stats_msg)
        else:
            val_logger.debug(self.stats_msg)

    def get_stats(self, mode):
        if mode == "meta_train":
            stats = self._meta_train_stats
        elif mode == "meta_val":
            stats = self._meta_val_stats
        else:
            stats = self._meta_test_stats
        return pd.DataFrame(stats)

    def get_ious(self, mode):
        if mode == "meta_train":
            ious = self._meta_train_ious
        elif mode == "meta_val":
            ious = self._meta_val_ious
        else:
            ious = self._meta_test_ious
        return pd.DataFrame(ious)

    def disp_stats(self):
        print(self.stats_msg)

config = load_config()
