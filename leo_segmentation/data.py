from .utils import meta_classes_selector, print_to_string_io, \
    train_logger, val_logger
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image
import collections, random
import pandas as pd
import numpy as np
import os

class Transform_image(object):

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im):
        w, h = im.size
        if h > w:
            im = im.transpose(method=Image.ROTATE_270).resize((self.img_width, self.img_height))
        else:
            im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im - 127.5)/127.5
        return im
    
class Transform_mask(object):

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im):
        w, h = im.size
        if h > w:
            im = im.transpose(method=Image.ROTATE_270).resize((self.img_width, self.img_height))
        else:
            im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im - 127.5)/127.5
        im = np.round(rgb2gray((im) > 0).astype(np.float32))
        return im

#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class Datagenerator(Dataset):
    """Data generator for meta train, meta val and meta test"""
    def __init__(self, config, dataset, data_type, generate_new_metaclasses=False):
        self._config  = config
        self._dataset = dataset
        self._data_type = data_type
        self.classes_dict = meta_classes_selector(config, dataset, generate_new_metaclasses)
        img_dims = config.data_params.img_dims
        self.transform_image = Transform_image(img_dims.width, img_dims.height)
        self.transform_mask = Transform_mask(img_dims.width, img_dims.height)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        config = self._config.data_params
        dataset_root_path = os.path.join(os.path.dirname(__file__), self._config.data_path, self._dataset)
        classes = self.classes_dict[self._data_type]
        num_classes = config.num_classes
        n_train_per_class = config.n_train_per_class[self._data_type]
        n_val_per_class = config.n_val_per_class[self._data_type]
        batch_size = config.num_tasks[self._data_type]
        img_datasets = datasets.ImageFolder(root = os.path.join(dataset_root_path, "images"))
        
        if batch_size > len(classes):
            raise ValueError("number of tasks must be less than the number of available classes")
        
        def data_path_assertions(data_path, img_or_mask):
            temp = data_path.split(os.sep)
            _img_or_mask, _selected_class = temp[-3], temp[-2]
            assert _img_or_mask == img_or_mask, "wrong data type (image or mask)"
            assert _selected_class == selected_class, "wrong class (selected class)"

        tr_imgs = []
        tr_masks = []
        val_imgs = []
        val_masks = []
        classes_selected = []

        for i in range(batch_size): 
            selected_class = (np.random.choice(classes, num_classes, replace=False))[0]
            classes_selected.append(selected_class)
            classes = list(set(classes) - set([selected_class]))
            tr_img_paths = []
            tr_masks_paths = []
            val_img_paths = []
            val_masks_paths = []

            img_paths = [i[0] for i in img_datasets.imgs if selected_class in i[0]]
            random.shuffle(img_paths)
            if self._data_type == "meta_train":
                img_paths = list(np.random.choice(img_paths, n_train_per_class + n_val_per_class, replace=False))
                
            for img_path in img_paths:
                data_path_assertions(img_path, "images")

            mask_paths = [i.replace("images", "masks") for i in img_paths]
            #create a list in the case only one image path is created
            img_paths  = [img_paths] if type(img_paths) == str else img_paths
            mask_paths  = [mask_paths] if type(mask_paths) == str else mask_paths
              
            tr_img_paths.extend(img_paths[:n_train_per_class])
            tr_masks_paths.extend(mask_paths[:n_train_per_class])
            val_img_paths.extend(img_paths[n_train_per_class:])
            val_masks_paths.extend(mask_paths[n_train_per_class:])
            
            tr_imgs.append(np.array([self.transform_image(Image.open(i)) for i in tr_img_paths]))
            tr_masks.append(np.array([self.transform_mask(Image.open(i)) for i in tr_masks_paths]))
            if self._data_type in ["meta_val", "meta_test"]: 
                val_imgs.append(val_img_paths)
                val_masks.append(val_masks_paths)
            else:
                val_imgs.append(np.array([self.transform_image(Image.open(i)) for i in val_img_paths]))
                val_masks.append(np.array([self.transform_mask(Image.open(i)) for i in val_masks_paths]))

        assert len(classes_selected) == len(set(classes_selected)), "classes are not unique"
        total_tr_img_paths = tr_imgs + tr_masks
        total_vl_img_paths = val_imgs + val_masks
        if self._data_type == "meta_train": 
            tr_data, tr_data_masks, val_data, val_masks = np.array(tr_imgs), np.array(tr_masks),\
                                                        np.array(val_imgs), np.array(val_masks)
            return tr_data, tr_data_masks, val_data, val_masks, \
                    classes_selected, total_tr_img_paths, total_vl_img_paths
        else:
            tr_data, tr_data_masks = np.array(tr_imgs), np.array(tr_masks)
            return tr_data, tr_data_masks, val_imgs, val_masks, \
                    classes_selected, total_tr_img_paths, total_vl_img_paths

    def get_batch_data(self):
        return self.__getitem__(0)
        
class TrainingStats():
    """Stores train statistics data"""
    def __init__(self, config):
        self._stats = []
        self.config = config
    
    def set_episode(self, episode):
        self.episode = episode
    
    def set_mode(self, mode):
        self.mode = mode

    def set_batch(self, batch):
        self.batch = batch

    def update_stats(self, **kwargs):
        self.kl_loss = kwargs["kl_loss"]
        self.total_val_loss = kwargs["total_val_loss"]
        self.mean_iou_dict =  kwargs["mean_iou_dict"]
        self._stats.append({
            "mode": self.mode,
            "episode": self.episode,
            "kl_loss": self.kl_loss,
            "total_val_loss": self.total_val_loss,
            "mean_iou_dict":self.mean_iou_dict
        })
        mean_iou_string = print_to_string_io(self.mean_iou_dict, pretty_print=True)
        msg = f"mode:{self.mode}, episode:{self.episode:03d}, kl_loss:{self.kl_loss:2f}, " 
        msg += f"total_val_loss:{self.total_val_loss:2f} \nval_mean_iou:{mean_iou_string}"
        self.stats_msg = msg
        if self.mode == "meta_train":
            train_logger.debug(self.stats_msg)
        else:
            val_logger.debug(self.stats_msg)

    def reset_stats(self):
        self._stats = []

    def get_latest_stats(self):
        return self._stats[-1]

    def get_stats(self):
        return pd.DataFrame(self._stats)

    def disp_stats(self):
        print(self.stats_msg)