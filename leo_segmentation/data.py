import os
import collections
import pandas as pd
import numpy as np
import random
from .utils import meta_classes_selector, print_to_string_io, \
    train_logger, val_logger, numpy_to_tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image


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
            im = im.transpose(method=Image.ROTATE_270).\
                             resize((self.img_width, self.img_height))
        else:
            im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im - 127.5)/127.5
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
            im = im.transpose(method=Image.ROTATE_270).\
                resize((self.img_width, self.img_height))
        else:
            im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im - 127.5)/127.5
        im = np.round(rgb2gray((im) > 0).astype(np.float32))
        return im


def rgb2gray(rgb):
    """ Convert a RGB Image to gray scale """
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class Datagenerator(Dataset):
    """Sample task data for Meta-train, Meta-val and Meta-train tasks

    Args:
        config (dict): config
        dataset (str): dataset name
        data_type (str): Meta-train, Meta-val or Meta-test
    """
    def __init__(self, config, dataset, data_type):
        self._config = config
        self._dataset = dataset
        self._data_type = data_type
        self.classes_dict = meta_classes_selector(config, dataset)
        img_dims = config.data_params.img_dims
        self.transform_image = Transform_image(img_dims.width, img_dims.height)
        self.transform_mask = Transform_mask(img_dims.width, img_dims.height)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        config = self._config.data_params
        dataset_root_path = os.path.join(os.path.dirname(__file__),
                                         self._config.data_path, self._dataset)
        classes = self.classes_dict[self._data_type]
        num_classes = config.num_classes
        n_train_per_class = config.n_train_per_class[self._data_type]
        n_val_per_class = config.n_val_per_class[self._data_type]
        batch_size = config.num_tasks[self._data_type]
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
            if self._data_type == "meta_train":
                img_paths = list(np.random.choice(img_paths,
                                 n_train_per_class + n_val_per_class,
                                 replace=False))
            for img_path in img_paths:
                data_path_assertions(img_path, "images")

            # Sample mask paths and convert them to the correct extensions
            mask_paths = [i.replace("images", "masks") for i in img_paths]
            mask_paths = [i.replace("jpg", "png") if not os.path.exists(i)
                          else i for i in mask_paths]
            mask_paths = [i.replace("png", "jpg") if not os.path.exists(i)
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
            if self._data_type in ["meta_val", "meta_test"]:
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
        if self._data_type == "meta_train":
            tr_data, tr_data_masks, val_data, val_masks = numpy_to_tensor(np.array(tr_imgs)),\
                                                        numpy_to_tensor(np.array(tr_masks)),\
                                                        numpy_to_tensor(np.array(val_imgs)),\
                                                        numpy_to_tensor(np.array(val_masks))
            return tr_data, tr_data_masks, val_data, val_masks,\
                classes_selected, total_tr_img_paths, total_vl_img_paths
        else:
            tr_data, tr_data_masks = numpy_to_tensor(np.array(tr_imgs)), numpy_to_tensor(np.array(tr_masks))
            return tr_data, tr_data_masks, val_imgs, val_masks,\
                classes_selected, total_tr_img_paths, total_vl_img_paths

    def get_batch_data(self):
        return self.__getitem__(0)


class TrainingStats:
    """ Stores train statistics data """
    def __init__(self, config):
        self._meta_train_stats = []
        self._meta_val_stats = []
        self._meta_test_stats = []
        self._meta_train_ious = []
        self._meta_val_ious = []
        self._meta_test_ious = []
        self.config = config

    def set_episode(self, episode):
        self.episode = episode

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

        mean_iou_string = print_to_string_io(self.mean_iou_dict, True)
        msg = f"mode: {self.mode}, episode: {self.episode: 03d},\
                total_val_loss: {self.total_val_loss:2f},\
                val_mean_iou:{mean_iou_string}"
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
