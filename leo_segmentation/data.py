import os
import pandas as pd
import numpy as np
import random
from leo_segmentation.utils import meta_classes_selector, print_to_string_io, \
    train_logger, val_logger, numpy_to_tensor, load_config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image
from collections import defaultdict


config = load_config()
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),\
    "leo_segmentation", "data", config.selected_data)
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

if config.selected_data == "pascal_5i":
    CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "diningtable", "dog", "horse", 
                "motorbike", "person", "pottedplant", "sheep", "sofa", 
                "train", "tvmonitor"]
    CLASSES_DICT = {i:CLASSES[i] for i in range(len(CLASSES))}


def class_to_img_mapping(fold, mode):
    def read_file(filename):
        with open(filename, "r") as f:
            temp_list = f.readlines()
            temp_list = [i.strip("\n").split("__") for i in temp_list]
            temp_list = [(i[0], int(i[1])) for i in temp_list]
            return temp_list

    class_img_mapping = defaultdict(list)
    class_counts = defaultdict(int)
    train_folds = [0, 1, 2, 3]
    train_folds.remove(fold)

    if mode == "meta_train":
        for fold_number in train_folds:
            textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'train', f'split{fold_number}_train.txt')
            train_files_list = read_file(textfile_path)
            for fname, class_idx in train_files_list:
                class_img_mapping[CLASSES_DICT[class_idx - 1]].append(fname)
                class_counts[CLASSES_DICT[class_idx - 1]] += 1
    else:
        textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'val', f'split{fold}_val.txt')
        val_files_list = read_file(textfile_path)
        for fname, class_idx in val_files_list:
            class_img_mapping[CLASSES_DICT[class_idx - 1]].append(fname)
            class_counts[CLASSES_DICT[class_idx - 1]] += 1

    return class_img_mapping, class_counts


class Transform_image:
    """Performs data preprocessing steps on input images
    Args:
        img_width (int): Input image width
        img_height (int): Input image height
    """

    def __init__(self, img_width, img_height, normalize=True):
        self.img_width = img_width
        self.img_height = img_height
        self.normalize = normalize

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
        im = (im) / 255.0
        im = np.transpose(im, (2, 0, 1))

        # normalize for all pytorch pretrained models
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            im = (im - mean) / std

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
        im = np.array(im) / 255
        if im.ndim > 2:
            im = im[..., :3]
            im = rgb2gray(im)
            im = (im > 0.5)
        im = im.astype("uint8")
        
        return im


class PascalDatagenerator(Dataset):
    """Sample task data for Meta-train, Meta-val and Meta-train tasks

    Args:
        dataset (str): dataset name
        mode (str): Meta-train, Meta-val or Meta-test
    """

    def __init__(self, dataset, data_type="meta_train"):
        fold = config.data_params.fold
        self._dataset = dataset
        self.mode = data_type
        self.class_img_mapping, \
        self.class_counts = class_to_img_mapping(fold, self.mode)
        val_classes = CLASSES[fold * 5:(fold + 1) * 5]
        self.classes = list(set(CLASSES) - set(val_classes)) \
            if self.mode == "meta_train" else val_classes
        img_dims = config.data_params.img_dims
        self.transform_image = Transform_image(img_dims.width, img_dims.height)
        self.transform_mask = Transform_mask(img_dims.width, img_dims.height)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        _config = config.data_params
        dataset_root_path = os.path.join(os.path.dirname(__file__),
                                         config.data_path, self._dataset)
        num_classes = _config.num_classes
        n_train_per_class = _config.n_train_per_class[self.mode]
        n_val_per_class = _config.n_val_per_class[self.mode]
        batch_size = _config.num_tasks[self.mode]

        if batch_size > len(self.classes):
            raise ValueError("number of tasks must be less than the number \
                             of available classes")

        tr_imgs = []
        tr_masks = []
        val_imgs = []
        val_masks = []
        classes_selected = []
        classes = self.classes.copy()
        total_tr_fnames = []
        total_vl_fnames = []

        for i in range(batch_size):
            selected_class = (np.random.choice(classes, num_classes,
                                               replace=False))[0]
            fname_list = self.class_img_mapping[selected_class]

            classes_selected.append(selected_class)
            classes.remove(selected_class)

            tr_img_paths = []
            tr_masks_paths = []
            val_img_paths = []
            val_masks_paths = []

            # Sample image paths belonging to classes
            random.shuffle(fname_list)
            if self.mode == "meta_train":
                img_paths = list(np.random.choice(fname_list,
                                                  n_train_per_class + n_val_per_class, replace=False))
                tr_img_paths.extend(img_paths[:n_train_per_class])
                tr_masks_paths.extend(img_paths[:n_train_per_class])
                val_img_paths.extend(img_paths[n_train_per_class:])
                val_masks_paths.extend(img_paths[n_train_per_class:])
                total_tr_fnames.extend(img_paths[:n_train_per_class])
                total_vl_fnames.extend(img_paths[n_train_per_class:])
            else:
                t_img_paths = list(np.random.choice(fname_list,
                                                    n_train_per_class, replace=False))
                v_img_paths = list(set(fname_list) - set(t_img_paths))
                tr_img_paths.extend(t_img_paths)
                tr_masks_paths.extend(t_img_paths)
                val_img_paths.extend(v_img_paths)
                val_masks_paths.extend(v_img_paths)
                total_tr_fnames.extend(t_img_paths)
                total_vl_fnames.extend(v_img_paths)

            mode = self.mode.replace("meta_", "")
            tr_img_paths = [os.path.join(DATA_DIR, 'JPEGImages', f'{img}.jpg') for img in tr_img_paths]
            tr_masks_paths = [os.path.join(DATA_DIR, 'Binary_map_aug', mode, \
                                           f'{CLASSES.index(selected_class) + 1}', f'{mask}.png') for mask in
                              tr_masks_paths]
            val_img_paths = [os.path.join(DATA_DIR, 'JPEGImages', f'{img}.jpg') for img in val_img_paths]
            val_masks_paths = [os.path.join(DATA_DIR, 'Binary_map_aug', mode, \
                                            f'{CLASSES.index(selected_class) + 1}', f'{mask}.png') for mask in
                               val_masks_paths]

            # Store np.arrays for train and val images for all data types
            # Store only paths of val images for Meta-val and Meta-test
            # print(tr_img_paths)
            tr_imgs.append(np.array([self.transform_image(Image.open(i))
                                     for i in tr_img_paths]))
            tr_masks.append(np.array([self.transform_mask(Image.open(i))
                                      for i in tr_masks_paths]))
            if self.mode == "meta_train":
                val_imgs.append(np.array([self.transform_image(Image.open(i))
                                          for i in val_img_paths]))
                val_masks.append(np.array([self.transform_mask(Image.open(i))
                                           for i in val_masks_paths]))
            else:
                val_imgs.append(val_img_paths)
                val_masks.append(val_masks_paths)

        assert len(classes_selected) == len(set(classes_selected)), \
            "classes are not unique"

        if self.mode == "meta_train":
            tr_imgs, tr_masks, val_imgs, val_masks = numpy_to_tensor(np.array(tr_imgs)), \
                                                     numpy_to_tensor(np.array(tr_masks)), \
                                                     numpy_to_tensor(np.array(val_imgs)), \
                                                     numpy_to_tensor(np.array(val_masks))
            return tr_imgs, tr_masks, val_imgs, val_masks, \
                   classes_selected, total_tr_fnames, total_vl_fnames
        else:
            # val_imgs and val_masks are lists
            tr_imgs, tr_masks = numpy_to_tensor(np.array(tr_imgs)), numpy_to_tensor(np.array(tr_masks))
            return tr_imgs, tr_masks, val_imgs, val_masks, classes_selected, \
                   total_tr_fnames, total_vl_fnames

    def get_batch_data(self):
        return self.__getitem__(0)


class GeneralDatagenerator(Dataset):
    """Sample task data for Meta-train, Meta-val and Meta-train tasks

    Args:
        dataset (str): dataset name
        data_type (str): Meta-train, Meta-val or Meta-test
    """
    def __init__(self, dataset, data_type):
        self._dataset = dataset
        self._data_type = data_type
        self.classes_dict = meta_classes_selector(config, dataset)
        img_dims = config.data_params.img_dims
        self.transform_image = Transform_image(img_dims.width, img_dims.height)
        self.transform_mask = Transform_mask(img_dims.width, img_dims.height)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        _config = config.data_params
        dataset_root_path = os.path.join(os.path.dirname(__file__),
                                         config.data_path, self._dataset)
        classes = self.classes_dict[self._data_type]
        num_classes = _config.num_classes
        n_train_per_class = _config.n_train_per_class[self._data_type]
        n_val_per_class = _config.n_val_per_class[self._data_type]
        batch_size = _config.num_tasks[self._data_type] if self._data_type != "meta_test" \
                        else len(classes)

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

    def __init__(self):
        self._meta_train_stats = []
        self._meta_val_stats = []
        self._meta_test_stats = []
        self._meta_train_ious = []
        self._meta_val_ious = []
        self._meta_test_ious = []
        self._stats = []
        self.config = config
        self._best_episode = 0
        self._best_iou = 0

    @property
    def best_episode(self):
        return self._best_episode

    def set_episode(self, episode):
        self.episode = episode

    def set_mode(self, mode):
        self.mode = mode

    def set_batch(self, batch):
        self.batch = batch

    def update_after_restart(self):
        msg = f"======= Restarted at Episode {self.episode} ====== \n"
        train_logger.debug(msg)
        val_logger.debug(msg)
        print(msg)


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
        self.update_best_iou(average_iou)
        mean_iou_string = print_to_string_io(mean_iou_dict, True)
        if self.mode == "meta_val":
            start_msg = "\t\t ======= Meta Val IOUs ====== \n"
        else:
            start_msg  = ""
        msg = f"Mode: {self.mode} | Episode: {self.episode: 03d} | " \
              + f"Total Val Loss: {self.total_val_loss:2f} " \
              + f"\n{mean_iou_string} " \
        
        if self.mode == "meta_val":
            end_msg =  f"====> Average of all IOUs: {average_iou}\n" \
              + f" ====> Best Episode: {self._best_episode}"\
              + f" | Best IOU: {self._best_iou}\n"
        else:
            end_msg = "\n"
        self.stats_msg = start_msg + msg + end_msg

        if self.mode == "meta_train":
            train_logger.debug(self.stats_msg)
        else:
            val_logger.debug(self.stats_msg)
        self._stats.append({
            "mode": self.mode,
            "episode": self.episode,
            "total_val_loss": self.total_val_loss,
            "mean_iou_dict": self.mean_iou_dict
        })
        #self.log_model_stats_to_file()

    def get_stats(self, mode):
        if mode == "meta_train":
            stats = self._meta_train_stats
        elif mode == "meta_val":
            stats = self._meta_val_stats
        else:
            stats = self._meta_test_stats
        return pd.DataFrame(stats)

    def log_model_stats_to_file(self):
        model_root = os.path.join(os.path.dirname(__file__), self.config.data_path, "models")
        model_dir = os.path.join(model_root, "experiment_{}" \
                                 .format(self.config.experiment.number))
        log_file = "train_log.txt" if self.mode == "meta_train" else "val_log.txt"

        with open(os.path.join(model_dir, log_file), "a") as f:
            mean_iou_string = print_to_string_io(self.mean_iou_dict, pretty_print=True)
            msg = f"\nmode:{self.mode}, episode:{self.episode:03d}, "
            msg += f"total_val_loss:{self.total_val_loss:2f} \nval_mean_iou:{mean_iou_string}"
            f.write(msg)

    def get_ious(self, mode):
        if mode == "meta_train":
            ious = self._meta_train_ious
        elif mode == "meta_val":
            ious = self._meta_val_ious
        else:
            ious = self._meta_test_ious
        return pd.DataFrame(ious)

    def get_latest_stats(self):
        return self._stats[-1]

    def disp_stats(self):
        print(self.stats_msg)

    def update_best_iou(self, iou):
        if self.mode == "meta_val": 
            if iou > self._best_iou:
                self._best_iou = iou
                self._best_episode = self.episode

def rgb2gray(rgb):
    """ Convert a RGB Image to gray scale """
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

