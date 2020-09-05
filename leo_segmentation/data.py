#contains data preprocessing functions
from utils import numpy_to_tensor, meta_classes_selector
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
        batch_size = config.num_tasks[self._data_type]
        img_datasets = datasets.ImageFolder(root = os.path.join(dataset_root_path, "images"))
        mask_datasets = datasets.ImageFolder(root = os.path.join(dataset_root_path, "masks"))

        if batch_size > len(classes):
            raise ValueError("number of tasks must be less than the number of available classes")

        
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

            def data_path_assertions(data_path, img_or_mask):
                temp = data_path.split(os.sep)
                _img_or_mask, _selected_class = temp[-3], temp[-2]
                assert _img_or_mask == img_or_mask, "wrong data type (image or mask)"
                assert _selected_class == selected_class, "wrong class (selected class)"
            
            img_paths = [i[0] for i in img_datasets.imgs if selected_class in i[0]]
            random.shuffle(img_paths)
            
            n_val_per_class = config.n_val_per_class[self._data_type]
            n_val_per_class = len(img_paths) - n_train_per_class if n_val_per_class == "rest" else n_val_per_class

            img_paths = list(np.random.choice(img_paths, n_train_per_class + n_val_per_class, replace=False))
            mask_paths = [i.replace("images", "masks") for i in img_paths]
           
            data_path_assertions(img_paths[-1], "images")
            data_path_assertions(mask_paths[-1], "masks")

            #create a list in the case only one image path is created
            img_paths  = [img_paths] if type(img_paths) == str else img_paths
            mask_paths  = [mask_paths] if type(mask_paths) == str else mask_paths
              
            tr_img_paths.extend(img_paths[:n_train_per_class])
            tr_masks_paths.extend(mask_paths[:n_train_per_class])
            val_img_paths.extend(img_paths[n_train_per_class:])
            val_masks_paths.extend(mask_paths[n_train_per_class:])
            
            tr_imgs.append(np.array([self.transform_image(Image.open(i)) for i in tr_img_paths]))
            tr_masks.append(np.array([self.transform_mask(Image.open(i)) for i in tr_masks_paths]))
            val_imgs.append(np.array([self.transform_image(Image.open(i)) for i in val_img_paths]))
            val_masks.append(np.array([self.transform_mask(Image.open(i)) for i in val_masks_paths]))

        assert len(classes_selected) == len(set(classes_selected)), "classes are not unique"

        return numpy_to_tensor(np.squeeze(np.array(tr_imgs))), numpy_to_tensor(np.squeeze(np.array(tr_masks))),\
               numpy_to_tensor(np.squeeze(np.array(val_imgs))), numpy_to_tensor(np.squeeze(np.array(val_masks))),\
               classes_selected

    def get_batch_data(self):
        return self.__getitem__(0)
        
class TrainingStats():
    """Stores train statistics data"""
    def __init__(self, config):
        self._stats = []
        self.config = config
    
    def set_episode(self, episode):
        self.episode = episode

    def set_batch(self, batch):
        self.batch = batch

    def update_stats(self, **kwargs):
        
        self.mode = kwargs["mode"]
        self.kl_loss = kwargs["kl_loss"]
        self.total_val_loss = kwargs["total_val_loss"]
        self._stats.append({
            "mode": self.mode,
            "episode": self.episode,
            "kl_loss": self.kl_loss,
            "total_val_loss": self.total_val_loss
        })
        self.log_model_stats_to_file()

    def update_inner_loop_stats(self, **kwargs):
        pass

    def reset_stats(self):
        self._stats = []

    def get_stats(self):
        return pd.DataFrame(self._stats)

    def get_latest_stats(self):
        return self._stats[-1]

    def log_inner_loop_stats_to_file(self):
        pass
    
    def log_model_stats_to_file(self):
        model_root = os.path.join(self.config.data_path, "models")
        model_dir  = os.path.join(model_root, "experiment_{}"\
                    .format(self.config.experiment.number))

        with open(os.path.join(model_dir, "model_log.txt"), "a") as f:
            msg = f"\nmode:{self.mode}, episode:{self.episode:03d}, kl_loss:{self.kl_loss:2f}, " 
            msg += f"total_val_loss:{self.total_val_loss:2f}"
            f.write(msg)
   
    def get_stats(self):
        return pd.DataFrame(self._stats)

    def get_latest_stats(self):
        return self._stats[-1]

    def disp_stats(self):
        msg = f"\nmode:{self.mode}, episode:{self.episode:03d}, kl_loss:{self.kl_loss:2f}, " 
        msg += f"total_val_loss:{self.total_val_loss:2f}"
        print(msg)

