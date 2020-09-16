from .utils import numpy_to_tensor, meta_classes_selector, load_npy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import collections, random
import pandas as pd
import numpy as np
import os


class Datagenerator(Dataset):
    """Data generator for meta train, meta val and meta test"""
    def __init__(self, config, dataset, data_type, generate_new_metaclasses=False):
        self._config  = config
        self._dataset = dataset
        self._data_type = data_type
        self.classes_dict = meta_classes_selector(config, dataset, generate_new_metaclasses)

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
            
            def loader(data_path, selected_class):
                paths_ = []
                sub_fn_path = os.path.join(data_path, selected_class)
                for fn in os.listdir(sub_fn_path):
                    paths_.append(os.path.join(sub_fn_path, fn))
                return paths_

            def data_path_assertions(data_path, img_or_mask):
                temp = data_path.split(os.sep)
                _img_or_mask, _selected_class = temp[-3], temp[-2]
                assert _img_or_mask == img_or_mask, "wrong data type (image or mask)"
                assert _selected_class == selected_class, "wrong class (selected class)"
            
            img_paths = os.path.join(dataset_root_path, "images")
            img_datasets = datasets.DatasetFolder(root=img_paths, loader=loader(img_paths, selected_class), extensions=".npy")
            
            img_paths = [i for i in img_datasets.loader if selected_class in i]
            random.shuffle(img_paths)
            img_paths  = list(np.random.choice(img_paths , n_train_per_class + n_val_per_class, replace=False))
            
            for img_path in img_paths:
                data_path_assertions(img_path, "images")
            
            img_paths_train = img_paths[:n_train_per_class]
            img_paths_val = img_paths[n_train_per_class:]
            mask_paths_train = [i.replace("images", "masks") for i in img_paths_train]
            mask_paths_val = [i.replace("images", "masks") for i in img_paths_val]
            
            tr_img_paths.extend(img_paths_train)
            tr_masks_paths.extend(mask_paths_train)
            val_img_paths.extend(img_paths_val)
            val_masks_paths.extend(mask_paths_val)
            
            tr_imgs.append(np.array([load_npy(i) for i in tr_img_paths]))
            tr_masks.append(np.array([load_npy(i) for i in tr_masks_paths]))
            val_imgs.append(np.array([load_npy(i) for i in val_img_paths]))
            val_masks.append(np.array([load_npy(i) for i in val_masks_paths]))

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
        model_root = os.path.join(os.path.dirname(__file__), self.config.data_path, "models")
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

