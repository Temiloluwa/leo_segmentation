from utils import numpy_to_tensor, meta_classes_selector, load_npy, print_to_string_io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import collections, random
import pandas as pd
import numpy as np
import os


class Datagenerator(Dataset):
    """Data generator for meta train, meta val and meta test"""

    def __init__(self, config, dataset, data_type, generate_new_metaclasses=False):
        self._config = config
        self._dataset = dataset
        self._data_type = data_type
        self.classes_dict = meta_classes_selector(config, dataset)
        #print(self._data_type)
        #print(self.classes_dict)

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
            img_datasets = datasets.DatasetFolder(root=img_paths, loader=loader(img_paths, selected_class),
                                                  extensions=".npy")

            img_paths = [i for i in img_datasets.loader if selected_class in i]
            random.shuffle(img_paths)

            if self._data_type == "meta_train":
                img_paths = list(np.random.choice(img_paths, n_train_per_class + n_val_per_class, replace=False))

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
            if self._data_type in ["meta_val", "meta_test"]:
                val_imgs.append(val_img_paths)
                val_masks.append(val_masks_paths)
            else:
                val_imgs.append(np.array([load_npy(i) for i in val_img_paths]))
                val_masks.append(np.array([load_npy(i) for i in val_masks_paths]))

        assert len(classes_selected) == len(set(classes_selected)), "classes are not unique"

        if self._data_type == "meta_train":
            tr_data, tr_data_masks, val_data, val_masks = numpy_to_tensor(np.array(tr_imgs)), \
                                                          numpy_to_tensor(np.array(tr_masks)), \
                                                          numpy_to_tensor(np.array(val_imgs)), \
                                                          numpy_to_tensor(np.array(val_masks))
            return tr_data, tr_data_masks, val_data, val_masks, classes_selected
        else:
            tr_data, tr_data_masks = numpy_to_tensor(np.array(tr_imgs)), numpy_to_tensor(np.array(tr_masks))
            return tr_data, tr_data_masks, val_imgs, val_masks, classes_selected

    def get_batch_data(self):
        return self.__getitem__(0)


class TrainingStats():
    """Stores train statistics data"""

    def __init__(self, config):
        self._meta_train_stats = []
        self._meta_val_stats = []
        self._meta_test_stats = []
        self._meta_train_ious = []
        self._meta_val_ious = []
        self._meta_test_ious = []
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
        self.mean_iou_dict = kwargs["mean_iou_dict"]
        self.mean_iou_dict["episode"] = self.episode
        self._stats.append({
            "mode": self.mode,
            "episode": self.episode,
            "kl_loss": self.kl_loss,
            "total_val_loss": self.total_val_loss,
            "mean_iou_dict": self.mean_iou_dict
        })
        _stats = {
            "mode": self.mode,
            "episode": self.episode,
            "kl_loss": self.kl_loss,
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
        msg = f"\nmode: {self.mode}, episode: {self.episode: 03d}, kl_loss:{self.kl_loss:2f}, "\
            + f"total_val_loss: {self.total_val_loss:2f}, "\
            + f"\nval_mean_iou:{mean_iou_string} "\
            + f"Average of all ious:{average_iou}"
        self.stats_msg = msg
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
        model_dir = os.path.join(model_root, "experiment_{}" \
                                 .format(self.config.experiment.number))
        log_file = "train_log.txt" if self.mode == "meta_train" else "val_log.txt"

        with open(os.path.join(model_dir, log_file), "a") as f:
            mean_iou_string = print_to_string_io(self.mean_iou_dict, pretty_print=True)
            msg = f"\nmode:{self.mode}, episode:{self.episode:03d}, kl_loss:{self.kl_loss:2f}, "
            msg += f"total_val_loss:{self.total_val_loss:2f} \nval_mean_iou:{mean_iou_string}"
            f.write(self.stats_msg)

    def get_stats(self):
        return pd.DataFrame(self._stats)

    def get_latest_stats(self):
        return self._stats[-1]

    def disp_stats(self):
        print(self.stats_msg)