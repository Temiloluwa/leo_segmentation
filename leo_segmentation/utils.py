import os
import sys
import pprint
import pickle
import json
import random
import yaml
import torch
import torch.optim as optim
import torchvision
import logging
import logging.config
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from io import StringIO


def load_config(config_path: str = "config.json"):
    """Loads config file"""
    config_path = os.path.join(project_root, config_path)
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return edict(config)


def meta_classes_selector(config, dataset, shuffle_classes=False):
    """ Returns a dictionary containing classes for meta_train, meta_val,
        and meta_test_splits
        Args:
            config (dict) : config
            dataset (str) : name of dataset
            ratio (list) : list containing ratios alloted to each data type
        Returns:
            meta_classes_splits (dict): classes all data types
    """

    classes = os.listdir(os.path.join(os.path.dirname(__file__),
                                 "data", f"{dataset}", "images"))
    classes = sorted(classes)
                                 
    def extract_splits(classes, meta_split):
        """ Returns class splits for a meta train, val or test """
        class_split = []
        for i in range(len(meta_split)//2):
            class_split.extend(classes[meta_split[i*2]:meta_split[i*2+1]])
        return class_split

    
    if "pascal" in dataset:
        pascal_splits = {
            "pascal_5i_fold_0":{"meta_train":[5, 20], "meta_val":[0, 5], "meta_test":[0, 5]},
            "pascal_5i_fold_1":{"meta_train":[0, 5, 10, 20], "meta_val":[5, 10], "meta_test":[5, 10]},
            "pascal_5i_fold_2":{"meta_train":[0, 10, 15, 20], "meta_val":[10, 15], "meta_test":[10, 15]},
            "pascal_5i_fold_3":{"meta_train":[0, 15], "meta_val":[15, 20], "meta_test":[15, 20]},
        }
        splits = edict(pascal_splits[dataset])
    else:
        splits = config.data_params.meta_class_splits

    meta_classes_splits = {"meta_train": extract_splits(classes, splits.meta_train),
                            "meta_val": extract_splits(classes, splits.meta_val),
                            "meta_test": extract_splits(classes, splits.meta_test)}

    total_count = len(set(meta_classes_splits["meta_train"] +
                        meta_classes_splits["meta_val"] +
                        meta_classes_splits["meta_test"]))
    assert total_count == len(classes), "check ratios supplied"
    
    return edict(meta_classes_splits)


def save_npy(np_array, filename):
    """Saves a .npy file to disk"""
    filename = f"{filename}.npy" if len(os.path.splitext(filename)[-1]) == 0\
        else filename
    with open(filename, "wb") as f:
        return np.save(f, np_array)


def load_npy(filename):
    """Reads a npy file"""
    filename = f"{filename}.npy" if len(os.path.splitext(filename)[-1]) == 0\
        else filename
    with open(filename, "rb") as f:
        return np.load(f)


def save_pickled_data(data, data_path):
    """Saves a pickle file"""
    with open(data_path, "wb") as f:
        data = pickle.dump(data, f)
    return data


def load_pickled_data(data_path):
    """Reads a pickle file"""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data


def numpy_to_tensor(np_data):
    """Converts numpy array to pytorch tensor"""
    config = load_config()
    np_data = np_data.astype(config.dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and config.use_gpu else "cpu")
    return torch.from_numpy(np_data).to(device)


def tensor_to_numpy(pytensor):
    """Converts pytorch tensor to numpy"""
    if pytensor.is_cuda:
        return pytensor.cpu().detach().numpy()
    else:
        return pytensor.detach().numpy()


def load_yaml(data_path):
    """Reads a yaml file"""
    with open(data_path, 'r') as f:
        return yaml.safe_load(f)


def list_to_tensor(_list, image_transformer):
    """Converts list of paths to pytorch tensor"""
    if type(_list[0]) == list:
        return [image_transformer(Image.open(i)) for i in _list]
    else:
        return np.expand_dims(image_transformer(Image.open(_list)), 0)


def create_log(config):
    """ Create Log File """
    experiment = config.experiment
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    msg = f"********************* Experiment {experiment.number} "
    msg += "*********************\n"
    msg += f"Description: {experiment.description}\n"
    log_filename = os.path.join(model_dir, "train_log.txt")
    log_data(msg, log_filename, overwrite=True)
    log_filename = os.path.join(model_dir, "val_log.txt")
    msg = "********************* Val stats *********************\n"
    log_data(msg, log_filename, overwrite=True)
    return None


def loggers(config):
    """Returns train and validation loggers"""
    config_dict = load_yaml(os.path.join(project_root, "logging.yaml"))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        create_log(config)
    config_dict["handlers"]["trainStatsHandler"]["filename"] = \
        os.path.join(model_dir, "train_log.txt")
    config_dict["handlers"]["valStatsHandler"]["filename"] = \
        os.path.join(model_dir, "val_log.txt")
    logging.config.dictConfig(config_dict)
    train_logger = logging.getLogger("train")
    val_logger = logging.getLogger("val")
    return train_logger, val_logger


def check_experiment(config):
    """ Checks if the experiment is new or not and
        creates a log file for a new experiment
    Args:
        config (dict)
    Returns:
        (bool)
    """
    experiment = config.experiment
    existing_models = os.listdir(model_root)
    checkpoint_paths = os.path.join(model_root,
                                    f"experiment_{experiment.number}")
    existing_checkpoints = os.listdir(checkpoint_paths)
    if f"experiment_{experiment.number}" in existing_models and\
       f"checkpoint_{experiment.episode}.pth.tar" in existing_checkpoints:
        return True
    elif f"experiment_{experiment.number}" in existing_models and\
         experiment.episode == -1:
        return True
    else:
        create_log(config)
        return False


def prepare_inputs(data):
    """ change the channel dimension for data
        Args:
            data (tensor): (num_examples_per_class, height, width, channels)
        Returns:
            data (tensor): (num_examples_per_class, channels, height, width)
    """
    if type(data) == list:
        return data
    if len(data.shape) == 4:
        data = data.permute((0, 3, 1, 2))
    return data


def get_named_dict(metadata, batch):
    """Returns a named dict"""
    tr_imgs, tr_masks, val_imgs, val_masks, _, _, _ = metadata
    data_dict = {'tr_imgs': prepare_inputs(tr_imgs[batch]),
                 'tr_masks': prepare_inputs(tr_masks[batch]),
                 'val_imgs':  prepare_inputs(val_imgs[batch]),
                 'val_masks': prepare_inputs(val_masks[batch])}
    return edict(data_dict)


def log_data(msg, log_filename, overwrite=False):
    """Log data to a file"""
    mode_ = "w" if not os.path.exists(log_filename) or overwrite else "a"
    msg = msg if overwrite else "\n" + msg
    with open(log_filename, mode_) as f:
        f.write(msg)


def display_data_shape(metadata):
    """Displays data shape"""
    if type(metadata) == tuple:
        tr_imgs, tr_masks, val_imgs, val_masks, _, _, _ = metadata
        print(f"num tasks: {len(tr_imgs)}")
        val_imgs_shape = f"{len(val_imgs)} list of paths"\
            if type(val_imgs) == list else val_imgs.shape
        val_masks_shape = f"{len(val_imgs)} list of paths"\
            if type(val_masks) == list else val_masks.shape
    print(f"tr_imgs shape: {tr_imgs.shape}, tr_masks shape: {tr_masks.shape}",
          f"val_imgs shape: {val_imgs_shape}, val_masks shape: {val_masks_shape}")


def calc_iou_per_class(pred_x, targets):
    """Calculates iou"""
    iou_per_class = []
    for i in range(len(pred_x)):
        pred = np.argmax(tensor_to_numpy(pred_x[i]), 0).astype(int)
        target = tensor_to_numpy(targets[i]).astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        iou_per_class.append(iou)
    mean_iou_per_class = np.mean(iou_per_class)
    return mean_iou_per_class


def plot_masks(mask_data, ground_truth=False):
    """ plots masks for tensorboard make_grid
        Args:
            mask_data(torch.Tensor) - mask data
            ground_truth(bool) - True if mask is a groundtruth else
                it is a prediction
    """
    if ground_truth:
        plt.imshow(np.mean(mask_data.numpy(), 0)/2 + 0.5, cmap="gray")
    else:
        plt.imshow(np.mean(mask_data.numpy())/2 + 0.5, cmap="gray")


def print_to_string_io(variable_to_print, pretty_print=True, logger=None):
    """ Prints value to string_io and returns value"""
    previous_stdout = sys.stdout
    sys.stdout = string_buffer = StringIO()
    pp = pprint.PrettyPrinter(indent=0)
    if pretty_print:
        pp.pprint(variable_to_print)
    else:
        print(variable_to_print)
    if logger is not None:
        logger.debug(variable_to_print)
    sys.stdout = previous_stdout
    string_value = string_buffer.getvalue()
    return string_value


project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config = load_config()
model_root = os.path.join(project_root, "leo_segmentation",
                          config.data_path, "models")
model_dir = os.path.join(model_root, "experiment_{}".
                         format(config.experiment.number))
train_logger, val_logger = loggers(config)
