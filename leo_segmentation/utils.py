import torch
import torch.optim as optim
import os, sys, pprint, pickle, json, random, yaml, torchvision, \
    logging, logging.config, tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
from io import StringIO

def load_config(config_path:str = "leo_segmentation/data/config.json"):
    """Loads config file"""
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return edict(config)

def meta_classes_selector(config, dataset, generate_new, shuffle_classes=False):
    """
    Returns a dictionary containing classes for meta_train, meta_val, and meta_test_splits
    e.g if total available classes are:["aeroplane", "dog", "cat", "sheep", "window"]
    ratio [3,2,1] returns: meta_train:["aeroplane", "dog"], meta_val:["cat", "sheep"], meta_test:["window"]
    Args:
        dataset(str) - name of dataset 
        ratio(list) - list containing number of classes to allot to each of meta_train,
                            meta_val, and meta_test. e.g [3,2,2]
        generate_new(bool) - generate new splits or load splits already saved as pickle file
        shuffle_classes(bool) - shuffle classes before splitting
    Returns:
        meta_classes_splits(dict): contains classes for meta_train, meta_val and meta_test
    """
    class_splits = config.data_params.meta_class_splits
    if dataset in config.datasets:
        data_path = os.path.join(os.path.dirname(__file__), config.data_path, f"{dataset}", "meta_classes.pkl")
        if os.path.exists(data_path) and not generate_new:
            meta_classes_splits = load_pickled_data(data_path)
        else:
            classes = os.listdir(os.path.join(os.path.dirname(__file__), "data", f"{dataset}", "images"))
            if shuffle_classes:
                random.shuffle(classes)
            meta_classes_splits = {"meta_train":classes[class_splits.meta_train[0]:class_splits.meta_train[1]],
                                   "meta_val":classes[class_splits.meta_val[0]:class_splits.meta_val[1]],
                                   "meta_test":classes[class_splits.meta_test[0]:class_splits.meta_test[1]]}

            total_count = class_splits.meta_train[1] - class_splits.meta_train[0] + \
                          class_splits.meta_val[1] - class_splits.meta_val[0] + \
                          class_splits.meta_test[1] - class_splits.meta_test[0] 
                          
            assert len(set(meta_classes_splits["meta_train"] + \
                        meta_classes_splits["meta_val"] + \
                        meta_classes_splits["meta_test"]))  == total_count, "error exists in the ratios supplied"
            
            if os.path.exists(data_path):
                os.remove(data_path)
                save_pickled_data(meta_classes_splits, data_path)
            else:
                save_pickled_data(meta_classes_splits, data_path)
    
    return edict(meta_classes_splits)

def save_npy(np_array, filename):
    """Saves a .npy file to disk"""
    filename = f"{filename}.npy" if len(os.path.splitext(filename)[-1]) == 0 else filename
    with open(filename, "wb") as f:
        return np.save(f, np_array)
    
def load_npy(filename):
    """Reads a npy file"""
    filename = f"{filename}.npy" if len(os.path.splitext(filename)[-1]) == 0 else filename
    with open(filename, "rb") as f:
        return np.load(f)

def save_pickled_data(data, data_path):
    """Saves a pickle file"""
    with open(data_path, "wb") as f:
        data = pickle.dump(data,f)
    return data

def load_pickled_data(data_path):
    """Reads a pickle file"""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

def load_yaml(data_path):
    """Reads a yaml file"""
    with open(data_path, 'r') as f:
        return  yaml.safe_load(f)

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
    msg = f"********************* Experiment {experiment.number} *********************\n"
    msg += f"Description: {experiment.description}\n"
    log_filename = os.path.join(model_dir, "train_log.txt")
    log_data(msg, log_filename, overwrite=True)
    log_filename = os.path.join(model_dir, "val_log.txt")
    msg = "********************* Val stats *********************\n"
    log_data(msg, log_filename, overwrite=True)
    return None

def loggers(config):
    """Returns train and validation loggers"""
    experiment = config.experiment
    model_root = os.path.join(os.path.dirname(__file__), config.data_path, "models")
    model_dir = os.path.join(model_root, "experiment_{}" \
                                .format(experiment.number))
    config_dict = load_yaml('leo_segmentation/data/logging.yaml')
    create_log(config)
    config_dict["handlers"]["trainStatsHandler"]["filename"] = os.path.join(model_dir, "train_log.txt")
    config_dict["handlers"]["valStatsHandler"]["filename"] = os.path.join(model_dir, "val_log.txt")
    logging.config.dictConfig(config_dict)
    train_logger = logging.getLogger("train")
    val_logger = logging.getLogger("val")
    return train_logger, val_logger

def check_experiment(config):
    """
    Checks if the experiment is new or not
    Creates a log file for a new experiment
    Args:
        config(dict)
    Returns:
        Bool
    """
    # implement logic to confirm if an experiment already exists
    return None

def get_named_dict(metadata, batch):
    """Returns a named dict"""
    tr_imgs, tr_masks, val_imgs, val_masks, _, _, _ = metadata
    data_dict = { 'tr_imgs':tr_imgs[batch],
                  'tr_masks':tr_masks[batch],
                  'val_imgs':val_imgs[batch],
                  'val_masks':val_masks[batch]}
    return edict(data_dict)


def display_data_shape(metadata):
    """Displays data shape"""
    if type(metadata) == tuple:
        tr_imgs, tr_masks, val_imgs, val_masks, _, _, _ = metadata
        print(f"num tasks: {len(tr_imgs)}")
        val_imgs_shape = f"{len(val_imgs)} list of paths" if type(val_imgs) == list else  val_imgs.shape
        val_masks_shape = f"{len(val_imgs)} list of paths"  if type(val_masks) == list else val_masks.shape
    
    print("tr_imgs shape: {},tr_masks shape: {}, val_imgs shape: {}, val_masks shape: {}". \
            format(tr_imgs.shape, tr_masks.shape, val_imgs_shape, val_masks_shape))
    

def log_data(msg, log_filename, overwrite=False):
    """Log data to a file"""
    mode_ = "w" if not os.path.exists(log_filename) or overwrite else "a"
    with open(log_filename, mode_) as f:
        f.write(msg)

def calc_iou_per_class(pred_x, targets):
    """Calculates iou"""
    iou_per_class = []
    for i in range(len(pred_x)):
        pred = np.argmax(pred_x[i].numpy(), -1).astype(int)
        target = targets[i].astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        iou_per_class.append(iou)
    mean_iou_per_class = np.mean(iou_per_class)
    return mean_iou_per_class

def plot_masks(mask_data, ground_truth=False):
    """
    plots masks for tensorboard make_grid
    Args:
        mask_data(torch.Tensor) - mask data
        ground_truth(bool) - True if mask is a groundtruth else it is a prediction
    """
    if ground_truth:
        plt.imshow(np.mean(mask_data.numpy(), 0)/2 + 0.5, cmap="gray")
    else:
        plt.imshow(np.mean(mask_data.numpy())/2 + 0.5, cmap="gray")

def print_to_string_io(variable_to_print, pretty_print=True):
    """ Prints value to string_io and returns value"""
    previous_stdout = sys.stdout
    sys.stdout = string_buffer = StringIO()
    pp = pprint.PrettyPrinter(indent=0)
    if pretty_print:
        pp.pprint(variable_to_print)
    else:
        print(variable_to_print)
    sys.stdout = previous_stdout
    string_value = string_buffer.getvalue()
    return string_value

if os.getcwd().split(os.sep)[-1] != "leo_segmentation":
    raise ValueError("Ensure your working directory is leo_segmentation")
config = load_config()
model_root = os.path.join(os.getcwd(), "leo_segmentation", config.data_path, "models")
model_dir = os.path.join(model_root, "experiment_{}".format(config.experiment.number))
train_logger, val_logger = loggers(config)