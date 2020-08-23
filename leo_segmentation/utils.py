import torch
import torch.optim as optim
import os, pickle, json, random
import numpy as np
from easydict import EasyDict as edict

def load_config(config_path:str = "data/config.json"):
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
    ratio = config.data_params.meta_train_val_test_ratio
    if dataset in config.datasets:
        data_path = os.path.join(os.path.dirname(__file__), config.data_path, f"{dataset}", "meta_classes.pkl")
        if os.path.exists(data_path) and not generate_new:
            meta_classes_splits = load_pickled_data(data_path)
        else:
            classes = os.listdir(os.path.join(os.path.dirname(__file__), "data", f"{dataset}", "train", "images"))
            if shuffle_classes:
                random.shuffle(classes)
            meta_classes_splits = {"meta_train":classes[:ratio[0]],
                                   "meta_val":classes[ratio[0]:ratio[0] + ratio[1]],
                                   "meta_test":classes[ratio[0] + ratio[1]:ratio[0] + ratio[1] + ratio[2]]}
            assert (len(meta_classes_splits["meta_train"]) + \
                    len(meta_classes_splits["meta_val"]) + \
                    len(meta_classes_splits["meta_test"]))  == len(classes), \
                    "error exists in the ratios supplied"
            
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

def numpy_to_tensor(np_data):
    """Converts numpy array to pytorch tensor"""
    config = load_config()
    np_data = np_data.astype(config.dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
    return torch.from_numpy(np_data).to(device)
    
def tensor_to_numpy(pytensor):
    """Converts pytorch tensor to numpy"""
    if pytensor.is_cuda:
        return pytensor.cpu().detach().numpy()
    else:
        return pytensor.detach().numpy()

def check_experiment(config):
    """
    Checks if the experiment is new or not
    Creates a log file for a new experiment
    Args:
        config(dict)
    Returns:
        Bool
    """
    experiment = config.experiment
    model_root = os.path.join(config.data_path, "models")
    model_dir = os.path.join(model_root, "experiment_{}" \
                             .format(experiment.number))
    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)
    existing_models = os.listdir(model_root)
    existing_checkpoints = os.listdir(os.path.join(model_root, f"experiment_{experiment.number}"))

    if f"experiment_{experiment.number}" in existing_models and \
        f"checkpoint_{experiment.episode}.pth.tar" in existing_checkpoints:
            return True
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        msg = f"*********************Experiment {experiment.number}********************\n"
        msg += f"Description: {experiment.description}"
        log_filename = os.path.join(model_dir, "model_log.txt")
        log_data(msg, log_filename)
        return None

def prepare_inputs(data):
    """
    change the channel dimension for data
    Args:
        data (tensor): (num_examples_per_class, height, width, channels)
    Returns:
        data (tensor): (num_examples_per_class, channels, height, width)
    """
  
    if len(data.shape) == 4:
        data = data.permute((0, 3, 1, 2))
    return data


def get_named_dict(metadata, batch):
    """Returns a named dict"""
    tr_data, tr_data_masks, val_data, val_masks = metadata
    data_dict = { 'tr_data': prepare_inputs(tr_data[batch]),
                  'tr_data_masks': prepare_inputs(tr_data_masks[batch]) ,
                  'val_data':  prepare_inputs(val_data[batch]),
                  'val_data_masks': prepare_inputs(val_masks[batch])}
    return edict(data_dict)


def display_data_shape(metadata):
    """Displays data shape"""
    if type(metadata) == tuple:
        tr_data, tr_data_masks, val_data, val_masks = metadata
        print(f"num tasks: {len(tr_data)}")
    else:
        tr_data, tr_data_masks, val_data, val_masks = metadata.tr_data,\
            metadata.tr_data_masks, metadata.val_data, metadata.val_data_masks 
   
    print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}". \
            format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
    

def log_data(msg, log_filename):
    """Log data to a file"""
    if os.path.exists(log_filename):
        mode_ = "a"
    else:
        mode_ = "w"
    with open(log_filename, mode_) as f:
        f.write(msg)

def calc_iou_per_class(pred_x, targets):
    """Calculates iou"""
    iou_per_class = []
    for i in range(len(pred_x)):
        pred = np.argmax(pred_x[i].cpu().detach().numpy(), 0).astype(int)
        target = targets[i].cpu().detach().numpy().astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        iou_per_class.append(iou)
        mean_iou_per_class = np.mean(iou_per_class)
    return mean_iou_per_class

def one_hot_target(mask, channel_dim=1):
    mask_inv = (~mask.type(torch.bool)).type(torch.float32)
    channel_zero = torch.unsqueeze(mask_inv, channel_dim)
    channel_one = torch.unsqueeze(mask, channel_dim)
    return torch.cat((channel_zero, channel_one), axis=channel_dim)

def softmax(py_tensor, channel_dim=1):
    py_tensor = torch.exp(py_tensor)
    return  py_tensor/torch.unsqueeze(torch.sum(py_tensor, dim=channel_dim), channel_dim)

def sparse_crossentropy(target, pred,  channel_dim=1, eps=1e-10):
    pred += eps
    loss = torch.sum(-1 * target * torch.log(pred), dim=channel_dim)
    return torch.mean(loss)