import torch
import torch.optim as optim
import os, pickle, json, random
import numpy as np
import torchvision
from matplotlib import pyplot as plt
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
    def create_log():
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        msg = f"*********************Experiment {experiment.number}********************\n"
        msg += f"Description: {experiment.description}"
        log_filename = os.path.join(model_dir, "model_log.txt")
        log_data(msg, log_filename)
        log_filename = os.path.join(model_dir, "val_stats_log.txt")
        msg = "*******************Val stats *************"
        log_data(msg, log_filename)
        return 

    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)
    existing_models = os.listdir(model_root)
    checkpoint_paths = os.path.join(model_root, f"experiment_{experiment.number}")
    if not os.path.exists(checkpoint_paths):
        create_log()
        return None
    existing_checkpoints = os.listdir(checkpoint_paths)

    if f"experiment_{experiment.number}" in existing_models and \
        f"checkpoint_{experiment.episode}.pth.tar" in existing_checkpoints:
            return True
    elif f"experiment_{experiment.number}" in existing_models and \
        experiment.episode == -1:
            return True
    else:
        create_log()
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
    tr_data, tr_data_masks, val_data, val_masks, _ = metadata
    data_dict = { 'tr_data': prepare_inputs(tr_data[batch]),
                  'tr_data_masks': prepare_inputs(tr_data_masks[batch]) ,
                  'val_data':  prepare_inputs(val_data[batch]),
                  'val_data_masks': prepare_inputs(val_masks[batch])}
    return edict(data_dict)


def display_data_shape(metadata):
    """Displays data shape"""
    if type(metadata) == tuple:
        tr_data, tr_data_masks, val_data, val_masks, _ = metadata
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

def plot_masks(mask_data, ground_truth=False):
    """
    plots masks for tensorboard make_grid
    Args:
        mask_data(torch.Tensor) - mask data
        ground_truth(bool) - True if mask is a groundtruth else it is a prediction
    """
    if ground_truth:
        plt.imshow(np.mean(mask_data.cpu().detach().numpy(), 0)/2 + 0.5, cmap="gray")
    else:
        plt.imshow(np.mean(mask_data.cpu().detach().numpy())/2 + 0.5, cmap="gray")


def summary_write_masks(batch_data, writer, grid_title, ground_truth=False):
    """
    Summary writer creates image grid for tensorboard
    Args:
        batch_data(torch.Tensor) - mask data
        writer - Tensorboard summary writer
        grid_title(str) - title of mask grid
        ground_truth(bool) - True if mask is a groundtruth else it is a prediction

    """
    if ground_truth:
        batch_data = torch.unsqueeze(batch_data, 1)
        masks_grid = torchvision.utils.make_grid(batch_data)
        episode = int(grid_title.split("_")[2])
        #plot_masks(masks_grid, ground_truth=True)
    else:
        batch_data = torch.unsqueeze(torch.argmax(batch_data, dim=1), 1)
        masks_grid = torchvision.utils.make_grid(batch_data)
        episode = int(grid_title.split("_")[1])
        #plot_masks(masks_grid)

    writer.add_image(grid_title, masks_grid, episode)

