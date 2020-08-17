import torch
import torch.optim as optim
import os, pickle, json, random
import numpy as np
from easydict import EasyDict as edict
from model import LEO


def load_config(config_path: str = "data/config.json"):
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
    # issue: config and dataset exchanged
    if config == "pascal_voc":
        temp1 = config
        temp2 = dataset
        config = temp2
        dataset = temp1
    ratio = config.data_params.meta_train_val_test_ratio
    if dataset in config.datasets:
        data_path = os.path.join(os.path.dirname(__file__), config.data_path, f"{dataset}", "meta_classes.pkl")
        if os.path.exists(data_path) and not generate_new:
            meta_classes_splits = load_pickled_data(data_path)
        else:
            classes = os.listdir(os.path.join(os.path.dirname(__file__), "data", f"{dataset}", "train", "images"))
            if shuffle_classes:
                random.shuffle(classes)
            meta_classes_splits = {"meta_train": classes[:ratio[0]],
                                   "meta_val": classes[ratio[0]:ratio[0] + ratio[1]],
                                   "meta_test": classes[ratio[0] + ratio[1]:ratio[0] + ratio[1] + ratio[2]]}
            assert (len(meta_classes_splits["meta_train"]) + \
                    len(meta_classes_splits["meta_val"]) + \
                    len(meta_classes_splits["meta_test"]) ==
                    len(classes), "error exists in the ratios supplied")

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

    if f"experiment_{experiment.number}" in existing_models and \
            len(os.listdir(os.path.join(model_root, f"experiment_{experiment.number}"))) > 1:
        return True
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        msg = f"*********************Experiment {experiment.number}********************\n"
        msg += f"Description: {experiment.description}"
        log_filename = os.path.join(model_dir, "model_log.txt")
        log_data(msg, log_filename)
        return None


def load_model(config):
    """
    Loads the model
    Args:
        config - global config
        **************************************************
        Note: The episode key in the experiment dict
        implies the checkpoint that should be loaded
        when the model resumes training. If episode is
        -1, then the latest model is loaded else it loads
        the checkpoint at the supplied episode
        *************************************************
    Returns:
        leo :loaded model that was saved
        optimizer: loaded weights of optimizer
        stats: stats for the last saved model
    """
    experiment = config.experiment
    model_dir = os.path.join(config.data_path, "models", "experiment_{}" \
                             .format(experiment.number))

    checkpoints = os.listdir(model_dir)
    checkpoints.pop()
    max_cp = max([int(cp[11]) for cp in checkpoints])
    # if experiment.episode == -1, load latest checkpoint
    episode = max_cp if experiment.episode == -1 else experiment.episode
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    log_filename = os.path.join(model_dir, "model_log.txt")
    msg = f"\n*********** checkpoint {episode} was loaded **************"
    log_data(msg, log_filename)

    leo = LEO(config)
    optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)
    leo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mode = checkpoint['mode']
    total_val_loss = checkpoint['total_val_loss']
    kl_loss = checkpoint['kl_loss']

    stats = {
        "mode": mode,
        "episode": episode,
        "kl_loss": kl_loss,
        "total_val_loss": total_val_loss
    }

    return leo, optimizer, stats


def save_model(model, optimizer, config, stats):
    """
    Save the model while training based on check point interval

    if episode number is not -1 then a prompt to delete checkpoints occur if
    checkpoints for that episode number exits.
    This only occurs if the prompt_deletion flag in the experiment dictionary
    is true else checkpoints that already exists are automatically deleted
    Args:
        model - trained model
        optimizer - optimized weights
        config - global config
        stats - dictionary containing stats for the current episode

    Returns:
    """
    data_to_save = {
        'mode': stats.mode,
        'episode': stats.episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'kl_loss': stats.kl_loss,
        'total_val_loss': stats.total_val_loss
    }

    experiment = config.experiment
    model_root = os.path.join(config.data_path, "models")
    model_dir = os.path.join(model_root, "experiment_{}" \
                             .format(experiment.number))

    checkpoint_path = os.path.join(model_dir, f"checkpoint_{stats.episode}.pth.tar")
    if not os.path.exists(checkpoint_path):
        torch.save(data_to_save, checkpoint_path)
    else:
        trials = 0
        while trials < 3:
            if experiment.prompt_deletion:
                print(f"Are you sure you want to delete checkpoint: {stats.episode}")
                print(f"Type Yes or y to confirm deletion else No or n")
                user_input = input()
            else:
                user_input = "Yes"
            positive_options = ["Yes", "y", "yes"]
            negative_options = ["No", "n", "no"]
            if user_input in positive_options:
                # delete checkpoint
                os.remove(checkpoint_path)
                torch.save(data_to_save, checkpoint_path)
                log_filename = os.path.join(model_dir, "model_log.txt")
                msg = msg = f"\n*********** checkpoint {stats.episode} was deleted **************"
                log_data(msg, log_filename)
                break

            elif user_input in negative_options:
                raise ValueError("Supply the correct episode number to start experiment")
            else:
                trials += 1
                print("Wrong Value Supplied")
                print(f"You have {3 - trials} left")
                if trials == 3:
                    raise ValueError("Supply the correct answer to the question")


def get_in_sequence(data):
    """
    converts the tensor data (num_class, num_eg_per_class, img_dim) to ( (num_class * num_eg_per_class), img_dim)
    Args:
        data (tensor): (num_class, num_eg_per_class, H, W) # currently channel is missing
    Returns:
        data (tensor): (total_num_eg, Channel, H, W)
    """
    dim_list = list(data.size())
    data = data.permute(1, 0, 4, 2, 3)
    data = data.contiguous().view(dim_list[4], dim_list[2], dim_list[3], -1)
    data = data.permute(3, 0, 1, 2)
    #data = data.unsqueeze(1)  # because in the sample_data num_channels is missing
    return data


def get_named_dict(metadata, batch):
    """Returns a named dict"""
    tr_data, tr_data_masks, val_data, val_masks = metadata
    data_dict = {'tr_data_orig': tr_data[batch],
                 'tr_data': get_in_sequence(tr_data[batch]),
                 'tr_data_masks': tr_data_masks[batch],
                 'val_data_orig': val_data[batch],
                 'val_data': get_in_sequence(val_data[batch]),
                 'val_data_masks': val_masks[batch]}
    return edict(data_dict)


def display_data_shape(metadata):
    """Displays data shape"""
    tr_data, tr_data_masks, val_data, val_masks = metadata
    print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}". \
          format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
    print(f"num tasks: {len(tr_data)}")


def log_data(msg, log_filename):
    """Log data to a file"""
    if os.path.exists(log_filename):
        mode_ = "a"
    else:
        mode_ = "w"
    with open(log_filename, mode_) as f:
        f.write(msg)
