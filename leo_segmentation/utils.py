import torch
import torch.optim as optim
import json
import pickle
import os
from easydict import EasyDict as edict
from model import LEO


def load_config(config_path: str = "data/config.json"):
    """Loads config file"""
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return edict(config)


def load_data(config: dict, dataset: str, data_type: str):
    """
    Reads a pickle file containing few-shot dataset
    Args:
        config(dict) - config dictionary
        dataset(str) - name of dataset
        data_type(str) - train, val, or test

    Returns:
        data_dict(dict): contains the following keys and values
        - embeddings - stores image/image embeddings
        - filenames -  filenames of the format <classname>_<imagename>.jpg
        - masks - segmentation masks for image
    """
    root_data_path = config.data_path
    #if data_type not in ["meta_train", "meta_val", "meta_test"]:
    #    raise ValueError("Make sure dataset files end with train, val or test")

    if dataset in config.datasets:
        data_path = os.path.join(root_data_path, dataset, f"{dataset}_{data_type}.pkl")
        data_dict = load_pickled_data(data_path)
    else:
        raise ValueError("Dataset does not exist")
    assert (list(data_dict.keys()) == ['embeddings', 'filenames', 'masks'])
    assert (len(data_dict["embeddings"]) == len(data_dict["filenames"]) == len(data_dict["masks"]))
    return data_dict


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
    if f"experiment_{experiment.number}" in existing_models:
        return True
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "model_log.txt"), "w") as f:
            msg = f"*********************Experiment {experiment.number}********************\n"
            msg += f"Description: {experiment.description}"
            f.write(msg)
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

    with open(os.path.join(model_dir, "model_log.txt"), "a") as f:
        msg = f"\n*********** checkpoint {episode} was loaded **************"
        f.write(msg)

    leo = LEO(config)
    optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)
    leo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    int_ov_union = checkpoint['int_ov_union']
    stats = {
        "episode": episode,
        "loss": loss,
        "int_ov_union": int_ov_union
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
        'episode': stats.episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': stats.loss,
        'int_ov_union': stats.int_ov_union
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
                with open(os.path.join(model_dir, "model_log.txt"), "a") as f:
                    msg = f"\n*********** checkpoint {stats.episode} was deleted **************"
                    f.write(msg)
                break
            elif user_input in negative_options:
                raise ValueError("Supply the correct episode number to start experiment")
            else:
                trials += 1
                print("Wrong Value Supplied")
                print(f"You have {3 - trials} left")
                if trials == 3:
                    raise ValueError("Supply the correct answer to the question")
