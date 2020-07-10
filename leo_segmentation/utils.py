# Contains helper functions
import torch
import json
import pickle
import os
from easydict import EasyDict as edict

def load_config(config_path:str="data/config.json"):
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return edict(config)

def load_data(config:dict, dataset:str, data_type:str):
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
    root_data_path = config.datasets_path
    if data_type not in ["meta_train", "meta_val", "meta_test"]:
        raise ValueError("Make sure dataset files end with train, val or test")

    if dataset in config.datasets:
        data_path = os.path.join(root_data_path, dataset, f"{dataset}_{data_type}.pkl")
        data_dict = load_pickled_data(data_path)
    else:
        raise ValueError("Dataset does not exist")
    assert(list(data_dict.keys()) == ['embeddings', 'filenames', 'masks'])
    assert(len(data_dict["embeddings"]) == len(data_dict["filenames"]) == len(data_dict["masks"]))
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
        return pytensor.numpy()

def optimize_model(model, train_stats_store):
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    return epoch, loss, accuracy

def load_model(model_path):
   
    model = LEO()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    return model


def save_model(model, model_path):
    """
    Returns a trained model

    Parameters:
    model_path (str): path containing stored model

    Returns:
    model(pkl): trained model

    """
    
    torch.save(model.state_dict(), model_path) 

    print("This return a trained model")
    return
