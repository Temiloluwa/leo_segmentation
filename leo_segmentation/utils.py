# Contains helper functions
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import LEO
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

def optimize_model(model,data,target,optimizer):
    
    """
    Optimizes the model
    Args:
        data - training data        
        target - target data 
    
    Returns:
        loss :train loss
        optimizer:weights after optimizing
    """
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    return loss,optimizer

def load_model(model,optimizer,model_path):

    """
    Loads the model
    Args:
        model - initialized from the LEO class        
        optimizer - SGD optimizer
        model_path:location where the model is saved
    
    Returns:
        model :loaded model that was saved
        optimizer:loaded weights of optimizer
        epoch:the last epoch where the training stopped
    """
   
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    
    return model,optimizer,epoch


def save_model(model,optimizer,model_path,epoch,loss):
    """
    Save the model while training
    Args:
        model - trained model       
        optimizer - optimized weights
        model_path-location where the model is saved
        epoch- last epoch it got trained
        loss- training loss
    
    Returns:
        loss :train loss
        optimizer:weights after optimizing
    """
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path)
    
