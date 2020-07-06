# Contains helper functions
import json
import pickle
import os

def load_config(config_path:str="data/config.json"):
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config

def load_data(config, dataset, data_type):
    root_data_path = config["datasets_path"]
    if dataset in config["datasets"]:
        data_path = os.path.join(root_data_path, dataset, f"{dataset}_{data_type}.pkl")
        data_dict = load_pickled_data(data_path)
    else:
        raise ValueError("Dataset does not exist")
    assert(list(data_dict.keys()) == ['embeddings', 'filenames', 'masks'])
    assert(len(data_dict["embeddings"]) == len(data_dict["filenames"]) == len(data_dict["masks"]))
    return data_dict

def load_pickled_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_model(model_path):
    """
    Returns a trained model

    Parameters:
    model_path (str): path containing stored model

    Returns:
    model(pkl): trained model

    """
    print("This return a trained model")
    return 