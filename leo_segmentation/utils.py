# Contains helper functions
import json
import pickle

def load_config(config_path:str="data/config.json"):
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config

def load_data(config, dataset, data_type):
    dataset_config = config["datasets"]
    data_path = dataset_config[dataset][f"{data_type}_data_path"]
    data_dict = load_pickled_data(data_path)
    assert(list(data_dict.keys() == ['embeddings', 'filenames', 'masks']))
    assert(len(data_dict["embeddings"] == len(data_dict["filenames"]) == len(data_dict["masks"])))
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