# Entry point for the project
from .utils import load_config
from .data import Datagenerator
from .model import Leo

import argparse
import torch

parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
args = parser.parse_args()
dataset = args.dataset[0]


def train_model(config):
    metatrain_dataloader = Datagenerator(dataset, config, data_type="train")
    batch_data = metatrain_dataloader.get_batch_data()
    #model = Leo()
    

def predict_model(config):
    pass



def main():
    config = load_config()
    if config["train"]:
        train_model(config)
    else:
        predict_model(config)
    
if __name__ == "__main__":
    main()