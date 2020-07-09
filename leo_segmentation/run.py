# Entry point for the project
from utils import load_config, optimize_model, load_model
from data import Datagenerator, TrainingStats
from model import LEO

import argparse
import torch

parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
args = parser.parse_args()
dataset = args.dataset[0]

def train_model(config):
    metatrain_dataloader = Datagenerator(dataset, config, data_type="meta_train")
    episodes = config.hyperparameters.episodes
    train_stats_store = TrainingStats() # to store training statistics

    for i in range(episodes):
        tr_data, tr_data_masks, val_data, val_masks = metatrain_dataloader.get_batch_data()
        print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}".\
            format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
    
        #write an if statement to load model or create a new model
        if blabblasa:
            model = LEO()  if blabllalb else load_model(model_path)
        
        
        epoch, loss, accuracy = optimize_model(model, train_stats_store)
        train_stats_store.update_stats(epoch, loss, accuracy)
    

def predict_model(config):
    pass



def main():
    config = load_config()
    if config.train:
        train_model(config)
    else:
        predict_model(config)
    
if __name__ == "__main__":
    main()