from data import Datagenerator, TrainingStats
from model import LEO, load_model, save_model
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from utils import load_config, check_experiment, get_named_dict
import numpy as np
import tensorflow as tf
import os
import argparse
import torch 
import torch.optim
import gc

#parser = argparse.ArgumentParser(description='Specify train or inference dataset')
#parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
#args = parser.parse_args()
#dataset = args.dataset

dataset = "pascal_voc_raw"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


def train_model(config):
    """Trains Model"""
    writer = SummaryWriter(os.path.join(config.data_path, "models", str(config.experiment.number)))
    """
    if check_experiment(config):
        leo, optimizer, stats = load_model(config)
        episodes_completed = stats["episode"]
        leo.eval()
        leo = leo.to(device)
        train_stats = TrainingStats(config)
        train_stats.set_episode(episodes_completed)
        train_stats.update_stats(**stats)
    else:
    """
    leo = LEO(config)
    train_stats = TrainingStats(config)
    tf.keras.backend.clear_session()
    episodes_completed = 0
    episodes = 100
    for episode in range(episodes_completed+1, episodes+1):
        train_stats.set_episode(episode)
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        metatrain_loss = leo.compute_loss(metadata, train_stats)
        print(f"Episode {episode}, Tr loss {metatrain_loss}")
        leo.evaluate(metadata)

        #if episode % config.checkpoint_interval == 0:
        #    save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))
            #writer.add_graph(leo, metadata[:-1])
            #writer.close()
          
        #leo.evaluate_val_imgs(metadata, class_in_metadata, train_stats, writer)
        #train_stats.disp_stats()

        #if episode == episodes:
        #    return leo, metadata, class_in_metadata
        #del metadata
        #gc.collect()
        #torch.cuda.ipc_collect()
        #torch.cuda.empty_cache()
        
        #meta-val
        #dataloader = Datagenerator(config, dataset, data_type="meta_val", generate_new_metaclasses=False)
        #_, train_stats = compute_loss(leo, dataloader, train_stats, config, mode="meta_val")
        #train_stats.disp_stats()

        #dataloader = Datagenerator(config, dataset, data_type="meta_test", generate_new_metaclasses=False)
        #_, train_stats = compute_loss(leo, dataloader ,train_stats, config, mode="meta_test")
        #train_stats.disp_stats()
        

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