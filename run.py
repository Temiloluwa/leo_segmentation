from leo_segmentation.data import Datagenerator, TrainingStats
from leo_segmentation.model import LEO, load_model, save_model
from leo_segmentation.utils import load_config, check_experiment, get_named_dict
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from IPython import get_ipython
import numpy as np
import tensorflow as tf
import os, argparse

try:
    shell = get_ipython().__class__.__name__
    dataset = "pascal_voc_raw"
except NameError:
    parser = argparse.ArgumentParser(description='Specify train or inference dataset')
    parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc_raw")
    args = parser.parse_args()
    dataset = args.dataset

def train_model(config):
    """Trains Model"""
    #writer = SummaryWriter(os.path.join(config.data_path, "models", str(config.experiment.number)))
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
    episodes = config.hyperparameters.episodes
    for episode in range(episodes_completed+1, episodes+1):
        train_stats.set_episode(episode)
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        metatrain_loss, train_stats = leo.compute_loss(metadata, train_stats)
        print(f"\nEpisode: {episode}, Tr loss: {metatrain_loss}")
        leo.evaluate(metadata)

        if episode % config.checkpoint_interval == 0:
            pass
            #save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))
            #writer.add_graph(leo, metadata[:-1])
            #writer.close()
            #train_stats.disp_stats()
          
        #dataloader = Datagenerator(config, dataset, data_type="meta_val", generate_new_metaclasses=False)
        #_, train_stats = compute_loss(leo, dataloader, train_stats, config, mode="meta_val")
        #train_stats.disp_stats()
def predict_model(config):
    pass
    #dataloader = Datagenerator(config, dataset, data_type="meta_test", generate_new_metaclasses=False)
    #_, train_stats = compute_loss(leo, dataloader ,train_stats, config, mode="meta_test")
    #train_stats.disp_stats()

def main():
    config = load_config()
    if config.train:
        train_model(config)
    else:
        predict_model(config)
    
if __name__ == "__main__":
    main()