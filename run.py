from functools import partial
import os, argparse, torch, gc
import numpy as np
import torch.optim as optim
from torch.nn import MSELoss
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from leo_segmentation.data import Datagenerator, TrainingStats
from leo_segmentation.model import LEO, load_model, save_model
from leo_segmentation.utils import load_config, check_experiment, get_named_dict

try:
    shell = get_ipython().__class__.__name__
    dataset = "pascal_voc"
except NameError:
    parser = argparse.ArgumentParser(description='Specify train or inference dataset')
    parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
    args = parser.parse_args()
    dataset = args.dataset

def load_model_and_params(config):
    """Loads model and accompanying parameters"""
    leo, optimizer, stats = load_model(config)
    episodes_completed = stats["episode"]
    leo.eval()
    leo = leo.to(device)
    train_stats = TrainingStats(config)
    train_stats.set_episode(episodes_completed)
    train_stats.update_stats(**stats)
    return leo, optimizer, train_stats

def train_model(config):
    """Trains Model"""
    #writer = SummaryWriter(os.path.join(config.data_path, "models", str(config.experiment.number)))
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
    if check_experiment(config):
        leo, optimizer, train_stats = load_model_and_params()
    else:
        leo = LEO(config).to(device)
        train_stats = TrainingStats(config)
        episodes_completed = 0
        optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)

    episodes =  config.hyperparameters.episodes

    for episode in range(episodes_completed+1, episodes+1):
        train_stats.set_episode(episode)
        train_stats.set_mode("meta_train")
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        metatrain_loss, train_stats = leo.compute_loss(metadata, train_stats, mode="meta_train")
        optimizer.zero_grad()
        metatrain_loss.backward()
        optimizer.step()

        if episode % config.checkpoint_interval == 0:
            save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))
            #writer.add_graph(leo, metadata[:-1])
            #writer.close()
        
        dataloader = Datagenerator(config, dataset, data_type="meta_val")
        train_stats.set_mode("meta_val")
        metadata = dataloader.get_batch_data()
        _, train_stats = leo.compute_loss(metadata, train_stats, mode="meta_val")
        train_stats.disp_stats()

        if episode != episodes:
            del metadata
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        model_and_params = leo, _ , train_stats
        predict_model(config, model_and_params)

        return leo, metadata
        
def predict_model(config, model_and_params):
    leo, _ , train_stats = model_and_params
    dataloader = Datagenerator(config, dataset, data_type="meta_test")
    train_stats.set_mode("meta_test")
    metadata = dataloader.get_batch_data()
    _, train_stats = leo.compute_loss(metadata, train_stats, mode="meta_test")
    train_stats.disp_stats()

def main():
    config = load_config()
    if config.train:
        train_model(config)
    else:
        model_and_params = load_model_and_params(config)
        predict_model(config, model_and_params)
    
if __name__ == "__main__":
    main()