from data import Datagenerator, TrainingStats
from model import LEO, load_model, save_model
from  torch.nn import MSELoss
from easydict import EasyDict as edict
from utils import load_config, check_experiment
import torch.optim as optim
import argparse
import torch 
import torch.optim
import gc

parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
args = parser.parse_args()
dataset = args.dataset

def train_model(config):
    """Trains Model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
    if check_experiment(config):
        leo, optimizer, stats = load_model(config)
        episodes_completed = stats["episode"]
        leo.eval()
        leo = leo.to(device)
        train_stats = TrainingStats(config)
        train_stats.set_episode(episodes_completed)
        train_stats.update_stats(**stats)
    else:
        leo = LEO(config).to(device)
        train_stats = TrainingStats(config)
        episodes_completed = 0
    
    episodes =  config.hyperparameters.episodes
    optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)

    for episode in range(episodes_completed+1, episodes+1):
        train_stats.set_episode(episode)
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        class_in_metadata = dataloader.classes_dict["meta_train"]
        metatrain_loss, train_stats = leo.compute_loss(metadata, train_stats, config, mode="meta_train")
        optimizer.zero_grad()
        metatrain_loss.backward()
        optimizer.step()

        if episode % config.checkpoint_interval == 0:
            save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))
        
        leo.evaluate_val_data(metadata, class_in_metadata, train_stats)
        train_stats.disp_stats()
        del metadata
        gc.collect()

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
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