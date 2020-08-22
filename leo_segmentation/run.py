from utils import load_config, check_experiment, load_model, save_model,\
    display_data_shape,get_named_dict
from data import Datagenerator, TrainingStats
from model import LEO
from  torch.nn import MSELoss
from easydict import EasyDict as edict
import torch.optim as optim
import argparse
import torch 
import torch.optim


parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
args = parser.parse_args()
dataset = args.dataset

def compute_loss(model, meta_dataloader, train_stats, config, mode="meta_train"):
    """
    Computes the  outer loop loss

    Args:
        model (object) : leo model
        meta_dataloader (object): Dataloader 
        train_stats: (object): train stats objecdt
        config (dict): config
        mode (str): meta_train, meta_val or meta_test
    Returns:
        (tuple) total_val_loss (list), train_stats
    """
    metadata = meta_dataloader.get_batch_data()
    num_tasks = len(metadata[0])
    display_data_shape(metadata)
    total_val_loss = []
    for batch in range(num_tasks):
        data_dict = get_named_dict(metadata, batch)
        display_data_shape(data_dict)
        latents = model.forward_encoder(data_dict.tr_data)
        tr_loss, adapted_seg_weights = model.leo_inner_loop(\
                        data_dict.tr_data, latents, data_dict.tr_data_masks)
        val_loss = model.finetuning_inner_loop(data_dict, tr_loss, adapted_seg_weights)
        total_val_loss.append(val_loss)
    stats_data = {
        "mode": mode,
        "kl_loss": 0,
        "total_val_loss":sum(total_val_loss)/len(total_val_loss)
    }
    train_stats.update_stats(**stats_data)
    return total_val_loss, train_stats
    

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
        #meta-train
        dataloader = Datagenerator(config, dataset, data_type="meta_train", generate_new_metaclasses=False)
        metatrain_loss, train_stats = compute_loss(leo, dataloader, train_stats, config)
        metatrain_loss = sum(metatrain_loss)/len(metatrain_loss)
        optimizer.zero_grad()
        metatrain_loss.backward()
        optimizer.step()

        if episode % config.checkpoint_interval == 0:
            save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))

        train_stats.disp_stats()
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