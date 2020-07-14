# Entry point for the project
from utils import load_config, check_experiment, optimize_model, load_model, save_model
from data import Datagenerator, TrainingStats
from model import LEO
from easydict import EasyDict as edict
import torch.optim as optim
import argparse
import torch

parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
args = parser.parse_args()
dataset = args.dataset[0]

def train_model(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
    if check_experiment(config.experiment):
        leo, optimizer, stats = load_model(leo, optimizer, config)
        episodes_completed = stats["episodes"]
        leo.eval()
        leo = leo.to(device)
        train_stats = TrainingStats()
        train_stats.update_stats(**stats)
    else:
        leo = LEO().to(device)
        train_stats = TrainingStats()
        episodes_completed = 0
    
    metatrain_dataloader = Datagenerator(dataset, config, data_type="meta_train")
    episodes =  config.hyperparameters.episodes
    optimizer = optim.SGD(leo.parameters(), lr=0.1, momentum=0.9)

     
    for episode in range(1, episodes+1-episodes_completed):
        tr_data, tr_data_masks, val_data, val_masks = metatrain_dataloader.get_batch_data()
        loss, optimizer = optimize_model(leo, tr_data, tr_data_masks, val_masks, optimizer, train_stats)
        stats = edict(train_stats.get_latest_stats())
        print(f"episode:{stats.episode:03d}, loss:{stats.loss:02f}, iou:{stats.int_ov_union:02f}")
        
        if episodes % config.checkpoint_interval == 0:
            print("tr_data shape: {}, tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}".\
                format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
            save_model(model, optimizer, config, train_stats.get_latest_stats())


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