# Entry point for the project
from utils import load_config, optimize_model, load_model,save_model
from data import Datagenerator, TrainingStats
from model import LEO
import torch.optim as optim
import argparse
import torch

parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
args = parser.parse_args()
dataset = args.dataset[0]
model_path='' 
def train_model(config):
    device = torch.device("cuda")
    metatrain_dataloader = Datagenerator(dataset, config, data_type="meta_train")
    episodes = config.hyperparameters.episodes
    episodes_completed=0
    train_stats_store = TrainingStats() # to store training statistics
    optimizer = optim.SGD(LEO.parameters(), lr=0.1, momentum=0.9)
    if load_model == True:
        LEO,optimizer,episodes_completed=load_model(LEO,optimizer,model_path)
    
    for i in range(episodes-episodes_completed):
        LEO.train()
        tr_data, tr_data_masks, val_data, val_masks = metatrain_dataloader.get_batch_data()
        print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}".\
            format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
    #for loop for batch data in metatrain_dataloader
        loss,optimizer=optimize_model(LEO,batch_tr_data,batch_tr_data_masks,optimizer)
        if i % 1000 == 0: #saving model for every 1000 epoch
            save_model(LEO,optimizer,model_path,i,loss)


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