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

    device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
    Leo=LEO().to(device)
    metatrain_dataloader = Datagenerator(dataset, config, data_type="meta_train")
    episodes = config.hyperparameters.episodes
    
    train_stats_store = TrainingStats() # to store training statistics
    optimizer = optim.SGD(Leo.parameters(), lr=0.1, momentum=0.9)
    if config.load_model_status == True:
        Leo,optimizer=load_model(Leo,optimizer,config.episodes_completed)
    for i in range(episodes-config.episodes_completed):
        Leo.train()
        tr_data, tr_data_masks, val_data, val_masks = metatrain_dataloader.get_batch_data()
        print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}".\
            format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
    #for loop for batch data in metatrain_dataloader
        loss,optimizer=optimize_model(Leo,batch_tr_data,batch_tr_data_masks,optimizer)
        if batch_idx % config.logging_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            save_model(Leo,optimizer,model_path,i,loss)
            config.episodes_completed=i


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