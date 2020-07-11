# Entry point for the project
from utils import load_config
from data import Datagenerator
from model import LEO
from PIL import Image
from torchvision import transforms
from torch.autograd import variable
import argparse
import torch

parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="sample_data")
args = parser.parse_args()
dataset = args.dataset
print(dataset)

def get_in_sequence(data, batch_size, H, W):
    data = data.permute(0, 3, 4, 1, 2)
    data = data.contiguous().view(batch_size, H, W, -1)
    data = data.permute(0, 3, 1, 2)
    return data

def train_model(config):
    metatrain_dataloader = Datagenerator(dataset, config, data_type="train")
    epochs = config["hyperparameters"]["epochs"]

    for i in range(epochs):
        tr_data, tr_data_masks, val_data, val_masks = metatrain_dataloader.get_batch_data()
        print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}". \
              format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
        dim_list = list(tr_data.size())
        tr_data_ = get_in_sequence(tr_data, dim_list[0], dim_list[3], dim_list[4])
        tr_data_ = tr_data_.unsqueeze(2) #because in the sample_data num_channels is missing
        #tr_data_ = variable(tr_data_, requires_grad=True)
        model = LEO(config)
        for i, data in enumerate(tr_data_):
            data = data.clone().detach().requires_grad_(True)#torch.tensor(data, requires_grad = True)
            latents, kl = model.forward_encoder(data)
            tr_loss, adapted_classifier_weights = model.leo_inner_loop(data, latents)
            val_loss, val_accuracy = model.finetuning_inner_loop(data, tr_loss, adapted_classifier_weights)


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