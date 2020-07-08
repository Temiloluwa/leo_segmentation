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


def train_model(config):
    metatrain_dataloader = Datagenerator(dataset, config, data_type="train")
    epochs = config["hyperparameters"]["epochs"]
    for i in range(epochs):
        tr_data, tr_data_masks, val_data, val_masks = metatrain_dataloader.get_batch_data()
        tr_data_ = variable(tr_data, requires_grad=True)
        print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}". \
              format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
        model = LEO(config)
        for i, data in enumerate(tr_data_):
            latents = model.forward_encoder(data)
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