# Entry point for the project
from utils import load_config
from data import Datagenerator
from model import LEO
import argparse
import torch
import torch.optim
from easydict import EasyDict as edict

parser = argparse  .ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="sample_data")
args = parser.parse_args()
dataset = args.dataset
print(dataset)

def get_in_sequence(data):
    dim_list = list(data.size())
    data = data.permute(2, 3, 0, 1)
    data = data.contiguous().view(dim_list[2], dim_list[3], -1)
    data = data.permute(2, 0, 1)
    data = data.unsqueeze(1) #because in the sample_data num_channels is missing
    data = data
    return data

def train_model(config):
    metatrain_dataloader = Datagenerator(dataset, config, data_type="meta_train")
    metaval_dataloader = Datagenerator(dataset, config, data_type="meta_val")
    metatest_dataloader = Datagenerator(dataset, config, data_type="meta_test")

    epochs = config["hyperparameters"]["episodes"]

    #num_classes = config["data_params"]["num_classes"]
    #num_tr_examples_per_class_train = config["data_params"]["n_train_per_class"]["meta_train"]
    for i in range(epochs):
        mode = 'meta_train'
        total_val_loss, model = LEO_model(config, metatrain_dataloader, mode)
        metatrain_loss = sum(total_val_loss)/len(total_val_loss)
        metatrain_gradients, metatrain_variables = model.grads_and_vars(metatrain_loss)
        optimizer = torch.optim.Adam(metatrain_gradients, lr=config["hyperparameters"]["outer_loop_lr"])
        optimizer.step()
        optimizer.zero_grad()
        mode = 'meta_val'
        total_val_loss, _ = LEO_model(config,  metaval_dataloader, mode)
        mode = 'meta_test'
        total_val_loss, _ = LEO_model(config, metatest_dataloader, mode)
    print("completed Meta_train, Meta_test, Meta_val")

def LEO_model(config, metatrain_dataloader, mode):
    tr_data, tr_data_masks, val_data, val_masks = metatrain_dataloader.get_batch_data()
    print(mode)
    print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}". \
          format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))

    model = LEO(config, mode)
    print(len(tr_data))
    total_val_loss = []
    for i in range(len(tr_data)):
        data_dict = {'tr_data_orig': tr_data[i], 'tr_data': get_in_sequence(tr_data[i]),
                     'tr_data_masks': get_in_sequence(tr_data_masks[i]),
                     'val_data_orig': val_data[i], 'val_data': get_in_sequence(val_data[i]),
                     'val_data_masks': get_in_sequence(val_masks[i])}

        latents, kl = model.forward_encoder(data_dict["tr_data"])
        per_batch_tr_loss, adapted_segmentation_weights = model.leo_inner_loop(data_dict, latents)
        per_batch_val_loss = model.finetuning_inner_loop(data_dict, per_batch_tr_loss, adapted_segmentation_weights)
        per_batch_val_loss += config["kl_weight"] * kl
        total_val_loss.append(per_batch_val_loss)
    return total_val_loss, model

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