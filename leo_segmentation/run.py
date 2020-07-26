# Entry point for the project
from utils import load_config
from data import Datagenerator
from model import LEO
import argparse
import torch
import torch.optim


parser = argparse  .ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="sample_data")
args = parser.parse_args()
dataset = args.dataset
print(dataset)

class Meta_Learning():
    """
    contains functions to perform meta-learning task
    inspired from: https://github.com/deepmind/leo we implemented the model for segmentation task
    """
    def __init__(self, config):
        """
        Args:
            config (dict): contains all the model initial configuration parameters and hyperparameters
        """
        self.config = config
        self.model = LEO(config=config)
        self.mode = "meta_train"

    @staticmethod
    def get_in_sequence(data):
        """
        converts the tensor data (num_class, num_eg_per_class, img_dim) to ( (num_class * num_eg_per_class), img_dim)
        Args:
            data (tensor): (num_class, num_eg_per_class, H, W) # currently channel is missing
        Returns:
            data (tensor): (total_num_eg, Channel, H, W)
        """
        dim_list = list(data.size())
        data = data.permute(2, 3, 0, 1)
        data = data.contiguous().view(dim_list[2], dim_list[3], -1)
        data = data.permute(2, 0, 1)
        data = data.unsqueeze(1) #because in the sample_data num_channels is missing
        return data

    def train_model(self):
        """
        Trains the model to perform meta learning task
        """
        metatrain_dataloader = Datagenerator(dataset, self.config, data_type="meta_train")
        metaval_dataloader = Datagenerator(dataset, self.config, data_type="meta_val")
        metatest_dataloader = Datagenerator(dataset, self.config, data_type="meta_test")

        epochs = self.config["hyperparameters"]["episodes"]

        for i in range(epochs):
            self.mode = 'meta_train'
            total_val_loss = self.LEO_model(metatrain_dataloader)
            metatrain_loss = sum(total_val_loss)/len(total_val_loss)
            metatrain_gradients, metatrain_variables = self.model.grads_and_vars(metatrain_loss)
            optimizer = torch.optim.Adam(metatrain_gradients, lr=self.config["hyperparameters"]["outer_loop_lr"])
            optimizer.step()
            optimizer.zero_grad()
            self.mode = 'meta_val'
            total_val_loss = self.LEO_model(metaval_dataloader)
            self.mode = 'meta_test'
            total_val_loss = self.LEO_model(metatest_dataloader)
        print("completed Meta_train, Meta_test, Meta_val")

    def LEO_model(self, meta_dataloader):
        """
        Gets the meta data containing batches of meta_train, meta_val and meta_test depending on the mode and
        perfroms LEO optimization.

        Args:
            meta_dataloader (object) : which uses get_batch_data function to return data tensor in the following format
                train_data: (Batch, num_classes, num_eg_per_class, H, W) #channels missing currently (added manually while sequencing)
                train_data_masks: (Batch, num_classes, num_eg_per_class, channels, H, W)
                val_data: (Batch, num_classes, num_eg_per_class, H, W) #channels missing currently (added manually while sequencing)
                val_data_masks: (Batch, num_classes, num_eg_per_class, channels, H, W)
        Returns:
                total_val_loss (list): contains list of validation loss of batches of (val_data, val_data_masks)
        """
        tr_data, tr_data_masks, val_data, val_masks = meta_dataloader.get_batch_data()
        print("tr_data shape: {},tr_data_masks shape: {}, val_data shape: {},val_masks shape: {}". \
              format(tr_data.size(), tr_data_masks.size(), val_data.size(), val_masks.size()))
        print(len(tr_data))
        total_val_loss = []
        for i in range(len(tr_data)):
            data_dict = {'tr_data_orig': tr_data[i], 'tr_data': self.get_in_sequence(tr_data[i]),
                         'tr_data_masks': self.get_in_sequence(tr_data_masks[i]),
                         'val_data_orig': val_data[i], 'val_data': self.get_in_sequence(val_data[i]),
                         'val_data_masks': self.get_in_sequence(val_masks[i])}

            latents, kl_loss = self.model.forward_encoder(data_dict["tr_data"], self.mode)
            per_batch_tr_loss, adapted_segmentation_weights = self.model.leo_inner_loop(data_dict, latents)
            per_batch_val_loss = self.model.finetuning_inner_loop(data_dict, per_batch_tr_loss, adapted_segmentation_weights)
            per_batch_val_loss += self.config["kl_weight"] * kl_loss
            total_val_loss.append(per_batch_val_loss)
        return total_val_loss



def main():
    config = load_config()
    if config["train"]:
        meta_model = Meta_Learning(config)
        meta_model.train_model()

if __name__ == "__main__":
    main()