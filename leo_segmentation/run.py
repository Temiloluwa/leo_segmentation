from utils import load_config, check_experiment, load_model, save_model, \
    display_data_shape, get_named_dict
from data import Datagenerator, TrainingStats
from model import LEO
from torch.nn import MSELoss
from easydict import EasyDict as edict
import torch.optim as optim
import argparse
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser(description='Specify train or inference dataset')
parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
args = parser.parse_args()
dataset = args.dataset


def compute_loss(model, meta_dataloader, train_stats, config, device, iteration = 0, mode="meta_train"):
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
    kl_losses = []
    total_val_loss = []
    for batch in range(num_tasks):
        data_dict = get_named_dict(metadata, batch)
        print(data_dict.tr_data.is_cuda)
        latents, kl_loss = model.forward_encoder(data_dict.tr_data, mode, device)
        per_batch_tr_loss, adapted_segmentation_weights = model.leo_inner_loop(data_dict, latents, device)
        per_batch_val_loss = model.finetuning_inner_loop(data_dict, per_batch_tr_loss, adapted_segmentation_weights, device)
        per_batch_val_loss = per_batch_val_loss + (config.kl_weight * kl_loss)
        if mode == "meta_train":
            writer.add_scalar('train_loss', per_batch_tr_loss, iteration)
            writer.add_scalar('val_loss', per_batch_val_loss, iteration)
            #print(iteration)
            iteration += 1
        total_val_loss.append(per_batch_val_loss)
        kl_losses.append(kl_loss)
        #print("Batch number", batch)
    stats_data = {
        "mode": mode,
        "kl_loss": sum(kl_losses) / len(kl_losses),
        "total_val_loss": sum(total_val_loss) / len(total_val_loss)
    }
    train_stats.update_stats(**stats_data)
    return total_val_loss, train_stats, iteration


def train_model(config):
    """Trains Model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    episodes = config.hyperparameters.episodes
    # outerloop
    # optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)
    # criterion = MSELoss(reduction="mean")
    itr = 1
    for episode in range(episodes_completed + 1, episodes + 1):
        train_stats.set_episode(episode)
        dataloader = Datagenerator(config, dataset, data_type="meta_train", generate_new_metaclasses=False)
        metatrain_loss, train_stats, itr = compute_loss(leo, dataloader, train_stats, config, device, iteration = itr)
        metatrain_loss = sum(metatrain_loss) / len(metatrain_loss)
        writer.add_scalar('metatrain_loss', metatrain_loss, episode)
        metatrain_gradients, metatrain_variables = leo.grads_and_vars(metatrain_loss)
        # confirm this
        optimizer = torch.optim.Adam(metatrain_gradients, lr=config.hyperparameters.outer_loop_lr)
        optimizer.step()
        optimizer.zero_grad()
        if episode % config.checkpoint_interval == 0:
            save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))

        train_stats.disp_stats()
        dataloader = Datagenerator(dataset, config, data_type="meta_val")
        _, train_stats, _ = compute_loss(leo, dataloader, train_stats, config, device, mode="meta_val")
        train_stats.disp_stats()

        dataloader = Datagenerator(dataset, config, data_type="meta_test")
        _, train_stats, _ = compute_loss(leo, dataloader, train_stats, config, device, mode="meta_test")
        train_stats.disp_stats()


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
