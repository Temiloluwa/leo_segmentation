from functools import partial
import os, argparse, torch, gc, time
import numpy as np
import torch.optim as optim
from torch.nn import MSELoss
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from data import Datagenerator, TrainingStats
from model_combined2 import LEO, load_model, save_model
from utils import load_config, check_experiment, get_named_dict, log_data

try:
    shell = get_ipython().__class__.__name__
    dataset = "pascal_voc"
except NameError:
    parser = argparse.ArgumentParser(description='Specify train or inference dataset')
    parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
    args = parser.parse_args()
    dataset = args.dataset


def load_model_and_params(config, device):
    """Loads model and accompanying saved parameters"""
    leo, optimizer, stats = load_model(config)
    episodes_completed = stats["episode"]
    leo.eval()
    leo = leo.to(device)
    train_stats = TrainingStats(config)
    train_stats.set_episode(episodes_completed)
    train_stats.update_stats(**stats)
    return leo, optimizer, train_stats


def train_model(config):
    """Trains Model"""
    writer = SummaryWriter(os.path.join(config.data_path, "models", str(config.experiment.number)))
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")
    if check_experiment(config):
        leo, optimizer, train_stats = load_model_and_params(config, device)
    else:
        leo = LEO(config, writer).to(device)
        train_stats = TrainingStats(config)
        episodes_completed = 0
        optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)

    model_root = os.path.join(os.path.dirname(__file__), config.data_path, "models")
    log_file = os.path.join(model_root, "experiment_{}".format(config.experiment.number), "val_log.txt")
    total_episodes = config.hyperparameters.episodes
    episode_times = []
    for episode in range(episodes_completed + 1, total_episodes + 1):
        start_time = time.time()
        train_stats.set_episode(episode)
        train_stats.set_mode("meta_train")
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        metatrain_loss, train_stats, _ = leo.compute_loss(metadata, train_stats, mode="meta_train")
        optimizer.zero_grad()
        metatrain_loss.backward()
        optimizer.step()
        train_stats.disp_stats()
        if episode % config.checkpoint_interval == 0:
            save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))
            # writer.add_graph(leo, metadata[:-1])
            # writer.close()
        if episode % config.validation_step == 0:
            dataloader = Datagenerator(config, dataset, data_type="meta_val")
            train_stats.set_mode("meta_val")
            metadata = dataloader.get_batch_data()
            _, train_stats, _ = leo.compute_loss(metadata, train_stats, mode="meta_val")
            train_stats.disp_stats()
        episode_time = (time.time() - start_time) / 60
        log_msg = f"Episode: {episode}, Episode Time: {episode_time:0.03f} minutes\n"
        print(log_msg)
        log_data(log_msg, log_file)
        episode_times.append(episode_time)
        if episode != total_episodes:
            del metadata
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        else:
            model_and_params = leo, _, train_stats
            seg_weights = predict_model(config, model_and_params)

    log_msg = f"Total Model Training Time {np.sum(episode_times):0.03f} minutes\n"
    print(log_msg)
    log_data(log_msg, log_file)
    return leo, metadata, seg_weights


def predict_model(config, model_and_params):
    """Implement Predicion on Meta-Test"""
    leo, _, train_stats = model_and_params
    dataloader = Datagenerator(config, dataset, data_type="meta_test")
    train_stats.set_mode("meta_test")
    metadata = dataloader.get_batch_data()
    _, train_stats, seg_weights = leo.compute_loss(metadata, train_stats, mode="meta_test")
    train_stats.disp_stats()
    return seg_weights


def main():
    config = load_config()
    if config.train:
        train_model(config)
    else:
        model_and_params = load_model_and_params(config)
        predict_model(config, model_and_params)


if __name__ == "__main__":
    main()