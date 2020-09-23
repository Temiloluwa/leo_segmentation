import sys
import os
import argparse
import time
import gc
import torch
import torch.optim
import numpy as np
from easydict import EasyDict as edict
from leo_segmentation.data import Datagenerator, TrainingStats
from leo_segmentation.model import LEO, load_model, save_model
from leo_segmentation.utils import load_config, check_experiment,\
    get_named_dict, log_data, load_yaml, train_logger, val_logger, \
    print_to_string_io, save_pickled_data, model_dir

try:
    shell = get_ipython().__class__.__name__
    if shell == "NoneType":
        raise NameError("Move to except branch")
except NameError:
    parser = argparse.ArgumentParser(description='Specify dataset')
    parser.add_argument("-d", "--dataset", type=str, default="pascal_voc_raw")
    args = parser.parse_args()
    dataset = args.dataset


def load_model_and_params(config):
    """Loads model and accompanying saved parameters"""
    leo, optimizer, stats = load_model(config)
    episodes_completed = stats["episode"]
    leo.eval()
    leo = leo.to(device)
    train_stats = TrainingStats(config)
    train_stats.set_episode(episodes_completed)
    train_stats.update_stats(**stats)
    return leo, optimizer, train_stats


def train_model(config, dataset):
    """Trains Model"""
    # writer = SummaryWriter(os.path.join(config.data_path, "models",
    # str(config.experiment.number)))
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and config.use_gpu else "cpu")
    if check_experiment(config):
        # Load saved model and parameters
        leo, optimizer, train_stats = load_model_and_params(config)
    else:
        # Train a fresh model
        leo = LEO(config).to(device)
        train_stats = TrainingStats(config)
        episodes_completed = 0
    episodes = config.hyperparameters.episodes
    episode_times = []
    train_logger.debug(f"Start time")
    # leo_parameters = [params for name, params in leo.named_parameters() if "encoder" not in name]
    # optimizer_leo = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)
    # optimizer_maml = torch.optim.Adam([leo.seg_weight, leo.seg_bias], lr=config.hyperparameters.outer_loop_lr)
    for episode in range(episodes_completed+1, episodes+1):
        start_time = time.time()
        train_stats.set_episode(episode)
        train_stats.set_mode("meta_train")
        # meta-train stage
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        transformers = (dataloader.transform_image, dataloader.transform_mask)
        _, train_stats = leo.compute_loss(metadata, train_stats, transformers)
        if episode % config.checkpoint_interval == 0:
            save_model(leo, optimizer, config,
                       edict(train_stats.get_latest_stats()))
        # meta-val stage
        if episode % config.meta_val_interval == 0:
            dataloader = Datagenerator(config, dataset, data_type="meta_val")
            train_stats.set_mode("meta_val")
            metadata = dataloader.get_batch_data()
            _, train_stats = leo.compute_loss(metadata, train_stats,
                                              transformers, mode="meta_val")
            train_stats.disp_stats()
        episode_time = (time.time() - start_time)/60
        log_msg = print_to_string_io(f"Episode: {episode}, Episode Time:\
            {episode_time:0.03f} minutes", False)
        train_logger.debug(log_msg)
        episode_times.append(episode_time)
        del metadata
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # optimizer_leo.zero_grad()
        # optimizer_maml.zero_grad()
        # metatrain_loss.backward()
        # optimizer_leo.step()
        # optimizer_maml.step()
    model_and_params = leo, None, train_stats
    leo = predict_model(config, dataset, model_and_params, transformers)
    log_msg = print_to_string_io(f"Total Model Training Time \
        {np.sum(episode_times):0.03f} minutes", False)
    train_logger.debug(log_msg)
    train_logger.debug(f"End time")
    return leo


def predict_model(config, dataset, model_and_params, transformers):
    """Implement Predicion on Meta-Test"""
    leo, _, train_stats = model_and_params
    dataloader = Datagenerator(config, dataset, data_type="meta_test")
    train_stats.set_mode("meta_test")
    metadata = dataloader.get_batch_data()
    _, train_stats = leo.compute_loss(metadata, train_stats, transformers,
                                      mode="meta_test")
    train_stats.disp_stats()
    experiment = config.experiment
    for mode in ["meta_train", "meta_val", "meta_test"]:
        stats_df = train_stats.get_stats(mode)
        ious_df = train_stats.get_ious(mode)
        stats_df.to_pickle(os.path.join(model_dir,
                           f"experiment_{mode}_{experiment.number}_stats.pkl"))
        ious_df.to_pickle(os.path.join(model_dir,
                          f"experiment_{mode}_{experiment.number}_ious.pkl"))
    log_data("************** Hyperparameters Used ************\n",
             os.path.join(model_dir, "train_log.txt"))
    msg = print_to_string_io(config.hyperparameters, True)
    log_data(msg, os.path.join(model_dir, "train_log.txt"))
    return leo


def main():
    config = load_config()
    if config.train:
        train_model(config, dataset)
    else:
        def evaluate_model():
            dataloader = Datagenerator(config, dataset, data_type="meta_train")
            img_transformer = dataloader.transform_image
            mask_transformer = dataloader.transform_mask
            transformers = (img_transformer, mask_transformer)
            model_and_params = load_model_and_params(config)
            return predict_model(config, dataset, model_and_params,
                                 transformers)
        evaluate_model()


if __name__ == "__main__":
    main()
