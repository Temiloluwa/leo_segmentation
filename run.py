import sys, gc
import os
import argparse
import time
import torch
import torch.optim
import numpy as np
from easydict import EasyDict as edict
from leo_segmentation.data import Datagenerator, TrainingStats
from leo_segmentation.model import LEO, load_model, save_model, compute_loss
from leo_segmentation.utils import load_config, check_experiment, \
    get_named_dict, log_data, load_yaml, train_logger, val_logger, \
    print_to_string_io, save_pickled_data, model_dir

config = load_config()
#update_config({'selected_data': dataset})

try:
    shell = get_ipython().__class__.__name__
    if shell == "NoneType":
        raise NameError("Move to except branch")
except NameError:
    parser = argparse.ArgumentParser(description='Specify dataset')
    parser.add_argument("-d", "--dataset", type=str, default="fss1000")
    args = parser.parse_args()
    dataset = args.dataset


def load_model_and_params(device):
    """Loads model and accompanying saved parameters"""
    leo, optimizer, stats = load_model()
    episodes_completed = stats["episode"]
    leo.eval()
    leo = leo.to(device)
    train_stats = TrainingStats()
    train_stats.set_episode(episodes_completed)
    train_stats.set_mode(stats["mode"])
    train_stats.update_stats(**stats)
    return leo, optimizer, train_stats, episodes_completed


def train_model(config, dataset):
    """Trains Model"""
    # writer = SummaryWriter(os.path.join(config.data_path, "models",
    # str(config.experiment.number)))
    device = torch.device("cuda:0" if torch.cuda.is_available()
                                      and config.use_gpu else "cpu")
    if check_experiment(config):
        # Load saved model and parameters
        leo, optimizer, train_stats, episodes_completed = load_model_and_params(device)
    else:
        # Train a fresh model
        leo = LEO().to(device)
        train_stats = TrainingStats()
        episodes_completed = 0
        optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)

    leo.freeze_encoder()
    episodes = config.hyperparameters.episodes
    episode_times = []
    train_logger.debug("Start time")

    for episode in range(episodes_completed + 1, episodes + 1):
        start_time = time.time()
        train_stats.set_episode(episode)
        train_stats.set_mode("meta_train")
        leo.mode = 'meta_train'
        # meta-train stage
        dataloader = Datagenerator(dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        transformers = (dataloader.transform_image, dataloader.transform_mask)
        _, train_stats = compute_loss(leo, metadata, train_stats, transformers)
        if episode % config.checkpoint_interval == 0:
            save_model(leo, optimizer, config,
                       edict(train_stats.get_latest_stats()))
        
        # meta-val stage
        if episode % config.meta_val_interval == 0:
            dataloader = Datagenerator(dataset, data_type="meta_val")
            train_stats.set_mode("meta_val")
            metadata = dataloader.get_batch_data()
            leo.mode = "meta_val"
            _, train_stats = compute_loss(leo, metadata, train_stats,
                                          transformers, mode="meta_val")
        
        if episode % config.display_stats_interval == 0:
            train_stats.disp_stats()
        episode_time = (time.time() - start_time) / 60
        log_msg = f"Episode: {episode} |  Episode Time: {episode_time:0.03f} minutes\n"
        print_to_string_io(log_msg, False, train_logger)
        episode_times.append(episode_time)

    model_and_params = leo, None, train_stats
    leo = predict_model(dataset, model_and_params, transformers)
    log_msg = f"Total Model Training Time {np.sum(episode_times):0.03f} minutes"
    print_to_string_io(log_msg, False, train_logger)
    train_logger.debug("End time")
    return leo


def predict_model(dataset, model_and_params, transformers):
    """Implement Predicion on Meta-Test"""
    config = load_config()
    leo, _, train_stats = model_and_params
    data_type = "meta_test"
    if data_type == "pascal_5i":
        data_type = "meta_val"

    dataloader = Datagenerator(dataset, data_type=data_type)
    train_stats.set_mode("meta_test")
    metadata = dataloader.get_batch_data()
    _, train_stats = compute_loss(leo, metadata, train_stats, transformers,
                                  mode=data_type)
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
    if config.train:
        train_model(config, dataset)
    else:
        def evaluate_model():
            dataloader = Datagenerator(dataset, data_type="meta_train")
            img_transformer = dataloader.transform_image
            mask_transformer = dataloader.transform_mask
            transformers = (img_transformer, mask_transformer)
            model_and_params = load_model_and_params()
            return predict_model(dataset, model_and_params,
                                 transformers)

        evaluate_model()


if __name__ == "__main__":
    main()
