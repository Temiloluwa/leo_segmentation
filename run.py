import sys, gc
import os
import argparse
import time
import torch
import torch.optim
import numpy as np
from easydict import EasyDict as edict
from leo_segmentation.data import PascalDatagenerator, GeneralDatagenerator, TrainingStats
from leo_segmentation.model import LEO, load_model, save_model, compute_loss
from leo_segmentation.utils import load_config, check_experiment, \
    get_named_dict, log_data, load_yaml, train_logger, val_logger, \
    print_to_string_io, save_pickled_data, model_dir, update_config,\
    list_to_numpy, numpy_to_tensor

config = load_config()

try:
    shell = get_ipython().__class__.__name__
    if shell == "NoneType":
        raise NameError("Move to except branch")
except NameError:
    parser = argparse.ArgumentParser(description='Specify dataset')
    parser.add_argument("-d", "--dataset", type=str, default="pascal_5i")
    parser.add_argument("-f", "--fold", type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset
    fold = args.fold
    

def load_model_and_params(dataset, data_type):
    """ Loads saved model """
    device = torch.device("cuda:0" if torch.cuda.is_available()
                                      and config.use_gpu else "cpu")
    Datagenerator = PascalDatagenerator if dataset == "pascal_5i" \
            else GeneralDatagenerator

    dataloader = Datagenerator(dataset, data_type)
    transformers = (dataloader.transform_image, dataloader.transform_mask)
    metadata = dataloader.get_batch_data()
    data_dict = get_named_dict(metadata, 0)
    leo, train_stats = load_model(device, data_dict)
    return leo, train_stats, transformers

def train_model(config, dataset, fold=None):
    """Trains Model

    Args:
        config (dict): Config dictionary
        dataset (str): Training dataset
        fold (int): Supplied if Pascal 5i data is to be trained

    Raises:
        ValueError: If no Pascal 5i data fold is supplied

    Returns:
        leo (object): Leo Model
    """
    Datagenerator = PascalDatagenerator if dataset == "pascal_5i" \
            else GeneralDatagenerator
    device = torch.device("cuda:0" if torch.cuda.is_available()
                                      and config.use_gpu else "cpu")
    
    if dataset == "pascal_5i" and fold is None:
        raise ValueError("Supply fold to train the model")
    
    update_config({'selected_data': dataset})
    update_config({'fold': fold})

    if check_experiment(config):
        # Load saved model and parameters
        leo, train_stats, transformers = \
                load_model_and_params(dataset, data_type="meta_train")
        episodes_completed = train_stats.episode
    else:
        # Train a fresh model
        leo = LEO().to(device)
        train_stats = TrainingStats()
        episodes_completed = 0

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
        
        # meta-val stage
        if episode % config.meta_val_interval == 0:
            dataloader = Datagenerator(dataset, data_type="meta_val")
            train_stats.set_mode("meta_val")
            metadata = dataloader.get_batch_data()
            leo.mode = "meta_val"
            _, train_stats = compute_loss(leo, metadata, train_stats,
                                          transformers, mode="meta_val")
        
        if episode >= config.checkpoint_start and \
            train_stats.best_episode  == episode:
            save_model(leo, config, train_stats)
        
        if episode % config.display_stats_interval == 0:
            train_stats.disp_stats()
        episode_time = (time.time() - start_time) / 60
        log_msg = f"Episode: {episode} |  Episode Time: {episode_time:0.03f} minutes\n"
        print_to_string_io(log_msg, False, train_logger)
        episode_times.append(episode_time)


    leo = predict_model(dataset, leo, train_stats, transformers)
    log_msg = f"Total Model Training Time {np.sum(episode_times):0.03f} minutes"
    print_to_string_io(log_msg, False, train_logger)
    train_logger.debug("End time")
    return leo


def predict_model(dataset, leo, train_stats, transformers):
    """Create Predictions for Metatest

    Args:
        dataset (str): Name of training dataset
        leo (object): leo model 
        train_stats (object): train_stats object
        transformers (tuple): tuple of image and mask data transformers

    Returns:
        leo (object): model object
    """
    Datagenerator = PascalDatagenerator if dataset == "pascal_5i" \
            else GeneralDatagenerator
    data_type = "meta_test"
    if dataset == "pascal_5i":
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

def evaluate_model(dataset, 
                    support_imgs_or_paths,
                    support_masks_or_paths,
                    query_imgs_or_paths,
                    dataloader):
    """Evaluate model

    Args:
        dataset (str): Dataset used to train the model
        support_imgs_or_paths (tensor/list): tensor/List of support images or imgs
        query_imgs_or_paths (tensor/list): tensor/List of paths to query images 
        support_masks_or_paths (tensor/list): tensor/List of paths to support masks
        dataloader(object): dataloader

    Returns:
        prediction (tensor): predicted mask
        leo (object): model
        transformers (tuple): tuple of image transformers
    """
    if config.train:
        raise ValueError("Set config.train to False")
    
    support_imgs = support_imgs_or_paths
    support_masks = support_masks_or_paths
    query_imgs = query_imgs_or_paths
    transformers = (dataloader.transform_image, dataloader.transform_mask)
    if type(support_imgs_or_paths) is list:
        support_imgs = list_to_numpy(support_imgs_or_paths, transformers[0])
        support_imgs = numpy_to_tensor(support_imgs)

    if type(support_masks_or_paths) is list:
        support_masks = list_to_numpy(support_masks_or_paths, transformers[1])
        support_masks = numpy_to_tensor(support_masks)
    
    if type(query_imgs_or_paths) is list:
        query_imgs = list_to_numpy(query_imgs_or_paths, transformers[0])
        query_imgs = numpy_to_tensor(query_imgs)
      
    data_dict = edict({'tr_imgs': support_imgs,
                        'tr_masks': support_masks,
                        'val_imgs': query_imgs,
                        'val_masks': None})
    
    leo, _, _ = load_model_and_params(dataset, data_type="meta_val")
    seg_weight_grad, features, we, wd = leo.leo_inner_loop(data_dict.tr_imgs, data_dict.tr_masks)
    prediction = leo.finetuning_inner_loop(data_dict, features, seg_weight_grad,
                                      transformers, mode="meta_val", we=we, wd=wd)
    
    return prediction, leo, transformers

def main():
    if config.train:
        train_model(config, dataset, fold)

if __name__ == "__main__":
    main()
