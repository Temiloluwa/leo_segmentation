import sys, os, argparse, time, numpy as np, tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from leo_segmentation.data import Datagenerator, TrainingStats
from leo_segmentation.model import LEO, load_model, save_model
from leo_segmentation.utils import load_config, check_experiment, get_named_dict, \
                        log_data, load_yaml, train_logger, val_logger, print_to_string_io, \
                        save_pickled_data, model_dir
from easydict import EasyDict as edict
from IPython import get_ipython

try:
    shell = get_ipython().__class__.__name__
    dataset = "pascal_voc_raw"
except NameError:
    parser = argparse.ArgumentParser(description='Specify train or inference dataset')
    parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc_raw")
    args = parser.parse_args()
    dataset = args.dataset

#TO-DO change to tensorflow
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

def train_model(config):
    """Trains Model"""
    if check_experiment(config):
        leo, optimizer, train_stats = load_model_and_params(config)
    else:
        leo = LEO(config)
        train_stats = TrainingStats(config)
        episodes_completed = 0
        train_stats = TrainingStats(config)
        tf.keras.backend.clear_session()

    episodes = config.hyperparameters.episodes
    episode_times = []
    train_logger.debug(f"Start time")
    for episode in range(episodes_completed+1, episodes+1):
        start_time = time.time()
        train_stats.set_episode(episode)
        train_stats.set_mode("meta_train")
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        img_transformer = dataloader.transform_image
        mask_transformer = dataloader.transform_mask
        transformers = (img_transformer, mask_transformer) 
        metadata = dataloader.get_batch_data()
        _, train_stats = leo.compute_loss(metadata, train_stats, transformers)

        if episode % config.checkpoint_interval == 0:
            #save model
            pass
    
        if episode % config.meta_val_interval == 0:
            dataloader = Datagenerator(config, dataset, data_type="meta_val")
            train_stats.set_mode("meta_val")
            metadata = dataloader.get_batch_data()
            _, train_stats = leo.compute_loss(metadata, train_stats, transformers, mode="meta_val")
            train_stats.disp_stats()
        episode_time = (time.time() - start_time)/60
        log_msg = print_to_string_io(f"Episode: {episode}, Episode Time: {episode_time:0.03f} minutes", False)
        train_logger.debug(log_msg)
        episode_times.append(episode_time)

    model_and_params = leo, None, train_stats 
    leo = predict_model(config, dataset, model_and_params, transformers)
    log_msg = print_to_string_io(f"Total Model Training Time {np.sum(episode_times):0.03f} minutes", False)
    train_logger.debug(log_msg)
    train_logger.debug(f"End time")
    return leo

def predict_model(config, dataset, model_and_params, transformers):
    """Implement Predicion on Meta-Test"""
    leo, _ , train_stats = model_and_params
    dataloader = Datagenerator(config, dataset, data_type="meta_test")
    train_stats.set_mode("meta_test")
    metadata = dataloader.get_batch_data()
    _, train_stats = leo.compute_loss(metadata, train_stats, transformers, mode="meta_test")
    train_stats.disp_stats()
    experiment = config.experiment
    for mode in ["meta_train", "meta_val", "meta_test"]:
        stats_df = train_stats.get_stats(mode)
        ious_df = train_stats.get_ious(mode)
        stats_df.to_pickle(os.path.join(model_dir, f"experiment_{mode}_{experiment.number}_stats.pkl"))
        ious_df.to_pickle(os.path.join(model_dir, f"experiment_{mode}_{experiment.number}_ious.pkl"))
    log_data("************** Hyperparameters Used ************\n", os.path.join(model_dir, "train_log.txt"))
    msg = print_to_string_io(config.hyperparameters, True)
    log_data(msg, os.path.join(model_dir, "train_log.txt"))
    return leo

def main():
    config = load_config()
    if config.train:
        train_model(config)
    else:
        def evaluate_model():
            dataloader = Datagenerator(config, dataset, data_type="meta_train")
            img_transformer = dataloader.transform_image
            mask_transformer = dataloader.transform_mask
            transformers = (img_transformer, mask_transformer) 
            model_and_params = load_model_and_params(config)
            return predict_model(config, dataset, model_and_params, transformers)
        evaluate_model()
    
if __name__ == "__main__":
    main()