from data import Datagenerator, TrainingStats
from model import LEO, load_model, save_model
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from utils import load_config, check_experiment, get_named_dict, display_data_shape
import numpy as np
import tensorflow as tf
import os
import argparse
import torch 
import torch.optim
import gc

#parser = argparse.ArgumentParser(description='Specify train or inference dataset')
#parser.add_argument("-d", "--dataset", type=str, nargs=1, default="pascal_voc")
#args = parser.parse_args()
#dataset = args.dataset

dataset = "pascal_voc_raw"
@tf.function
def compute_loss(leo, metadata, train_stats, optimizer, loss_fn, mode="meta_train"):
    """
    Computes the  outer loop loss

    Args:
        model (object) : leo model
        meta_dataloader (object): Dataloader 
        train_stats: (object): train stats object
        config (dict): config
        mode (str): meta_train, meta_val or meta_test
    Returns:
        (tuple) total_val_loss (list), train_stats
    """
    num_tasks = len(metadata[0])
    if train_stats.episode % 5 == 1:
        display_data_shape(metadata)
    total_val_loss = []

    for batch in range(num_tasks):
        data_dict = get_named_dict(metadata, batch)
        #weights = self.seg_weight.clone()
        #bias = self.seg_bias.clone()
        #tr_loss, tr_decoder_output = self.leo_inner_loop(data_dict.tr_imgs, weights, bias, data_dict.tr_masks)
        with tf.GradientTape() as tape:
            o = leo.encoder(data_dict.tr_imgs)
            pred = leo.decoder(o)
            tr_loss =  loss_fn(data_dict.tr_masks, pred)
            #val_loss = self.finetuning_inner_loop(data_dict, tr_loss, tr_decoder_output, weights, bias)
            total_val_loss.append(tr_loss)
    
    total_val_loss = sum(total_val_loss)/len(total_val_loss)
    gradients = tape.gradient(total_val_loss, leo.decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, leo.decoder.trainable_variables))

    stats_data = {
        "mode": mode,
        "kl_loss": 0,
        "total_val_loss":total_val_loss
    }
    train_stats.update_stats(**stats_data)
    return total_val_loss, train_stats


def train_model(config):
    """Trains Model"""
    writer = SummaryWriter(os.path.join(config.data_path, "models", str(config.experiment.number)))
    """
    if check_experiment(config):
        leo, optimizer, stats = load_model(config)
        episodes_completed = stats["episode"]
        leo.eval()
        leo = leo.to(device)
        train_stats = TrainingStats(config)
        train_stats.set_episode(episodes_completed)
        train_stats.update_stats(**stats)
    else:
    """
    leo = LEO(config)
    train_stats = TrainingStats(config)
    tf.keras.backend.clear_session()
    episodes_completed = 0
    episodes = 10
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for episode in range(episodes_completed+1, episodes+1):
        train_stats.set_episode(episode)
        dataloader = Datagenerator(config, dataset, data_type="meta_train")
        metadata = dataloader.get_batch_data()
        metatrain_loss, train_stats = compute_loss(leo, metadata, train_stats, optimizer, loss_fn)
        print(f"Episode {episode}, Tr loss {metatrain_loss}")
        leo.evaluate(metadata)

        #if episode % config.checkpoint_interval == 0:
        #    save_model(leo, optimizer, config, edict(train_stats.get_latest_stats()))
            #writer.add_graph(leo, metadata[:-1])
            #writer.close()
          
        #leo.evaluate_val_imgs(metadata, class_in_metadata, train_stats, writer)
        #train_stats.disp_stats()

        #if episode == episodes:
        #    return leo, metadata, class_in_metadata
        #del metadata
        #gc.collect()
        #torch.cuda.ipc_collect()
        #torch.cuda.empty_cache()
        
        #meta-val
        #dataloader = Datagenerator(config, dataset, data_type="meta_val", generate_new_metaclasses=False)
        #_, train_stats = compute_loss(leo, dataloader, train_stats, config, mode="meta_val")
        #train_stats.disp_stats()

        #dataloader = Datagenerator(config, dataset, data_type="meta_test", generate_new_metaclasses=False)
        #_, train_stats = compute_loss(leo, dataloader ,train_stats, config, mode="meta_test")
        #train_stats.disp_stats()
        

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