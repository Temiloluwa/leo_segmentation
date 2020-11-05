# REFERENCE: This code is based on the Tensorflow Segmentation official tutorial
# https://www.tensorflow.org/tutorials/images/segmentation
# authors - Temiloluwa Adeoti
# description - Transfer Learning on PASCALVOC 5i dataset
# date - 4-11-2020

import pprint
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from pascal_embeddings_data import DataGenerator
from pascal_models import mobilenet_v2_encoder, Decoder
#from ..utils import load_pickled_data, save_pickled_data, save_npy,
import time
import itertools

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except RuntimeError as e:
    print(e)


def plot_prediction(encoder, decoder, batch_imgs, batch_masks, batch_classnames):
    fig = plt.figure(figsize=(30, 30))
    for i in range(10):
        pred_masks, _ = forward_model(encoder, decoder, np.expand_dims(batch_imgs[i], 0))
        fig.add_subplot(10,3,i*3+1)
        plt.imshow((np.squeeze(batch_imgs[i])))
        plt.title(batch_classnames[i])

        fig.add_subplot(10,3,i*3+2)
        plt.imshow(np.squeeze(batch_masks[i]), cmap="gray")
        plt.title("ground truth")

        fig.add_subplot(10,3,i*3+3)
        plt.imshow(np.squeeze(np.argmax(pred_masks.numpy(),axis=-1)), cmap="gray")
        plt.title("predicted")
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"plots/prediction_epoch-{epoch}.jpg")
    return


def plot_stats(stats, col):
    fig = plt.figure(figsize=(10, 5))
    plt.title(col)
    plt.plot(stats[col])
    plt.xlabel("epochs")
    plt.ylabel(col)
    plt.savefig(f"plots/col-stats-{epoch}.jpg")

            
def forward_model(encoder, decoder, x):
    encoder_output = encoder(x)
    logits, embeddings = decoder(encoder_output)
    return logits, embeddings


def compute_loss(encoder, decoder, x, masks):
    logits, _ = forward_model(encoder, decoder, x)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    scce_loss = scce(masks, logits)
    return scce_loss, logits


def calc_iou_per_class(encoder, decoder, X, y, class_names, ious_per_class):
    for i in range(len(X)):
        x = np.expand_dims(X[i], 0)
        target = np.expand_dims(y[i], 0).astype(int)
        logit, _ = forward_model(encoder, decoder, x)
        target = np.expand_dims(y[i], 0).astype(int)  
        pred = np.argmax(logit.numpy(), axis=-1).astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        ious_per_class[class_names[i]] += iou
    return ious_per_class


@tf.function
def train_step(encoder, decoder, x, masks, optimizer):
    """Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss, _  = compute_loss(encoder, decoder, x, masks)
    gradients = tape.gradient(loss, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify dataset')
    parser.add_argument("--fold", type=int, help='validation fold', default=1)
    parser.add_argument("--imgdims", type=tuple, default=(3, 384, 512))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--generate", type=bool, help='generate embeddings', default=False)
    parser.add_argument("--encoder", type=str, help='choose encoder', default='mobilenet_v2')
    args = parser.parse_args()
    
    FOLD = args.fold
    NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH = args.imgdims
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.bs
    EPOCHS = args.epochs
    FREQ = args.freq
    GENERATE = args.generate
    ENCODER = args.encoder

    pp = pprint.PrettyPrinter(width=100, compact=True)
    tf.keras.backend.clear_session()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    training_stats = []
    train_iou_per_class = defaultdict(int)
    data_generator = DataGenerator(fold=FOLD)
    train_count_per_class = Counter(data_generator.train_class_names)
    val_count_per_class = Counter(data_generator.val_class_names)
    
    encoder_options = {
                        "mobilenet_v2": mobilenet_v2_encoder,
                    }
    chosen_encoder = encoder_options[ENCODER]
    print(f"You have selected Model {ENCODER}")
    encoder = chosen_encoder(args.imgdims)
    decoder = Decoder()

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        val_iou_per_class = defaultdict(int)
        batch_losses = []
        data_generator.setmode("train")
        for i, (batch_imgs, batch_masks, batch_classnames) in enumerate(tqdm(data_generator)):
            batch_loss = train_step(encoder, decoder, batch_imgs, batch_masks, optimizer)
            batch_losses.append(batch_loss.numpy())
        train_loss = float(np.mean(batch_losses))
        batch_losses = []
        data_generator.setmode("val")
        for i, (batch_imgs, batch_masks, batch_classnames) in enumerate(tqdm(data_generator)):
            batch_losses, _ = compute_loss(encoder, decoder, batch_imgs, batch_masks)
            val_iou_per_class = calc_iou_per_class(encoder, decoder, batch_imgs, batch_masks, batch_classnames, val_iou_per_class)
        
        val_loss = float(np.mean(batch_losses))
        val_iou_per_class = {k: v/val_count_per_class[k] for k,v in val_iou_per_class.items()}
        end_time = time.time()
        epoch_time = (end_time - start_time)/60

        training_stats.append({
            "epoch": epoch,
            "train loss": train_loss,
            "val loss": val_loss,
            "epoch time": epoch_time
            })
            
        print(f"Epoch:{epoch}, Train loss:{train_loss:.3f}, Val loss:{val_loss:.3f}, Epoch Time:{epoch_time:.5f} minutes")
        pp.pprint(f"Val ious: {val_iou_per_class}")
        if epoch % FREQ == 1:
            print(f"img shape-{batch_imgs.shape}, masks shape-{batch_masks.shape}, num_images-{len(batch_classnames)}")
            data_generator.setmode("val")
            batch_imgs, batch_masks, batch_classnames  = next(iter(data_generator))
            plot_prediction(encoder, decoder, batch_imgs, batch_masks, batch_classnames)
            plot_stats(pd.DataFrame(training_stats), "val loss")
    
    pp.pprint(f"Train ious: {train_iou_per_class}")
    total_training_time = sum([i["epoch time"] for i in training_stats])
    print(f"Total model training time {total_training_time:.2f} minutes")

    data_generator.setmode("train")
    for i, (batch_imgs, batch_masks, batch_classnames) in enumerate(tqdm(data_generator)):
        train_iou_per_class = calc_iou_per_class(encoder, decoder, batch_imgs, batch_masks, batch_classnames, train_iou_per_class)
        train_iou_per_class = {k: v/train_count_per_class[k] for k,v in train_iou_per_class.items()}
    
    """
    encoder.save_weights("./data/pascal_voc/embedding_ckpt/encoder_weights")
    decoder.save_weights("./data/pascal_voc/embedding_ckpt/decoder_weights")
  
    
    train_stats_save_path_root = os.path.join(os.path.dirname(__file__), "data", "pascal_voc", "emb_train_stats")
    os.makedirs(train_stats_save_path_root, exist_ok=True)

    test_saved_model(encoder, decoder, chosen_encoder, batch_imgs, img_dims)

    save_pickled_data(training_stats, os.path.join(train_stats_save_path_root, f"training_stats_exp_{experiment_number}.pkl"))
    save_pickled_data(iou_per_class_list, os.path.join(train_stats_save_path_root, f"iou_per_class_list_exp_{experiment_number}.pkl"))
    if generate_embeddings:
        save_embeddings(encoder, decoder, "train", **model_kwargs)
        save_embeddings(encoder, decoder,"val", **model_kwargs)
    test_saved_embeddings()
    """
