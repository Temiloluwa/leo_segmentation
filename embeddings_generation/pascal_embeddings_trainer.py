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
from pascal_embeddings_data import DataGenerator, FewShotDataGenerator
from pascal_embeddings_models import mobilenet_v2_encoder_v2, Decoder, Decoderv4
import time
import itertools

pp = pprint.PrettyPrinter(width=100, compact=True)
parser = argparse.ArgumentParser(description='Specify dataset')
parser.add_argument("--fold", type=int, help='validation fold', default=1)
parser.add_argument("--imgdims", type=tuple, default=(3, 384, 512))
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--freq", type=int, default=50)
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
    support_img, query_img, support_mask = x
    support_mask = tf.expand_dims(support_mask, -1)
    support_skips = encoder(support_img, training=False)
    query_skips = encoder(query_img, training=False)
    logits, embeddings = decoder(support_skips, query_skips, support_mask)
    return logits, embeddings


def compute_loss(encoder, decoder, x, masks):
    logits, _ = forward_model(encoder, decoder, x)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    scce_loss = scce(masks, logits)
    return scce_loss, logits


def calc_iou_per_class(encoder, decoder, x, y, class_names, ious_per_class):
    target = y.astype(int)
    logit, _ = forward_model(encoder, decoder, x)
    pred = np.argmax(logit.numpy(), axis=-1).astype(int)
    iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
    ious_per_class[class_names] += iou
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
    gradients, _ = tf.clip_by_global_norm(gradients, 3.0)
    optimizer.apply_gradients(zip(gradients, decoder.trainable_variables))
    return loss


if __name__ == "__main__":    
    tf.keras.backend.clear_session()
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    training_stats = []
    train_data_generator = FewShotDataGenerator(fold=FOLD)
    val_data_generator = FewShotDataGenerator(fold=FOLD, mode="val")
    val_iou_per_class = defaultdict(int)
    
    encoder_options = {
                        "mobilenet_v2": mobilenet_v2_encoder_v2,
                    }
    chosen_encoder = encoder_options[ENCODER]
    print(f"You have selected Model {ENCODER}")
    encoder = chosen_encoder(args.imgdims)
    decoder = Decoderv4(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        batch_losses = []
        for support_imgs, query_imgs, support_masks, query_masks,_, _ in train_data_generator.get_batch_data():
            batch_size = len(support_imgs)
            for i in range(batch_size):
                input_data = (support_imgs[i], query_imgs[i], support_masks[i])
                batch_loss = train_step(encoder, decoder, input_data, query_masks[i], optimizer)
                batch_losses.append(batch_loss.numpy())
        train_loss = float(np.mean(batch_losses))
        val_loss = 0
        val_count_per_class = defaultdict(int)
       
        if epoch % FREQ == 0:
            batch_losses = []
            for support_imgs, query_imgs, support_masks, query_masks, support_fnames, query_fnames in val_data_generator.get_batch_data():
                input_data = (support_imgs, query_imgs, support_masks)
                batch_loss, _ = compute_loss(encoder, decoder, input_data, query_masks)
                batch_losses.append(batch_loss.numpy())
                query_class_names = query_fnames[0].split("-")[0]
                val_iou_per_class = calc_iou_per_class(encoder, decoder, input_data, query_masks, query_class_names, val_iou_per_class)
                val_count_per_class[query_class_names] += 1
                
       
            val_loss = float(np.mean(batch_losses))
            mean_iou = []
            for k,v in val_iou_per_class.items():
                iou = v/val_count_per_class[k]
                val_iou_per_class[k] = iou
                mean_iou.append(iou)
            print("Mean IOU ", np.mean(mean_iou))
            pp.pprint(f"Val ious: {val_iou_per_class}")
        
        

        end_time = time.time()
        epoch_time = (end_time - start_time)/60

        training_stats.append({
            "epoch": epoch,
            "train loss": train_loss,
            "val loss": val_loss,
            "epoch time": epoch_time
            })
           
        print(f"Epoch:{epoch}, Train loss:{train_loss:.3f}, Val loss:{val_loss:.3f}, Epoch Time:{epoch_time:.5f} minutes")
        
        """
        if epoch % FREQ == 1:
            print(f"img shape-{batch_imgs.shape}, masks shape-{batch_masks.shape}, num_images-{len(batch_classnames)}")
            batch_imgs, batch_masks, batch_classnames  = next(iter(data_generator))
            plot_prediction(encoder, decoder, batch_imgs, batch_masks, batch_classnames)
            plot_stats(pd.DataFrame(training_stats), "train loss")
            plot_stats(pd.DataFrame(training_stats), "val loss")]
        """
    
    total_training_time = sum([i["epoch time"] for i in training_stats])
    print(f"Total model training time {total_training_time/60:.2f} hours")

    
    
