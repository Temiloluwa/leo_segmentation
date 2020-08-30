from IPython import get_ipython
# ##### REFERENCE: This code is based on the Tensorflow Segmentation official tutorial
# ##### Copyright 2019 The TensorFlow Authors.
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

from collections import OrderedDict, Counter
from tqdm import tqdm
from PIL import Image
import time

grouped_by_classes_root = os.path.join(os.path.dirname(__file__), "data", "grouped_by_classes")
former_data_root = os.path.join(os.path.dirname(__file__), "data", "original_pascalvoc5i", "pascal-5")

train_classes = os.listdir(os.path.join(grouped_by_classes_root, "train", "images" ))
val_classes = os.listdir(os.path.join(grouped_by_classes_root,"val", "images"))
train_classes = sorted([i.lower() for i in train_classes])
val_classes = sorted([i.lower() for i in val_classes])
print("train_classes", train_classes)
print("val_classes", val_classes)

def load_pickled_data(data_path):
    """Reads a pickle file"""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

def save_pickled_data(data, data_path):
    """Saves a pickle file"""
    with open(data_path, "wb") as f:
        data = pickle.dump(data,f)
    return data


#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class Transform_image(object):

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im):
        w, h = im.size
        if h > w:
            im = im.transpose(method=Image.ROTATE_270).resize((img_width, img_height))
        else:
            im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im - 127.5)/127.5
        return im
    

class Transform_mask(object):

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im):
        w, h = im.size
        if h > w:
            im = im.transpose(method=Image.ROTATE_270).resize((img_width, img_height))
        else:
            im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im - 127.5)/127.5
        im = np.round(rgb2gray((im) > 0).astype(np.float32))
        return im

num_classes = len(val_classes)
img_paths = {}
masks_paths = {}
for i in range(num_classes):
    if i not in img_paths:
        img_paths[val_classes[i]] = os.path.join(grouped_by_classes_root, "val", "images" ,val_classes[i])
    
    if i not in masks_paths: 
        masks_paths[val_classes[i]] = os.path.join(grouped_by_classes_root, "val", "masks", val_classes[i])

class SampleOneClass(Dataset):
    """
    CustomData dataset
    """
    def __init__(self, class_name):
        super(SampleOneClass, self).__init__()
        self.class_name = class_name
        self.img_datasets = img_val_datasets
        self.mask_datasets = mask_val_datasets
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.img_paths = [i[0] for i in self.img_datasets.imgs if class_name in i[0]]
        self.mask_paths = [i[0] for i in self.mask_datasets.imgs if class_name in i[0]]
        self.class_counts = Counter([i[0].split(os.sep)[-2] for i in self.img_datasets.imgs])
        
    def __getitem__(self, idx):
        img_paths =  self.img_paths[idx]
        mask_paths = self.mask_paths[idx]
        img_paths  = [img_paths] if type(img_paths) == str else img_paths
        mask_paths  = [mask_paths] if type(mask_paths) == str else mask_paths           
        img_p_class = np.array([self.transform_image(Image.open(i)) for i in img_paths])
        msk_p_class = np.array([self.transform_mask(Image.open(i)) for i in mask_paths])

        return img_p_class, msk_p_class

    def __len__(self):
        return self.class_counts[self.class_name]


num_channels, img_height, img_width =  3, 384,512
transform_image = Transform_image(img_width, img_height)
transform_mask = Transform_mask(img_width, img_height)

img_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "train", "images"), transform=transform_image)
mask_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "train", "masks"), transform=transform_mask)

img_val_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "val", "images"), transform=transform_image)
mask_val_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "val", "masks"), transform=transform_mask)
    
total_num_imgs = len(img_datasets.imgs)
img_filenames = ["_".join(i[0].split("/")[-2:]) for i in img_datasets.imgs]
mask_filenames = ["_".join(i[0].split("/")[-2:]) for i in mask_datasets.imgs]


base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",  
    input_shape=(img_height, img_width, num_channels), 
    include_top=False,
)  

layer_names = [
    'block_1_expand_relu',   # 188, 250, 96
    'block_3_expand_relu',   # 94, 125, 144
    'block_6_expand_relu',   # 47, 63, 192
    'block_13_expand_relu',  # 24, 32, 576
    'block_16_project',      # 12, 16, 320
]

layers = [base_model.get_layer(name).output for name in layer_names]
# Freeze the base_model
encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
encoder.trainable = False

inputs = tf.keras.Input(shape=((img_height, img_width, num_channels)))
# Downsampling through the model
skips = encoder(inputs, training=False)

conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv1b = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv2 = tf.keras.layers.Conv2D(filters=8*2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv2b = tf.keras.layers.Conv2D(filters=8*2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv3 = tf.keras.layers.Conv2D(filters=8*3, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv3b = tf.keras.layers.Conv2D(filters=8*3, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv4 = tf.keras.layers.Conv2D(filters=8*4, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv4b = tf.keras.layers.Conv2D(filters=8*4, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv5 = tf.keras.layers.Conv2D(filters=8*5, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
conv5b = tf.keras.layers.Conv2D(filters=8*5, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
convfinal = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
upsample1 = tf.keras.layers.Conv2DTranspose(8, 3, strides=2,padding='same')
upsample2 = tf.keras.layers.Conv2DTranspose(8*2, 3, strides=2,padding='same')
upsample3 = tf.keras.layers.Conv2DTranspose(8*3, 3, strides=2,padding='same')
upsample4 = tf.keras.layers.Conv2DTranspose(8*4, 3, strides=2,padding='same')
upsample5 = tf.keras.layers.Conv2DTranspose(8*5, 3, strides=2,padding='same')
concat = tf.keras.layers.Concatenate()
encoder_output = skips[-1]

#comments on input size are wrong
print(encoder_output.shape)
x = conv1(encoder_output)
x = tf.keras.layers.Dropout(0.5)(x)
x = conv1b(x)
x = upsample1(x)
x = concat([x, skips[-2]])
x = conv2(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = conv2b(x)
x = upsample2(x)
x = concat([x, skips[-3]])
x = conv3(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = conv3b(x)
x = upsample3(x)
x = concat([x, skips[-4]])
x = conv4(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = conv4b(x)
x = upsample4(x)
x = concat([x, skips[-5]])
out4 = conv5(x)
x = tf.keras.layers.Dropout(0.5)(out4)
x = conv5b(x)
x = upsample5(x)
output = convfinal(x)

model = tf.keras.Model(inputs=inputs, outputs=[output, out4])
model.summary()
optimizer = tf.keras.optimizers.Adam(1e-4)

def compute_loss(model, x, masks):
  logits = model(x)[0]
  scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  scce_loss = scce(masks, logits)
  return scce_loss, logits

def calc_val_loss_and_iou_per_class(model, epoch, freq):
  class_ious = {}
  val_loss = []
  for class_ in val_classes:
    class_one = SampleOneClass(class_)
    iou_per_class = []
    loss_per_class = []
    for j in range(len(class_one)):
        inp_img, target = class_one[j]
        loss, logits = compute_loss(model, inp_img, target)
        pred = np.argmax(logits.numpy(),axis=-1).astype(int)
        target = target.astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        iou_per_class.append(iou)
        loss_per_class.append(loss)
    mean_iou_per_class = np.mean(iou_per_class)
    mean_loss_per_class = np.mean(loss_per_class)
    val_loss.append(mean_loss_per_class)
    
    if epoch % freq == 1:
        print(f"Mean IOU for class {class_} is {mean_iou_per_class}")
    class_ious[f"{class_}"] = mean_iou_per_class
  val_loss = np.mean(val_loss)
  return class_ious, val_loss

def plot_prediction(model, input_data, masks, filenames):  
  fig = plt.figure(figsize=(30, 30))
  ground_truth_masks = masks
  for i in range(10):
    pred_masks = model(input_data[i].reshape(1,img_height,img_width,3))[0]
    fig.add_subplot(10,3,i*3+1)
    plt.imshow((input_data[i]*127.5+127.5).astype("uint8"))
    plt.title(filenames[idx])

    fig.add_subplot(10,3,i*3+2)
    plt.imshow(ground_truth_masks[i].reshape(img_height, img_width), cmap="gray")
    plt.title("ground truth")

    fig.add_subplot(10,3,i*3+3)
    plt.imshow(np.argmax(pred_masks.numpy(),axis=-1)[0].reshape(img_height, img_width), cmap="gray")
    plt.title("predicted")
  plt.subplots_adjust(hspace=0.5)
  plt.show()
  return

def plot_stats(stats, col):
  fig = plt.figure(figsize=(10, 5))
  plt.plot(stats[col])
  plt.xlabel("epochs")
  plt.ylabel(col)
  plt.show()

@tf.function
def train_step(model, x, masks, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss, _  = compute_loss(model, x, masks)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss


tf.keras.backend.clear_session()
epochs = 30
freq = 5
bs = 10
training_stats = []
iou_per_class_list = []

for epoch in range(1, epochs + 1):
    start_time = time.time()
    dataloader_img = iter(DataLoader(img_datasets, batch_size=bs,shuffle=False, num_workers=0))
    dataloader_mask = iter(DataLoader(mask_datasets, batch_size=bs,shuffle=False, num_workers=0))
    batch_losses = []
       
    for j in range(total_num_imgs//bs+1):
        batch_imgs = next(dataloader_img)[0].numpy()
        batch_masks = next(dataloader_mask)[0].numpy()
        batch_loss = train_step(model, batch_imgs, batch_masks, optimizer)
        batch_losses.append(batch_loss.numpy())
   
    train_loss = float(np.mean(batch_losses))
    iou_per_class, val_loss = calc_val_loss_and_iou_per_class(model, epoch, freq)
    iou_per_class_list.append(iou_per_class)
    end_time = time.time()
    epoch_time = (end_time - start_time)/60

    training_stats.append({
      "epoch":epoch,
      "train loss":train_loss,
      "val loss":val_loss,
      "epoch time": epoch_time
    })
    print(f"Epoch:{epoch}, Train loss:{train_loss}, Val loss:{val_loss},Epoch Time:{epoch_time}")
    #if epoch % freq == 0:
        #plot_prediction(model, batch_imgs, batch_masks, img_filenames)
        #plot_stats(pd.DataFrame(training_stats), "train loss")
        #
save_pickled_data(training_stats, "training_stats.pkl")
save_pickled_data(iou_per_class_list, "iou_per_class_list.pkl")

