import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
import time


grouped_by_classes_root = os.path.join(os.getcwd(), "data", "grouped_by_classes")
former_data_root = os.path.join(os.getcwd(),"data", "original_pascalvoc5i", "pascal-5")

classes = os.listdir(os.path.join(grouped_by_classes_root, "images"))
classes = [i.lower() for i in classes]
classes = sorted(classes)

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

def save_npy(np_array, filename):
    with open(f"{filename[:-4]}.npy", "wb") as f:
        return np.save(f, np_array)
    
def load_npy(filename):
    with open(f"{filename[:-4]}.npy", "rb") as f:
        return np.load(f)

def shuffle(X_train, X_val, y_train, y_val, fn_train, fn_val):
    train_indices = np.arange(len(X_train))
    val_indices = np.arange(len(X_val))
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    return X_train[train_indices], X_val[val_indices], \
            y_train[train_indices], y_val[val_indices], \
            fn_train[train_indices], fn_val[val_indices]

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


img_height, img_width =  384,512

transform_image = Transform_image(img_width, img_height)
transform_mask = Transform_mask(img_width, img_height)

img_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "images"), transform=transform_image)
mask_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "masks"), transform=transform_mask)
    
total_num_imgs = len(img_datasets.imgs)
img_filenames = ["_".join(i[0].split("/")[-2:]) for i in img_datasets.imgs]
mask_filenames = ["_".join(i[0].split("/")[-2:]) for i in mask_datasets.imgs]


num_samples_train, num_channels = total_num_imgs, 3
num_samples_val  = 0
num_samples = num_samples_val + num_samples_train

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(img_height, img_width, num_channels), #375,500
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

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
convsemifinal = tf.keras.layers.Conv2D(filters=14, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
convfinal = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', use_bias=False)
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
x = tf.keras.layers.Dropout(0.25)(x)
x = conv1b(x)
x = upsample1(x)
x = concat([x, skips[-2]])
out1 = conv2(x)
x = tf.keras.layers.Dropout(0.25)(out1)
x = conv2b(x)
x = upsample2(x)
x = concat([x, skips[-3]])
out2 = conv3(x)
x = tf.keras.layers.Dropout(0.25)(out2)
x = conv3b(x)
x = upsample3(x)
x = concat([x, skips[-4]])
out3 = conv4(x)
x = tf.keras.layers.Dropout(0.25)(out3)
x = conv4b(x)
x = upsample4(x)
x = concat([x, skips[-5]])
x = conv5(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = conv5b(x)
x = upsample5(x)
out4 = convsemifinal(x)
output = convfinal(out4)
#x = tf.keras.layers.Activation('sigmoid')(output)


class SampleOneClass(Dataset):
    """
    CustomData dataset
    """
    def __init__(self):
        super(SampleOneClass, self).__init__()
        self.img_datasets = img_datasets
        self.mask_datasets = mask_datasets
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __getitem__(self, class_name):
        img_paths = [i[0] for i in self.img_datasets.imgs if class_name in i[0]]
        mask_paths = [i[0] for i in self.mask_datasets.imgs if class_name in i[0]]
        img_p_class = np.array([self.transform_image(Image.open(i)) for i in img_paths])
        msk_p_class = np.array([self.transform_mask(Image.open(i)) for i in mask_paths])
        if not hasattr(self, "filenames"):
            self.filenames = [ i[0].split(os.sep)[-1] for i in self.img_datasets.imgs if class_name in i[0]]

        return img_p_class,msk_p_class

    def get_filenames(self):
        return self.filenames

    def __len__(self):
        return len(img_paths)



def compute_loss(model, x, masks):
  output = model(x)
  logits = output[0]
  scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  scce_loss = scce(masks, logits)
  return scce_loss

def calc_iou_per_class(model, X, y, data_type):
  bs = 10
  iou_per_class = []
  for j in range(len(X)//bs+1):
    output = model(X[bs*j:(j+1)*bs])
    logits = output[0]
    targets = y[bs*j:(j+1)*bs]
    for i in range(len(logits)):
      pred = np.argmax(logits[i].numpy(),axis=-1).astype(int)
      target = targets[i].astype(int)
      iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
      iou_per_class.append(iou)
  mean_iou_per_class = np.mean(iou_per_class)
  if epochs % freq == 0:
    print(f"{data_type} Mean IOU for class {selected_class} is {mean_iou_per_class}")
  return mean_iou_per_class


def plot_prediction(model, input_data, masks):  
  fig = plt.figure(figsize=(30, 30))
  num_plots = len(input_data)//2
  ground_truth_masks = masks
  for i in range(num_plots):
    output = model(input_data[i].reshape(1,img_height,img_width,3))
    pred_masks = output[0]
    fig.add_subplot(num_plots,3,i*3+1)
    plt.imshow((input_data[i]*127.5+127.5).astype("uint8"))

    fig.add_subplot(num_plots,3,i*3+2)
    plt.imshow(ground_truth_masks[i].reshape(img_height, img_width), cmap="gray")
    plt.title("ground truth")

    fig.add_subplot(num_plots,3,i*3+3)
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
def _train_step(model, x, masks, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
 
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x, masks)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

class Compute_loss():
    def __init__(self):
      pass

    @tf.function
    def __call__(self, model, x, masks, optimizer):
      with tf.GradientTape() as tape:
        loss = compute_loss(model, x, masks)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return loss

for selected_class in classes:
  tf.keras.backend.clear_session()
  train_step = Compute_loss()
  model = tf.keras.Model(inputs=inputs, outputs=[output, out4])
  model.summary()
  optimizer = tf.keras.optimizers.Adam(1e-3)
  tr_val_split = 0.3

  one_class = SampleOneClass()
  imgs_in_class, msks_in_class = one_class[selected_class]
  fn = np.array(one_class.get_filenames())
  X_train, X_val, y_train, y_val, fn_train, fn_val = train_test_split(
    imgs_in_class, msks_in_class,fn, test_size=tr_val_split)

  epochs = 30
  freq = 5
  bs = 10
  num_batches = num_samples//bs
  training_stats = []
  iou_train = []
  iou_val = []

  for epoch in range(1, epochs + 1):
      start_time = time.time()
      X_train, X_val, y_train, y_val, fn_train, fn_val = \
        shuffle(X_train, X_val, y_train, y_val, fn_train, fn_val)
      
      assert(len(X_train) == len(y_train) == len(fn_train))
      assert(len(X_val) == len(y_val) == len(fn_val))

      batch_losses = []
      batch_val_losses = []
        
      for j in range(len(X_train)//bs+1):
          batch_imgs = X_train[j*bs:(j+1)*bs]
          batch_masks = y_train[j*bs:(j+1)*bs]
          batch_loss = train_step(model, batch_imgs, batch_masks, optimizer)
          batch_losses.append(batch_loss.numpy())
      
      train_loss = float(np.mean(batch_losses))
      for i in range(len(X_val)//bs + 1):
        batch_imgs_val = X_val[i*bs:(i+1)*bs]
        batch_masks_val = y_val[i*bs:(i+1)*bs]
        batch_val_loss = compute_loss(model, batch_imgs_val, batch_masks_val)
        batch_val_losses.append(batch_val_loss.numpy())
      
      val_loss = float(np.mean(batch_val_losses))
      iou_tr = calc_iou_per_class(model, X_train, y_train,"Train")
      iou_v = calc_iou_per_class(model, X_val, y_val, "Val")
      iou_train.append(iou_tr)
      iou_val.append(iou_v)
      end_time = time.time()
      epoch_time = end_time - start_time

      training_stats.append({
        "epoch":epoch,
        "train loss":train_loss,
        "val loss":val_loss,
        "epoch time": epoch_time,
        "iou_tr":iou_tr,
        "iou_v":iou_v
      })
      print(f"Class{selected_class} Epoch:{epoch}, Train loss:{train_loss}, Val loss:{val_loss},iou tr:{iou_tr}, iou_v:{iou_v}, Epoch Time:{epoch_time}")

  path_root = os.path.join(os.getcwd(), "data", "pascal_voc")
 
  def save_data(X, y, fn, data_type):
    images_save_path_data_type_root = os.path.join(path_root, f"{data_type}", "images", selected_class)
    masks_save_path_data_type_root = os.path.join(path_root, f"{data_type}", "masks", selected_class)
    for i in range(len(X)):
      x_ = np.expand_dims(X[i], 0)
      y_ = np.expand_dims(y[i], 0)
      fn_ = fn[i]
 
      if not os.path.exists(images_save_path_data_type_root):
        os.makedirs(images_save_path_data_type_root, exist_ok=True)
        os.makedirs(masks_save_path_data_type_root, exist_ok=True)

      img_file_path = os.path.join(images_save_path_data_type_root, fn_)
      mask_file_path = os.path.join(masks_save_path_data_type_root, fn_)
      selected_embedding = 1
      output = model(x_)[selected_embedding].numpy()
      save_npy(output, img_file_path)
      save_npy(y_,  mask_file_path)

  save_data(X_train, y_train, fn_train, "train")
  save_data(X_val, y_val, fn_val, "val")    