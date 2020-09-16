# REFERENCE: This code is based on the Tensorflow Segmentation official tutorial
# https://www.tensorflow.org/tutorials/images/segmentation
# authors - Temiloluwa Adeoti
# description - Transfer Learning on PASCALVOC 5i dataset
# date - 30-08-2020

import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import numpy as np
from collections import OrderedDict, Counter
from tqdm import tqdm
from PIL import Image
from pascal_models import mobilenet_v2_encoder, Decoder
from utils import load_pickled_data, save_pickled_data, save_npy, rgb2gray
import time
import itertools

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except RuntimeError as e:
    print(e)

def test_saved_model(encoder, decoder, chosen_encoder, input_imgs, img_dims):
    oracle = forward_model(encoder, decoder, input_imgs)
    retrieved_encoder = chosen_encoder(img_dims)
    retrieved_encoder.load_weights("./data/embedding_ckpt/encoder_weights")
    retrieved_decoder = Decoder()
    retrieved_decoder.load_weights("./data//embedding_ckpt/decoder_weights")
    retrieved_outputs = forward_model(retrieved_encoder, retrieved_decoder, input_imgs)
    test_types = ["final output", "embeddings"]
    for _test, _oracle, _retrieved_output in zip(test_types, oracle, retrieved_outputs):
        print(f"{_test} testing, oracle shape: {_oracle.shape}, retr. shape: {_retrieved_output.shape}")
        try:
            np.testing.assert_allclose(_retrieved_output, _oracle, rtol=0.7, atol=1e-5)
            print("Reconstruction okay")
            diff = _retrieved_output - _oracle 
            print(f"max absolute diff: {np.max(np.absolute(diff))}, max relative diff: {np.nanmax(diff/_oracle)}")
        except AssertionError as e:
                print(f"Error occured in reconstructions\n{e}")

def test_saved_embeddings():
    emb_path_root = os.path.join(os.path.dirname(__file__), "data", "pascal_voc")
    data_path_root = os.path.join(os.path.dirname(__file__), "data", "grouped_by_classes")
    data_types = ["train", "val"]
    data_categories = ["images", "masks"]
    
    for data_type, data_category in itertools.product(data_types, data_categories):
        data_type_root = os.path.join(data_path_root, f"{data_type}", f"{data_category}")
        emb_type_root = os.path.join(emb_path_root, f"{data_type}", f"{data_category}")
        
        for selected_class in os.listdir(data_type_root):
            data_class_root = os.path.join(data_type_root, selected_class)
            emb_class_root = os.path.join(emb_type_root, selected_class)
            data = [i[:-4] for i in os.listdir(data_class_root)]
            emb = [i[:-4] for i in os.listdir(emb_class_root)]
            assert set(data) - set(emb) == set(), "not all files generated"
            print(f"class: {selected_class}, data_type: {data_type}, data_category: {data_category} is ok")

def save_embeddings(encoder, decoder, data_type, **kwargs):
    path_root = os.path.join(os.path.dirname(__file__), "data", "pascal_voc")
    for selected_class in kwargs[f"{data_type}_classes"]:
        images_save_path_data_type_root = os.path.join(path_root, f"{data_type}", "images", selected_class)
        masks_save_path_data_type_root = os.path.join(path_root, f"{data_type}", "masks", selected_class)
        class_one = SampleOneClass(selected_class, data_type, **kwargs)
        for j in range(len(class_one)):
            inp_img, target = class_one[j]
            target = np.squeeze(target)
            fn = class_one.file_names[j]
            output_embedding = np.squeeze(forward_model(encoder, decoder, inp_img)[1].numpy())
        
            if not os.path.exists(images_save_path_data_type_root):
                os.makedirs(images_save_path_data_type_root, exist_ok=True)
                os.makedirs(masks_save_path_data_type_root, exist_ok=True)

            img_file_path = os.path.join(images_save_path_data_type_root, fn)
            mask_file_path = os.path.join(masks_save_path_data_type_root, fn)
            save_npy(output_embedding, img_file_path)
            save_npy(target,  mask_file_path)

class Transform_image(object):

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im):
        w, h = im.size
        if h > w:
            im = im.transpose(method=Image.ROTATE_270).resize((self.img_width, self.img_height))
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
            im = im.transpose(method=Image.ROTATE_270).resize((self.img_width, self.img_height))
        else:
            im = im.resize((self.img_width, self.img_height))
        im = np.array(im)
        im = im.astype(np.float32)
        im = (im - 127.5)/127.5
        im = np.round(rgb2gray((im) > 0).astype(np.float32))
        return im

class SampleOneClass(Dataset):
    """
    CustomData dataset
    """
    def __init__(self, class_name, train_or_val,**kwargs):
        super(SampleOneClass, self).__init__()
        self.class_name = class_name
        self.img_datasets = datasets.ImageFolder(root=os.path.join(kwargs["grouped_by_classes_root"],\
             f"{train_or_val}", "images"), transform=kwargs["transform_image"])
        self.mask_datasets =  datasets.ImageFolder(root=os.path.join(kwargs["grouped_by_classes_root"], \
            f"{train_or_val}", "masks"), transform=kwargs["transform_mask"])
        self.transform_image = kwargs["transform_image"]
        self.transform_mask = kwargs["transform_mask"]
        self.img_paths = [i[0] for i in self.img_datasets.imgs if class_name in i[0]]
        self.mask_paths = [i[0] for i in self.mask_datasets.imgs if class_name in i[0]]
        self.file_names = [i.split(os.sep)[-1] for i in self.img_paths]
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

def forward_model(encoder, decoder, x):
    encoder_output = encoder(x)
    logits, embeddings = decoder(encoder_output)
    return logits, embeddings

def compute_loss(encoder, decoder, x, masks):
    logits, _ = forward_model(encoder, decoder, x)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    scce_loss = scce(masks, logits)
    return scce_loss, logits

def calc_val_loss_and_iou_per_class(encoder, decoder, epoch, freq, **kwargs):
  class_ious = {}
  val_loss = []
  bs = kwargs["bs"]
  for class_ in kwargs["val_classes"]:
    class_one = SampleOneClass(class_, "val", **kwargs)
    iou_per_class = []
    loss_per_class = []
    for j in range(len(class_one)//bs ):
        inp_img, target = class_one[j*bs:(j+1)*bs]
        loss, logits = compute_loss(encoder, decoder, inp_img, target)
        pred = np.argmax(logits.numpy(),axis=-1).astype(int)
        target = target.astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        iou_per_class.append(iou)
        loss_per_class.append(loss)
    mean_iou_per_class = np.mean(iou_per_class)
    mean_loss_per_class = np.mean(loss_per_class)
    val_loss.append(mean_loss_per_class)
    
    if epoch % freq == 0:
        print(f"Mean IOU for class {class_} is {mean_iou_per_class:.5f}")
    class_ious[f"{class_}"] = mean_iou_per_class
  val_loss = np.mean(val_loss)
  return class_ious, val_loss

def plot_prediction(encoder, decoder, kwargs):
    selected_class =  np.random.choice(kwargs["val_classes"])
    print(f"Showing predictions for class {selected_class}")
    class_one = SampleOneClass(selected_class, "val", **kwargs)
    random_indices = np.random.choice(len(class_one), 10) 
    fig = plt.figure(figsize=(30, 30))

    for i,j in enumerate(random_indices):
        input_data, ground_truth_masks = class_one[j]
        pred_masks, _ = forward_model(encoder, decoder, input_data)

        fig.add_subplot(10,3,i*3+1)
        plt.imshow((np.squeeze(input_data)*127.5 + 127.5).astype("uint8"))
        plt.title(class_one.file_names[j])

        fig.add_subplot(10,3,i*3+2)
        plt.imshow(np.squeeze(ground_truth_masks) , cmap="gray")
        plt.title("ground truth")

        fig.add_subplot(10,3,i*3+3)
        plt.imshow(np.squeeze(np.argmax(pred_masks.numpy(),axis=-1)), cmap="gray")
        plt.title("predicted")
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return

def plot_stats(stats, col):
    fig = plt.figure(figsize=(10, 5))
    plt.title(col)
    plt.plot(stats[col])
    plt.xlabel("epochs")
    plt.ylabel(col)
    plt.show()

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

def train_model(encoder, decoder, epochs, freq, show_stats=False,**model_kwargs):
    bs, grouped_by_classes_root, _ = model_kwargs["bs"], model_kwargs["grouped_by_classes_root"], model_kwargs["train_classes"]
    _ , transform_image, transform_mask = model_kwargs["val_classes"], model_kwargs["transform_image"], model_kwargs["transform_mask"]
    
    img_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "train", "images"), transform=transform_image)
    mask_datasets = datasets.ImageFolder(root=os.path.join(grouped_by_classes_root, "train", "masks"), transform=transform_mask)
    total_num_train_imgs = len(img_datasets.imgs)
    tf.keras.backend.clear_session()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    training_stats = []
    iou_per_class_list = []
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        dataloader_img = iter(DataLoader(img_datasets, batch_size=bs,shuffle=False, num_workers=0))
        dataloader_mask = iter(DataLoader(mask_datasets, batch_size=bs,shuffle=False, num_workers=0))
        batch_losses = []
        
        for j in range(total_num_train_imgs//bs):
            batch_imgs = next(dataloader_img)[0].numpy()
            batch_masks = next(dataloader_mask)[0].numpy()
            batch_loss = train_step(encoder, decoder, batch_imgs, batch_masks, optimizer)
            batch_losses.append(batch_loss.numpy())
    
        train_loss = float(np.mean(batch_losses))
        iou_per_class, val_loss = calc_val_loss_and_iou_per_class(encoder, decoder, epoch, freq, **model_kwargs)
        iou_per_class_list.append(iou_per_class)
        end_time = time.time()
        epoch_time = (end_time - start_time)/60

        training_stats.append({
        "epoch":epoch,
        "train loss":train_loss,
        "val loss":val_loss,
        "epoch time": epoch_time
        })
        print(f"Epoch:{epoch}, Train loss:{train_loss:.5f}, Val loss:{val_loss:.5f}, Epoch Time:{epoch_time:.5f} minutes")


        if epoch % freq == 0 and show_stats:
            plot_prediction(encoder, decoder, model_kwargs)
            plot_stats(pd.DataFrame(training_stats), "val loss")
    
    total_training_time = sum([i["epoch time"] for i in training_stats])
    print(f"Total model training time {total_training_time:.2f} minutes")
    encoder.save_weights("./data/embedding_ckpt/encoder_weights")
    decoder.save_weights("./data/embedding_ckpt/decoder_weights")
    return training_stats, iou_per_class_list, decoder, batch_imgs

def main(**train_kwargs):
    """
    Main function

    Args:
        train_kwargs(dict): supply the following key word arguments else defaults would be used
                            defaults are shown in brackets
                            num_channels, img_height, img_width (tuple) - (3, 384, 512)
                            epochs(int) - 30, freq(int) - 5, bs(int) - 10, experiment_number(int) - 0,
                            model(str) - mobilenet_v2
    Returns:
        training_stats (list): list of dictionaries containing the training statistics
        iou_per_class_list (list): list of dictionaries containing validation ious per class
        model (keras model)
    """
   
    num_channels, img_height, img_width =  train_kwargs.get("image_shape", (3, 384, 512))
    epochs, freq, bs = train_kwargs.get("epochs", 30), train_kwargs.get("freq", 1), train_kwargs.get("bs", 1)
    experiment_number = train_kwargs.get("experiment_number", 2)
    generate_embeddings = train_kwargs.get("generate_embeddings", True)
    choose_encoder = train_kwargs.get("encoder", "mobilenet_v2")

    grouped_by_classes_root = os.path.join(os.path.dirname(__file__), "data", "grouped_by_classes")
    train_classes = os.listdir(os.path.join(grouped_by_classes_root, "train", "images" ))
    val_classes = os.listdir(os.path.join(grouped_by_classes_root,"val", "images"))
    train_classes = sorted([i.lower() for i in train_classes])
    val_classes = sorted([i.lower() for i in val_classes])
    print("train_classes", train_classes)
    print("val_classes", val_classes)

    transform_image = Transform_image(img_width, img_height)
    transform_mask = Transform_mask(img_width, img_height)
    
    #update dict with_encoder options
    encoder_options = {"mobilenet_v2": mobilenet_v2_encoder,}

    chosen_encoder = encoder_options[choose_encoder]
    print(f"You have selected Model {choose_encoder}")
    img_dims = num_channels, img_height, img_width
    encoder = chosen_encoder(img_dims)
    decoder = Decoder()

    model_kwargs = {"bs":bs, "grouped_by_classes_root":grouped_by_classes_root, "train_classes":train_classes, \
                    "val_classes":val_classes, "transform_image":transform_image, "transform_mask":transform_mask}
    
    training_stats, iou_per_class_list, decoder, batch_imgs = train_model(encoder, decoder, epochs, freq, show_stats=False, **model_kwargs)

    train_stats_save_path_root = os.path.join(os.path.dirname(__file__), "data", "pascal_voc", "emb_train_stats")
    os.makedirs(train_stats_save_path_root, exist_ok=True)

    test_saved_model(encoder, decoder, chosen_encoder, batch_imgs, img_dims)

    save_pickled_data(training_stats, os.path.join(train_stats_save_path_root, f"training_stats_exp_{experiment_number}.pkl"))
    save_pickled_data(iou_per_class_list, os.path.join(train_stats_save_path_root, f"iou_per_class_list_exp_{experiment_number}.pkl"))
    if generate_embeddings:
        save_embeddings(encoder, decoder, "train", **model_kwargs)
        save_embeddings(encoder, decoder,"val", **model_kwargs)
    test_saved_embeddings()
    return training_stats, iou_per_class_list, encoder, decoder

if __name__ == "__main__":
    main()