import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import pprint
from pathlib import Path
from itertools import cycle, islice
from collections import Counter, defaultdict
from tqdm import tqdm
from PIL import Image
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

pp = pprint.PrettyPrinter(width=41, compact=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "leo_segmentation", "data", "pascal_5i")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", "sofa", 
           "train", "tvmonitor"]
CLASSES_DICT = {i:CLASSES[i] for i in range(len(CLASSES))}
INPUT_SIZE = (384, 512)
BATCH_SIZE = 4
CHOSEN_CLASSES_TRAIN = ["bicycle"]
CHOSEN_CLASSES_VAL = ["car"]
NUM_VALIDATION_EXAMPLES = 1000

"""
PSUEDO CODE FOR CNET DATALOADER
1. Creates a longlist of all training data instances [img name, class index]
2. Train images are all images except the fold to be used or validataion
2. Groups all training data instances by class
3. for each batch based on batch size:
    releases a pair of query,support images
4. for validation select all images from the selected fold:
    - make 3 copies of the list
    - shuffle the concatenation of these three list
    - then select 1000 images
    - select support images for each of the 1000 images
"""
def rotate(img):
    w, h = img.size
    if h > w:
        img = img.transpose(method=Image.ROTATE_270)
    return img


query_mask_transformer = transforms.Resize((INPUT_SIZE[0]//4, INPUT_SIZE[1]//4), interpolation=Image.NEAREST)
support_mask_transformer = transforms.Resize((INPUT_SIZE[0]//8, INPUT_SIZE[1]//8), interpolation=Image.NEAREST)
img_transformer = transforms.Resize(INPUT_SIZE, interpolation=Image.BILINEAR)
to_array = lambda x: np.array(x)


class DataGenerator:
    index = 0
    max_batch_class_length = None
    shuffle = True
    def __init__(self, fold, chosen_classes=CHOSEN_CLASSES_TRAIN, mode="train"):
        self.chosen_classes = chosen_classes
        self.mode = mode
        self.class_to_img_mapping(fold)
        #self.val_classes = CLASSES[fold*5:(fold + 1)*5]
        #self.train_classes = list(set(CLASSES) - set(self.val_classes))

    def class_to_img_mapping(self, fold):
        self.class_img_mapping = defaultdict(list)
        self.class_counts = defaultdict(int)
        train_folds = [0, 1, 2, 3]
        train_folds.remove(fold)
        
        if self.mode == "train":
            for fold_number in train_folds:
                textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'train', f'split{fold_number}_train.txt')
                train_files_list = read_file(textfile_path)
                for fname, class_idx in train_files_list:
                    self.class_img_mapping[CLASSES_DICT[class_idx-1]].append(fname)
                    self.class_counts[CLASSES_DICT[class_idx-1]] += 1

        if self.mode == "val" or self.mode != "train":
            textfile_path = os.path.join(DATA_DIR, 'Binary_map_aug', 'val', f'split{fold}_val.txt')
            val_files_list = read_file(textfile_path)
            for fname, class_idx in val_files_list:
                self.class_img_mapping[CLASSES_DICT[class_idx-1]].append(fname)
                self.class_counts[CLASSES_DICT[class_idx-1]] += 1
        return self

    
    def __iter__(self):
        if not self.shuffle:
            return self
            
        self.idx_chosen_classes = {}
        for _class in self.chosen_classes:
            _class_length = self.class_counts[_class]
            temp = list(range(_class_length))
            np.random.shuffle(temp)
            self.idx_chosen_classes[_class] = temp
            
        for _class in self.chosen_classes:
            _class_length = self.class_counts[_class]
            if self.max_batch_class_length is None:
                self.__len__()

            if _class_length != self.max_batch_class_length:
                _class_idx = self.idx_chosen_classes[_class]
                _class_idx = list(islice(cycle(_class_idx), self.max_batch_class_length))
                # cycle the indices until they have the same length has max batch length
                self.idx_chosen_classes[_class] = _class_idx

        for _class in self.chosen_classes:
            assert len(self.idx_chosen_classes[_class]) == self.max_batch_class_length, "all classes should be of the same length"
            assert len(set(self.idx_chosen_classes[_class])) == self.class_counts[_class], "error occured generating indices"
        
        if self.mode == "val" or self.mode != "train":
            assert len(self.idx_chosen_classes[_class]) == NUM_VALIDATION_EXAMPLES, f"num examples should be {NUM_VALIDATION_EXAMPLES}"

        return self

    def get_one_class_batch(self, chosen_class):
        idx = self.index
        fname_list = self.class_img_mapping[chosen_class]
        indices = self.idx_chosen_classes[chosen_class]
        query_imgs = [] 
        support_imgs = []
        query_masks = []
        support_masks = []

        for i in range(BATCH_SIZE):
            query_fn = fname_list[indices[idx]]
            end_loop = False 
            while not end_loop:
                support_fn = np.random.choice(fname_list, size=1)[0]
                if support_fn != query_fn:
                    end_loop = True
            idx += 1

            query_img_path = os.path.join(DATA_DIR, 'JPEGImages', f'{query_fn}.jpg')
            query_imgs.append(to_array(img_transformer(rotate(Image.open(query_img_path)))))
            
            support_img_path = os.path.join(DATA_DIR, 'JPEGImages', f'{support_fn}.jpg')
            support_imgs.append(to_array(img_transformer(rotate(Image.open(support_img_path)))))
            
            query_mask_path = os.path.join(DATA_DIR, 'Binary_map_aug', self.mode, f'{CLASSES.index(chosen_class) + 1}', f'{query_fn}.png')
            query_masks.append(to_array(query_mask_transformer(rotate(Image.open(query_mask_path)))))
            
            support_mask_path = os.path.join(DATA_DIR, 'Binary_map_aug', self.mode, f'{CLASSES.index(chosen_class) + 1}', f'{support_fn}.png')
            support_masks.append(to_array(support_mask_transformer(rotate(Image.open(support_mask_path)))))
        
        return query_imgs, support_imgs, query_masks, support_masks


    def __next__(self):
        query_imgs = [] 
        support_imgs = []
        query_masks = []
        support_masks = []
        for _class in self.chosen_classes:
            batch = self.get_one_class_batch(_class)
            query_imgs.extend(batch[0])
            support_imgs.extend(batch[1])
            query_masks.extend(batch[2])
            support_masks.extend(batch[3])
        
        query_imgs = np.array(query_imgs)
        query_imgs = (query_imgs/255).astype("float32")
        support_imgs = np.array(support_imgs)
        support_imgs = (support_imgs/255).astype("float32")
        query_masks = np.array(query_masks)
        query_masks = (query_masks/255).astype("float32")
        support_masks = np.array(support_masks)
        support_masks = (support_masks/255).astype("float32")
        if self.index + BATCH_SIZE >= self.max_batch_class_length:
            raise StopIteration

        self.index += BATCH_SIZE
        return query_imgs, support_imgs, query_masks, support_masks

    def __len__(self):
        self.max_batch_class_length = np.max([self.class_counts[_class] for _class in self.chosen_classes])
        if self.mode == "val" or self.mode != "train":
            self.max_batch_class_length = NUM_VALIDATION_EXAMPLES
        return self.max_batch_class_length//BATCH_SIZE


def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

img_height, img_width, num_channels = *INPUT_SIZE, 3
base_model = tf.keras.applications.ResNet50(
        weights="imagenet", 
        input_shape=(img_height, img_width, num_channels),
        include_top=False,
    )

layer_names = [
    'conv2_block3_out',
    'conv3_block4_out',
] 

layers = [base_model.get_layer(name).output for name in layer_names]
encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
encoder.trainable = False

support_inputs = tf.keras.Input(shape=((img_height, img_width, num_channels)))
query_inputs = tf.keras.Input(shape=((img_height, img_width, num_channels)))
support_masks = tf.keras.Input(shape=((img_height//8, img_width//8, 1)))

support_skips = encoder(support_inputs, training=False)
query_skips = encoder(query_inputs, training=False)

global_pooling = tf.keras.layers.AveragePooling2D(pool_size=(img_height//8, img_width//8), padding="same")
global_upsampling = tf.keras.layers.UpSampling2D(size=(img_height//8, img_width//8), interpolation='bilinear')

base_conv = 256
conv0m = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=2, padding='same', use_bias=False)
conv0 = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu",  dilation_rate=2, strides=1, padding='same', use_bias=False)
conv1 = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu",  dilation_rate=2, strides=1, padding='same', use_bias=False)
conv2a = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=1, activation="relu", strides=1, padding='same', use_bias=False)
conv2b = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=1, activation="relu", strides=1, padding='same', use_bias=False)
conv2c = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", dilation_rate=6, strides=1, padding='same', use_bias=False)
conv2d = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", dilation_rate=12, strides=1, padding='same', use_bias=False)
conv2e = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", dilation_rate=18, strides=1, padding='same', use_bias=False) 
conv3 = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=1, activation="relu", strides=1, padding='same', use_bias=False)
conv4a = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
conv4b = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
conv5a = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
conv5b = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
conv6a = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
conv6b = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
convfinal = tf.keras.layers.SeparableConv2D(filters=2, kernel_size=1, activation="relu", strides=1, padding='same', use_bias=False)
up_conv1 = tf.keras.layers.Conv2DTranspose(filters=base_conv, kernel_size=3, activation="relu", strides=2, padding='same', use_bias=False)
conv8a = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
conv8b = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=3, activation="relu", strides=1, padding='same', use_bias=False)
concat = tf.keras.layers.Concatenate()


query = concat([query_skips[-1], conv0m(query_skips[-2])])
support = concat([support_skips[-1], conv0m(support_skips[-2])])
query = conv0(query)
support = conv0(support)
feature_shape = query.shape[1:-1]
area = global_pooling(support_masks) * img_height * img_width // 8 * 8
filtered_features = support_masks * support_skips[-1]
fitered_features = global_pooling(filtered_features)/area * img_height * img_width
fitered_features = global_upsampling(fitered_features)
latents = concat([query, fitered_features])
o = conv1(latents)
o_ = conv4a(o)
o  = o + conv4b(o_)
o_ = conv5a(o)
o  = o + conv5b(o_)
o_ = conv6a(o)
o  = o + conv6b(o_)
global_features = global_pooling(o)
global_features = global_upsampling(global_features)
o = concat([global_features, conv2a(o), conv2b(o), conv2c(o), conv2d(o), conv2e(o)])
o = conv3(o)
o = up_conv1(o)
o = conv8a(o)
o = conv8b(o)
o = convfinal(o)
model = tf.keras.Model(inputs=[support_inputs, query_inputs, support_masks], outputs=[o, fitered_features])
model.summary()

def compute_loss(model, x, masks):
    logits, _ = model(x)
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    scce_loss = scce(masks, logits)
    return scce_loss

def calc_iou_per_class(model, X, y):
    ious = []
    for i in range(len(X[0])):
        X_s, X_q, X_m_s = X
        X_s = np.expand_dims(X_s[i], 0)
        X_q = np.expand_dims(X_q[i], 0)
        X_m_s = tf.expand_dims(X_m_s[i], 0)
        
        logit, _ = model((X_s, X_q, X_m_s))
        target = np.expand_dims(y[i], 0).astype(int)  
        pred = np.argmax(logit.numpy(),axis=-1).astype(int)
        iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
        ious.append(iou)
    return np.mean(ious)


def plot_prediction(model, imgs, masks):  
    fig = plt.figure(figsize=(30, 30))
    X_s, X_q, X_m_s = imgs
    X_s = np.expand_dims(X_s, 0) if X_s.ndim==3 else X_s
    X_q = np.expand_dims(X_q, 0) if X_q.ndim==3 else X_q
    X_m_s = np.expand_dims(X_m_s, 0) if X_m_s.ndim==2 else X_m_s
    masks = np.expand_dims(masks, 0) if masks.ndim==2 else masks
    pred_masks, _ = model((X_s, X_q, X_m_s))
    pred_masks = pred_masks.numpy()
    pred_masks = np.argmax(pred_masks, axis=-1)
    num_plots = len(imgs)
    for i in range(num_plots):
        fig.add_subplot(num_plots,3,i*3+1)
        plt.imshow((X_q[i]*255).astype("uint8"))

        fig.add_subplot(num_plots,3,i*3+2)
        plt.imshow(masks[i], cmap="gray")
        plt.title("ground truth")

        fig.add_subplot(num_plots,3,i*3+3)
        plt.imshow(pred_masks[i], cmap="gray")
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

def resize_masks(masks, size=(img_height//8, img_width//8)):
  masks = np.transpose(masks, (1,2,0))
  masks = tf.image.resize(masks, size, method="nearest")
  masks = np.transpose(masks, (2,0,1))
  return masks


@tf.function
def train_step(model, x, masks, optimizer):
    """
    Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, masks)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

tf.keras.backend.clear_session()
chosen_classes = ["chair"]
train_data_generator = DataGenerator(fold=0, chosen_classes=chosen_classes)
#val_data_generator = DataGenerator(fold=0, chosen_classes=CHOSEN_CLASSES_VAL, mode="val")

num_batches = len(train_data_generator)
train_test_split = 0.7
train_split = int(0.7 * num_batches)
epochs = 60
freq = 1
bs = 10
lr = 0.00025
training_stats = []
epoch_times = []
optimizer = tf.keras.optimizers.Adam(lr)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    train_batch_loss = []
    val_batch_loss = []

    train_data_generator.index = 0
    for i, (query_imgs, support_imgs, query_masks, support_masks) in enumerate(tqdm(train_data_generator)):
        if i <= train_split:
            batch_loss = train_step(model, [support_imgs, query_imgs, support_masks], query_masks, optimizer)
            train_batch_loss.append(batch_loss.numpy())
        else:
            batch_loss = compute_loss(model, [support_imgs, query_imgs, support_masks], query_masks)
            val_batch_loss.append(batch_loss.numpy())

    train_loss = float(np.mean(train_batch_loss))
    val_loss = float(np.mean(val_batch_loss))

    train_data_generator.index = 0
    train_data_generator.shuffle = False
    
    for i, (query_imgs, support_imgs, query_masks, support_masks) in enumerate(tqdm(train_data_generator)):
        if i <= train_split:
            train_iou = calc_iou_per_class(model, [support_imgs, query_imgs, support_masks], query_masks)
        else:
            val_iou = calc_iou_per_class(model, [support_imgs, query_imgs, support_masks], query_masks)
  
    end_time = time.time()
    epoch_time = (end_time - start_time)/60
    
    if epoch % freq == 0:
        print(f"Epoch:{epoch:03d}- Epoch Time:{epoch_time:3f} train loss:{train_loss:3f}, train_iou:{train_iou:3f}, val loss:{val_loss:3f}, val_iou:{val_iou:3f}")
    epoch_times.append(epoch_time)
   
    training_stats.append({
      "epoch":epoch,
      "train loss":train_loss,
      "val loss":val_loss,
      "epoch time": epoch_time,
      "train iou": train_iou,
      "val iou": val_iou
    })
    
    #if epoch % freq == 0:
    #    plot_prediction(model, [support_images, query_images, support_masks], query_masks)
    #    plot_stats(pd.DataFrame(training_stats), "train loss")
    #    plot_stats(pd.DataFrame(training_stats), "val loss")
