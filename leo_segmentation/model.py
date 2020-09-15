import numpy as np
import tensorflow as tf
#list of keras models
#https://www.tensorflow.org/api_docs/python/tf/keras/applications

def mobilenet_v2_encoder(img_dims):
    num_channels, img_height, img_width = img_dims
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",  
        input_shape=(img_height, img_width, num_channels), 
        include_top=False,
    )  
    layer_names = [
        'block_1_expand_relu',   
        'block_3_expand_relu',   
        'block_6_expand_relu',   
        'block_13_expand_relu',  
        'block_16_project',      
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]
    encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
    encoder.trainable = False
    return encoder
    
class Decoder(tf.keras.Model):
  def __init__(self, dropout_probs=0.25):
    super(Decoder, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv1b = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv2 = tf.keras.layers.Conv2D(filters=8*2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv2b = tf.keras.layers.Conv2D(filters=8*2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv3 = tf.keras.layers.Conv2D(filters=8*3, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv3b = tf.keras.layers.Conv2D(filters=8*3, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv4 = tf.keras.layers.Conv2D(filters=8*4, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv4b = tf.keras.layers.Conv2D(filters=8*4, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv5 = tf.keras.layers.Conv2D(filters=8*5, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.conv5b = tf.keras.layers.Conv2D(filters=8*5, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.convfinal = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
    self.upsample1 = tf.keras.layers.Conv2DTranspose(8, 3, strides=2,padding='same')
    self.upsample2 = tf.keras.layers.Conv2DTranspose(8*2, 3, strides=2,padding='same')
    self.upsample3 = tf.keras.layers.Conv2DTranspose(8*3, 3, strides=2,padding='same')
    self.upsample4 = tf.keras.layers.Conv2DTranspose(8*4, 3, strides=2,padding='same')
    self.upsample5 = tf.keras.layers.Conv2DTranspose(8*5, 3, strides=2,padding='same')
    self.concat = tf.keras.layers.Concatenate()
    self.dropout1 = tf.keras.layers.Dropout(dropout_probs)
    self.dropout2 = tf.keras.layers.Dropout(dropout_probs)
    self.dropout3 = tf.keras.layers.Dropout(dropout_probs)
    self.dropout4 = tf.keras.layers.Dropout(dropout_probs)
    self.dropout5 = tf.keras.layers.Dropout(dropout_probs)

  def call(self, encoder_outputs):
    x = self.conv1(encoder_outputs[-1])
    x = self.dropout1(x)
    x = self.conv1b(x)
    x = self.upsample1(x)
    x = self.concat([x, encoder_outputs[-2]])
    x = self.conv2(x)
    x = self.dropout2(x)
    x = self.conv2b(x)
    x = self.upsample2(x)
    x = self.concat([x, encoder_outputs[-3]])
    x = self.conv3(x)
    x = self.dropout3(x)
    x = self.conv3b(x)
    x = self.upsample3(x)
    x = self.concat([x, encoder_outputs[-4]])
    x = self.conv4(x)
    x = self.dropout4(x)
    x = self.conv4b(x)
    x = self.upsample4(x)
    x = self.concat([x, encoder_outputs[-5]])
    emb_out = self.conv5(x)
    x = self.dropout5(emb_out)
    x = self.conv5b(x)
    x = self.upsample5(x)
    output = self.convfinal(x)
    return [output, emb_out]
