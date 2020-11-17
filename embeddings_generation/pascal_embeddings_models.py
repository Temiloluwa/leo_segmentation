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
    self.convembeddings = tf.keras.layers.Conv2D(filters=14, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
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
    x = self.conv5(x)
    x = self.dropout5(x)
    x = self.conv5b(x)
    x = self.upsample5(x)
    emb_out = self.convembeddings(x)
    output = self.convfinal(x)
    return [output, emb_out]




def mobilenet_v2_encoder_v2(img_dims):
    num_channels, img_height, img_width = img_dims
    base_model = tf.keras.applications.MobileNetV2(
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=(img_height, img_width, num_channels), #375,500
                    include_top=False,
                    )  # Do not include the ImageNet classifier at the top.

    layer_names = [
        'block_1_expand_relu',   # stride 2
        'block_3_expand_relu',   # stride 4
        'block_6_expand_relu',   # stride 8
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]
    # Freeze the base_model
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
    self.convembeddings = tf.keras.layers.Conv2D(filters=14, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
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
    x = self.conv5(x)
    x = self.dropout5(x)
    x = self.conv5b(x)
    x = self.upsample5(x)
    emb_out = self.convembeddings(x)
    output = self.convfinal(x)
    return [output, emb_out]


class Decoderv2(tf.keras.Model):
  def __init__(self, img_height, img_width, num_channels, base_conv = 256, dropout_probs=0.25):
    super(Decoderv2, self).__init__()
    self.img_height = img_height
    self.img_width = img_width
    self.num_channels = num_channels
    self.conv3 = tf.keras.layers.SeparableConv2D(filters=base_conv//4, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.conv4 = tf.keras.layers.SeparableConv2D(filters=base_conv//8, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.conv5 = tf.keras.layers.SeparableConv2D(filters=base_conv//16, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.convfinal = tf.keras.layers.SeparableConv2D(filters=2, kernel_size=1, strides=1, padding='same', use_bias=False)
    self.concat = tf.keras.layers.Concatenate()
    self.upsample1 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')    
    self.relu1 = tf.keras.layers.Activation('relu')
    self.relu2 = tf.keras.layers.Activation('relu')
    self.relu3 = tf.keras.layers.Activation('relu')
    self.relu4 = tf.keras.layers.Activation('relu')

    self.global_pooling = tf.keras.layers.AveragePooling2D(pool_size=(img_height//8, img_width//8), padding="same")
    self.global_upsampling = tf.keras.layers.UpSampling2D(size=(img_height//8, img_width//8), interpolation='bilinear')
    
        

  def call(self, support_skips, query_skips, support_mask):
    support_latents = support_skips[-1]
    query_latents = query_skips[-1]

    area = self.global_pooling(support_mask) * self.img_height * self.img_width //64
    filtered_features = support_mask * support_latents
    fitered_features = self.global_pooling(filtered_features) / area * self.img_height * self.img_width //64
    fitered_features = self.global_upsampling(fitered_features)
    latents = self.concat([query_latents, fitered_features])
    x = self.relu1(latents)
    x = self.upsample1(x)
    x = self.concat([x, query_skips[-2]])
    x = self.conv3(x)
    x = self.relu2(x)
    x = self.upsample1(x)
    x = self.concat([x, query_skips[-3]])
    x = self.conv4(x)
    x = self.relu3(x)
    x = self.conv5(x)
    x = self.relu4(x)
    output = self.convfinal(x)

    return output, x

class Decoderv3(tf.keras.Model):
  def __init__(self, img_height, img_width, num_channels, base_conv = 256, dropout_probs=0.25):
    super(Decoderv3, self).__init__()
    self.img_height = img_height
    self.img_width = img_width
    self.num_channels = num_channels
    self.conv3 = tf.keras.layers.SeparableConv2D(filters=base_conv//4, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.conv4 = tf.keras.layers.SeparableConv2D(filters=base_conv//8, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.conv5 = tf.keras.layers.SeparableConv2D(filters=base_conv//16, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.convfinal = tf.keras.layers.SeparableConv2D(filters=2, kernel_size=1, strides=1, padding='same', use_bias=False)
    self.concat = tf.keras.layers.Concatenate()
    self.upsample1 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')    
    self.relu1 = tf.keras.layers.Activation('relu')
    self.relu2 = tf.keras.layers.Activation('relu')
    self.relu3 = tf.keras.layers.Activation('relu')
    self.relu4 = tf.keras.layers.Activation('relu')

    self.downsample2 =tf.keras.layers.experimental.preprocessing.Resizing(img_height//4, img_width//4, interpolation="nearest")
    self.downsample1 =tf.keras.layers.experimental.preprocessing.Resizing(img_height//8, img_width//8, interpolation="nearest")

    self.global_pooling = tf.keras.layers.AveragePooling2D(pool_size=(img_height//8, img_width//8), padding="same")
    self.global_upsampling = tf.keras.layers.UpSampling2D(size=(img_height//8, img_width//8), interpolation='bilinear')
        

  def call(self, support_skips, query_skips, support_mask):
    support_latents = support_skips[-1]
    query_latents = query_skips[-1]

    area = self.global_pooling(self.downsample1(support_mask))
    filtered_features = self.downsample1(support_mask) * support_latents
    fitered_features = self.global_pooling(filtered_features) / area 
    fitered_features = self.global_upsampling(fitered_features)
    latents = self.concat([query_latents, fitered_features, self.downsample1(support_mask)])
    x = self.upsample1(latents)
    x = self.concat([x, query_skips[-2], support_skips[-2]*self.downsample2(support_mask), self.downsample2(support_mask)])
    x = self.conv3(x)
    x = self.relu2(x)
    x = self.upsample1(x)
    x = self.concat([x, query_skips[-3], support_skips[-3]*support_mask, support_mask])
    x = self.conv4(x)
    x = self.relu3(x)
    x = self.conv5(x)
    x = self.relu4(x)
    output = self.convfinal(x)

    return output, x


class Decoderv4(tf.keras.Model):
  def __init__(self, img_height, img_width, num_channels, base_conv = 256, dropout_probs=0.25):
    super(Decoderv4, self).__init__()
    self.img_height = img_height
    self.img_width = img_width
    self.num_channels = num_channels
    self.conv2 = tf.keras.layers.SeparableConv2D(filters=base_conv, kernel_size=1, strides=1, padding='same', use_bias=False)
    self.conv3 = tf.keras.layers.SeparableConv2D(filters=base_conv//4, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.conv4 = tf.keras.layers.SeparableConv2D(filters=base_conv//8, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.conv5 = tf.keras.layers.SeparableConv2D(filters=base_conv//16, kernel_size=3, strides=1, padding='same', use_bias=False)
    self.convfinal = tf.keras.layers.SeparableConv2D(filters=2, kernel_size=1, strides=1, padding='same', use_bias=False)
    self.concat = tf.keras.layers.Concatenate()
    self.upsample1 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')    
    self.relu1 = tf.keras.layers.Activation('relu')
    self.relu2 = tf.keras.layers.Activation('relu')
    self.relu3 = tf.keras.layers.Activation('relu')
    self.relu4 = tf.keras.layers.Activation('relu')

    self.downsample2 =tf.keras.layers.experimental.preprocessing.Resizing(img_height//4, img_width//4, interpolation="nearest")
    self.downsample1 =tf.keras.layers.experimental.preprocessing.Resizing(img_height//8, img_width//8, interpolation="nearest")

    self.global_pooling = tf.keras.layers.AveragePooling2D(pool_size=(img_height//8, img_width//8), padding="same")
    self.global_upsampling = tf.keras.layers.UpSampling2D(size=(img_height//8, img_width//8), interpolation='bilinear')
        

  def call(self, support_skips, query_skips, support_mask):
    support_latents = support_skips[-1]
    query_latents = query_skips[-1]
  
    area = self.global_pooling(self.downsample1(support_mask))
    filtered_features = self.downsample1(support_mask) * support_latents
    filtered_features = self.global_pooling(filtered_features) / area 
    filtered_features = self.global_upsampling(filtered_features)

    query_latents = tf.repeat(query_latents, repeats=(5,), axis=0)
    support_mask  = self.downsample1(support_mask)
    latents = tf.concat([query_latents, support_latents, filtered_features, support_mask], axis=-1)
    latents = self.conv2(latents)
    latents = tf.expand_dims(tf.reduce_mean(latents, axis=0), axis=0)
    x = self.upsample1(latents)
    x = self.concat([x, query_skips[-2]])
    x = self.conv3(x)
    x = self.relu2(x)
    x = self.upsample1(x)
    x = self.concat([x, query_skips[-3]])
    x = self.conv4(x)
    x = self.relu3(x)
    x = self.conv5(x)
    x = self.relu4(x)
    output = self.convfinal(x)

    return output, x