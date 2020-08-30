import numpy as np
import tensorflow as tf
#list of keras models
#https://www.tensorflow.org/api_docs/python/tf/keras/applications

def init_mobilenet_v2_backbone(num_channels, img_height, img_width):
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
    return model

def init_xception_backbone(num_channels, img_height, img_width):
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",  
        input_shape=(img_height, img_width, num_channels), 
        include_top=False,
    )  

    layer_names = [
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]
    # Freeze the base_model
    encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
    encoder.trainable = False

    inputs = tf.keras.Input(shape=((img_height, img_width, num_channels)))
    # Downsampling through the model
    skips = encoder(inputs, training=False)

    model = tf.keras.Model(inputs=inputs, outputs=[])
    model.summary()
    return model