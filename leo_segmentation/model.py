import os
import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .utils import display_data_shape, get_named_dict, calc_iou_per_class,\
    log_data, load_config, list_to_tensor, model_dir

config = load_config()
hyp = config.hyperparameters


def mobilenet_v2_encoder(img_dims):
    """ Initialize the encoder using weights from mobilenetv2"""
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
    """ Decoder for the LEO model"""
    def __init__(self, dropout_probs=hyp.dropout_rate):
        super(Decoder, self).__init__()
        self.conv1 = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*1, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv1b = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*1, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv2 = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv2b = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv3 = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*3, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv3b = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*3, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv4 = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*4, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv4b = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*4, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv5 = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*5, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.conv5b = tf.keras.layers.SeparableConv2D(filters=hyp.base_num_covs*5, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.convfinal = tf.keras.layers.SeparableConv2D(filters=2, kernel_size=3, strides=1, padding='same', activation="relu", use_bias=False)
        self.upsample1 = tf.keras.layers.Conv2DTranspose(hyp.base_num_covs*1, 3, strides=2,padding='same')
        self.upsample2 = tf.keras.layers.Conv2DTranspose(hyp.base_num_covs*2, 3, strides=2,padding='same')
        self.upsample3 = tf.keras.layers.Conv2DTranspose(hyp.base_num_covs*3, 3, strides=2,padding='same')
        self.upsample4 = tf.keras.layers.Conv2DTranspose(hyp.base_num_covs*4, 3, strides=2,padding='same')
        self.upsample5 = tf.keras.layers.Conv2DTranspose(hyp.base_num_covs*5, 3, strides=2,padding='same')
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
        output = self.upsample5(x)
        return output


class LEO(tf.keras.Model):
    """
    contains functions to perform latent embedding optimization
    """
    def __init__(self, mode="meta_train"):
        super(LEO, self).__init__()
        img_dims = config.data_params.img_dims
        self.config = config
        self.mode = mode
        self.img_dims = (img_dims.channels, img_dims.height, img_dims.width)
        self.encoder = mobilenet_v2_encoder(self.img_dims)
        self.decoder = Decoder()
        self.optimizer = tf.keras.optimizers.Adam(hyp.outer_loop_lr)
        self.init = tf.initializers.GlorotUniform()
        self.seg_weight = tf.Variable(self.init((3, 3, hyp.base_num_covs*5 + 3, 2)))
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def forward_encoder(self, x):
        """Performs forward pass through the encoder"""
        encoder_outputs = self.encoder(x)
        return encoder_outputs

    def forward_decoder(self, encoder_outputs):
        """Performs forward pass through the decoder"""
        output = self.decoder(encoder_outputs)
        return output

    def forward_segnetwork(self, decoder_out, x, weight):
        """ 
            - Receives features from the decoder
            - Concats the features with input image
            - Convolution layer acts on the concatenated input
            Args:
                decoder_out(tf.tensor): decoder output features
                x(tf.tensor): input images
                weight(tf.tensor): kernels for the segmentation network
            Returns:
                pred(tf.tensor): predicted logits
        """
        x = tf.concat([decoder_out, x], -1)
        pred = tf.nn.convolution(x, weight, strides=1, padding='SAME')
        return pred

    def __call__(self, x, latents=None):
        """
           Performs a forward pass through the entire network
           - The Autoencoder generates features using the inputs
           - Features are concatenated with the inputs
           - The concatenated features are segmented
           Args:
                x(tf.tensor): input image
                latents(tf.tensor): output of the bottleneck
           Returns:
                latents(tf.tensor): output of the bottleneck
                features(tf.tensor): output of the decoder
                pred(tf.tensor): predicted logits
        
        """
        encoder_outputs = self.forward_encoder(x)
        if latents != None:
            encoder_outputs = encoder_outputs[:4] + [latents]
        else:
            latents = encoder_outputs[-1]
        features = self.forward_decoder(encoder_outputs)
        pred = self.forward_segnetwork(features, x, self.seg_weight)
        return latents, features, pred
        
    @tf.function
    def leo_inner_loop(self, x, y):
        """
        This function performs innerloop optimization
            - It updates the latents taking gradients wrt the training loss
            - It generates better features after the latents are updated
        Args:
            x(tf.tensor): input training image
            y(tf.tensor): input training mask
        
        Returns:
            seg_weight_grad(tf.tensor): The last gradient of the training loss wrt to 
                                        the segmenation weights
            features(tf.tensor): The last generated features from the decoder
        """
        
        inner_lr = self.config.hyperparameters.inner_loop_lr
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.seg_weight)
            latents, _, pred = self.__call__(x)
            tr_loss =  self.loss_fn(y, pred) 
            for _ in range(self.config.hyperparameters.num_adaptation_steps):
                latents_grad = tape.gradient(tr_loss, latents)
                latents -= inner_lr * latents_grad
                latents, features, pred = self.__call__(x, latents)
                tr_loss =  self.loss_fn(y, pred)
        seg_weight_grad = tape.gradient(tr_loss, self.seg_weight)   
        return seg_weight_grad, features

    @tf.function
    def finetuning_inner_loop(self, data_dict, tr_features, seg_weight_grad, mode):
        """
        Finetunes the segmenation weights/kernels by performing MAML
        Args:
            data_dict (dict): contains tr_imgs, tr_masks, val_imgs, val_masks
            tr_features (tf.tensor): tensor containing decoder features
            segmentation_grad (tf.tensor): gradients of the training loss to the segmenation weights
        Returns:
            val_loss (tf.tensor): validation loss
            seg_weight_grad (tf.tensor): gradient of validation loss wrt segmentation weights
            decoder_gradients (tf.tensor): gradient of validation loss wrt decoder weights
        """
        finetuning_lr = self.config.hyperparameters.finetuning_lr
        weight = self.seg_weight - finetuning_lr * seg_weight_grad
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(weight)
            for _ in range(self.config.hyperparameters.num_finetuning_steps - 1):
                pred = self.forward_segnetwork(tr_features, data_dict.tr_imgs, weight)
                tr_loss =  self.loss_fn(data_dict.tr_masks, pred)
                seg_weight_grad = tape.gradient(tr_loss, weight)
                weight -= finetuning_lr * seg_weight_grad

            if mode == "meta_train": 
                encoder_outputs = self.forward_encoder(data_dict.val_imgs)
                features = self.forward_decoder(encoder_outputs)
                pred = self.forward_segnetwork(features, data_dict.val_imgs, weight)
                val_loss =  self.loss_fn(data_dict.val_masks, pred)
                seg_weight_grad =  tape.gradient(val_loss, weight)
                decoder_gradients = tape.gradient(val_loss, self.decoder.trainable_variables)
                return val_loss, seg_weight_grad, decoder_gradients, pred, weight
            else:
                return None, None, None, None, weight

               
    def evaluate(self, data, prediction, weight, transformers, mode):
        img_transformer, mask_transformer = transformers
        if mode == "meta_train":
            mean_iou = calc_iou_per_class(prediction, data.val_masks)
            return mean_iou, None
        else:
            mean_ious = []
            val_losses = []
            val_img_paths = data.val_imgs
            val_mask_paths = data.val_masks
            for _img_path, _mask_path in tqdm(zip(val_img_paths, val_mask_paths)):
                input_img = list_to_tensor(_img_path, img_transformer)
                input_mask = list_to_tensor(_mask_path, mask_transformer)
                encoder_outputs = self.forward_encoder(input_img)
                features = self.forward_decoder(encoder_outputs)
                prediction = self.forward_segnetwork(features, input_img, weight)
                val_loss =  self.loss_fn(input_mask, prediction)
                mean_iou = calc_iou_per_class(prediction, input_mask)
                mean_ious.append(mean_iou)
                val_losses.append(val_loss)
            mean_iou = np.mean(mean_ious)
            val_loss = np.mean(val_losses)
        return mean_iou, val_loss
         
         
    def compute_loss(self, metadata, train_stats, transformers, mode="meta_train"):
        """
            Performs meta optimization across tasks
            returns the meta validation loss across tasks 
        Args:
            metadata(dict): dictionary containing training data
            train_stats(object): object that stores training statistics
            transformers(tuple): tuple of train and validation datatransformers 
            mode(str): meta_train, meta_val or meta_test
        Returns:
            total_val_loss(float32): meta-validation loss
            train_stats(object): object that stores training statistics 
        """
        num_tasks = len(metadata[0])
        if train_stats.episode % self.config.display_stats_interval == 1:
            display_data_shape(metadata)

        classes = metadata[4]
        total_val_loss = []
        kl_losses = None
        mean_iou_dict = {} 
        total_gradients = None
        for batch in range(num_tasks):
            data = get_named_dict(metadata, batch)
            seg_weight_grad, features = self.leo_inner_loop(data.tr_imgs, data.tr_masks)
            tr_val_loss, seg_weight_grad, decoder_gradients, prediction, weight = self.finetuning_inner_loop(data, features, \
                                                                        seg_weight_grad, mode)
            if mode == "meta_train":
                decoder_gradients = [grad/num_tasks for grad in decoder_gradients]
                if total_gradients == None:
                    total_gradients = decoder_gradients
                    seg_weight_grad = seg_weight_grad/num_tasks
                else:
                    total_gradients = [total_gradients[i] + decoder_gradients[i] for i in range(len(decoder_gradients))]
                    seg_weight_grad += seg_weight_grad/num_tasks
                
            mean_iou, val_loss = self.evaluate(data, prediction, weight, transformers, mode)
            val_loss = tr_val_loss if mode == "meta_train" else val_loss
            mean_iou_dict[classes[batch]] = mean_iou
            total_val_loss.append(val_loss)
        

        if mode == "meta_train":
            self.optimizer.apply_gradients(zip(total_gradients, self.decoder.trainable_variables))
            self.optimizer.apply_gradients([(seg_weight_grad, self.seg_weight)])
        total_val_loss = float(sum(total_val_loss)/len(total_val_loss))  
        stats_data = {
            "mode": mode,
            "kl_loss": 0,
            "total_val_loss":total_val_loss,
            "mean_iou_dict":mean_iou_dict
        }
        train_stats.update_stats(**stats_data)
        return total_val_loss, train_stats


#TO-DO: Change from pytorch to tensorflow
def save_model(model, optimizer, config, stats):
    """
    Save the model while training based on check point interval
    
    if episode number is not -1 then a prompt to delete checkpoints occur if 
    checkpoints for that episode number exits.
    This only occurs if the prompt_deletion flag in the experiment dictionary
    is true else checkpoints that already exists are automatically deleted

    Args:
        model - trained model
        optimizer - optimized weights
        config - global config
        stats - dictionary containing stats for the current episode
    
    Returns:
    """
    data_to_save = {
        'mode': stats.mode,
        'episode': stats.episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_val_loss': stats.total_val_loss
    }

    checkpoint_path = os.path.join(model_dir, f"checkpoint_{stats.episode}.pth.tar")
    if not os.path.exists(checkpoint_path):
        torch.save(data_to_save, checkpoint_path)
    else:
        trials = 0
        while trials < 3:
            if experiment.prompt_deletion:
                print(f"Are you sure you want to delete checkpoint: {stats.episode}")
                print(f"Type Yes or y to confirm deletion else No or n")
                user_input = input()
            else:
                user_input = "Yes"
            positive_options = ["Yes", "y", "yes"]
            negative_options = ["No", "n", "no"]
            if user_input in positive_options:
                # delete checkpoint
                os.remove(checkpoint_path)
                torch.save(data_to_save, checkpoint_path)
                log_filename = os.path.join(model_dir, "model_log.txt")
                msg = msg = f"\n*********** checkpoint {stats.episode} was deleted **************"
                log_data(msg, log_filename)
                break

            elif user_input in negative_options:
                raise ValueError("Supply the correct episode number to start experiment")
            else:
                trials += 1
                print("Wrong Value Supplied")
                print(f"You have {3 - trials} left")
                if trials == 3:
                    raise ValueError("Supply the correct answer to the question")


#TO-DO: Change from pytorch to tensorflow
def load_model(config):

    """
    Loads the model
    Args:
        config - global config
        **************************************************
        Note: The episode key in the experiment dict
        implies the checkpoint that should be loaded
        when the model resumes training. If episode is
        -1, then the latest model is loaded else it loads
        the checkpoint at the supplied episode
        *************************************************
    Returns:
        leo :loaded model that was saved
        optimizer: loaded weights of optimizer
        stats: stats for the last saved model
    """
    checkpoints = os.listdir(model_dir)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[-1] == ".tar"]
    max_cp = max([int(cp.split(".")[0].split("_")[1]) for cp in checkpoints])
    #if experiment.episode == -1, load latest checkpoint
    episode = max_cp if experiment.episode == -1 else experiment.episode
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    log_filename = os.path.join(model_dir, "model_log.txt")
    msg = f"\n*********** checkpoint {episode} was loaded **************" 
    log_data(msg, log_filename)
    
    leo = LEO()
    optimizer = torch.optim.Adam(leo.parameters(), lr=hyp.outer_loop_lr)
    leo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mode = checkpoint['mode']
    total_val_loss = checkpoint['total_val_loss']
  
    stats = {
        "mode": mode,
        "episode": episode,
        "total_val_loss": total_val_loss
        }

    return leo, optimizer, stats
