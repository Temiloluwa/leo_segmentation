# Architecture definition
# Computational graph creation
import torch, os, numpy as np
import tensorflow as tf
from utils import display_data_shape, get_named_dict, calc_iou_per_class,\
    log_data, load_config, summary_write_masks

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
    
class LEO:
    """
    contains functions to perform latent embedding optimization
    """
    def __init__(self, config, mode="meta_train"):
        super(LEO, self).__init__()
        self.config = config
        self.mode = mode
        self.img_dims = (3, 384, 512)
        self.encoder = mobilenet_v2_encoder(self.img_dims)
        self.decoder = Decoder()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        #seg_network =  nn.Conv2d(16 + 8*4 + 3 , 2, kernel_size=3, stride=1, padding=1)
        #self.seg_weight = seg_network.weight.detach().to(self.device)
        #self.seg_weight.requires_grad = True
        #self.seg_bias = seg_network.bias.detach().to(self.device)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def forward_encoder(self, x):
        encoder_outputs = self.encoder(x)
        latents, features = encoder_outputs[:-1], encoder_outputs[-1] 
        return encoder_outputs

    def forward_decoder(self, encoder_outputs):
        output, _ = self.decoder(encoder_outputs)
        return output

    def call(self, x):
        o = self.forward_encoder(x)
        o = self.forward_decoder(o)
        return o

    def evaluate(self, metadata):
        num_tasks = len(metadata[0])
    
        def cal_iou(x, target, class_name, batch):
            iou_per_class = []
            logits = self.call(x)
            pred = np.argmax(logits.numpy(),axis=-1).astype(int)
            target = target.astype(int)
            iou = np.sum(np.logical_and(target, pred))/np.sum(np.logical_or(target, pred))
            iou_per_class.append(iou)
            mean_iou_per_class = np.mean(iou_per_class)
            print(f"class: {class_name[batch]}, iou: {mean_iou_per_class}")
        
        print("**Train**")
        for batch in range(num_tasks):   
            data_dict = get_named_dict(metadata, batch)
            cal_iou(data_dict.tr_imgs, data_dict.tr_masks, metadata[-1], batch)
        
        print("**Validation**")
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            cal_iou(data_dict.val_imgs, data_dict.val_masks, metadata[-1], batch)



    """
    def forward_segnetwork(self, decoder_out, x, weight, bias, target):
        o = torch.cat([decoder_out, x], dim=1)
        pred = F.conv2d(o, weight, bias, padding=1)
        loss = self.loss_fn(pred, target.long())
        return pred, loss
    
    def forward_decoder(self, x, latents, weight, bias, features, target):
        decoder_output = self.decoder(latents, features)
        pred, loss = self.forward_segnetwork(decoder_output, x, weight, bias, target)
        return loss, pred, decoder_output

    def forward(self, x, weight, bias, target, latents=None):
        if latents == None:
            latents, features = self.forward_encoder(x)
        else:
            _, features = self.forward_encoder(x)
        loss, pred, decoder_output = self.forward_decoder(x, latents, weight, bias, features, target)
        return loss, pred, latents, decoder_output
    """    
    def leo_inner_loop(self, x, weight, bias, target):
        """
        This function does "latent code optimization" that is back propagation  until latent codes and
        updating the latent weights
        Args:
            data (dict) : contains tr_imgs, tr_masks, val_imgs, val_masks
            latents (tensor) : shape ((num_classes * num_eg_per_class), latent_channels, H, W)
        Returns:
            tr_loss : computed as crossentropyloss (groundtruth--> tr_imgs_mask, prediction--> einsum(tr_imgs, segmentation_weights))
            segmentation_weights : shape(num_classes, num_eg_per_class, channels, H, W)
        """
        inner_lr = self.config.hyperparameters.inner_loop_lr
        tr_loss, _ , latents, _ = self.forward(x, weight, bias, target)
        #initial_latents = latents.clone()   
        for _ in range(self.config.hyperparameters.num_adaptation_steps):
            latents_grad = torch.autograd.grad(tr_loss, [latents], create_graph=False)[0]
            with torch.no_grad():
                latents -= inner_lr * latents_grad
            tr_loss, _ , latents, tr_decoder_output  = self.forward(x, weight, bias, target, latents)
        return tr_loss, tr_decoder_output

    def finetuning_inner_loop(self, data, tr_loss, tr_decoder_output, weight, bias):
        """
        This function does "segmentation_weights optimization"
        Args:
            data (dict) : contains tr_imgs, tr_masks, val_imgs, val_masks
            leo_loss (tensor_0shape) : computed as crossentropyloss (groundtruth--> tr_imgs_mask, prediction--> einsum(tr_imgs, segmentation_weights))
           segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        Returns:
            val_loss (tensor_0shape) : computed as crossentropyloss (groundtruth--> val_imgs_mask, prediction--> einsum(val_imgs, segmentation_weights))
        """
        finetuning_lr = self.config.hyperparameters.finetuning_lr
        for _ in range(self.config.hyperparameters.num_finetuning_steps):
            grad_weight, grad_bias = torch.autograd.grad(tr_loss, [weight, bias])
            with torch.no_grad():
                weight -= finetuning_lr * grad_weight
                bias -= finetuning_lr * grad_bias
            _, tr_loss = self.forward_segnetwork(tr_decoder_output, data.tr_imgs, weight, bias, data.tr_masks)
        val_loss, _, _, _ = self.forward(data.val_imgs, weight, bias, data.val_masks)
        return val_loss
    
    
    def compute_loss(self, metadata, train_stats, mode="meta_train"):
        """
        Computes the  outer loop loss

        Args:
            model (object) : leo model
            meta_dataloader (object): Dataloader 
            train_stats: (object): train stats object
            config (dict): config
            mode (str): meta_train, meta_val or meta_test
        Returns:
            (tuple) total_val_loss (list), train_stats
        """
        num_tasks = len(metadata[0])
        if train_stats.episode % self.config.display_stats_interval == 1:
            display_data_shape(metadata)

        total_gradients = None
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            #weights = self.seg_weight.clone()
            #bias = self.seg_bias.clone()
            #tr_loss, tr_decoder_output = self.leo_inner_loop(data_dict.tr_imgs, weights, bias, data_dict.tr_masks)
            total_val_loss = []
            with tf.GradientTape() as tape:
                pred = self.call(data_dict.tr_imgs)
                tr_loss =  self.loss_fn(data_dict.tr_masks, pred)
                total_val_loss.append(tr_loss)
                #val_loss = self.finetuning_inner_loop(data_dict, tr_loss, tr_decoder_output, weights, bias)
            gradients = tape.gradient(tr_loss, self.decoder.trainable_variables)
            gradients = [grad/num_tasks for grad in gradients]
            if total_gradients == None:
                total_gradients = gradients
            else:
                total_gradients = [total_gradients[i] + gradients[i] for i in range(len(gradients))]
        total_val_loss = sum(total_val_loss)/len(total_val_loss)
        self.optimizer.apply_gradients(zip(total_gradients, self.decoder.trainable_variables))
        """"
        stats_data = {
            "mode": mode,
            "kl_loss": 0,
            "total_val_loss":total_val_loss
        }
        train_stats.update_stats(**stats_data)
        """
        return total_val_loss

    def evaluate_val_imgs(self, metadata, classes, train_stats, writer):
        log_msg = ""
        num_tasks = len(metadata[0])
        for batch in range(num_tasks):
            data_dict = get_named_dict(metadata, batch)
            weights = self.seg_weight.clone()
            bias = self.seg_bias.clone()
            _, predictions, _, _ = self.forward(data_dict.val_imgs, weights, bias, data_dict.val_masks)
            iou = calc_iou_per_class(predictions, data_dict.val_masks)
            batch_msg = f"\nClass: {classes[batch]}, Episode: {train_stats.episode}, Val IOU: {iou}"
            print(batch_msg[1:])
            log_msg += batch_msg
            grid_title = f"pred_{train_stats.episode}_class_{classes[batch]}"
            summary_write_masks(predictions, writer, grid_title)
            grid_title = f"ground_truths_{train_stats.episode}_class_{classes[batch]}"
            summary_write_masks(data_dict.val_masks, writer, grid_title, ground_truth=True)
        log_filename = os.path.join(os.path.dirname(__file__), "data", "models",\
                         f"experiment_{self.config.experiment.number}", "val_stats_log.txt")
        log_data(log_msg, log_filename)


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
        'kl_loss': stats.kl_loss,
        'total_val_loss': stats.total_val_loss
    }

    experiment = config.experiment
    model_root = os.path.join(config.data_path, "models")
    model_dir = os.path.join(model_root, "experiment_{}" \
                             .format(experiment.number))

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
    experiment = config.experiment
    model_dir  = os.path.join(config.data_path, "models", "experiment_{}"\
                 .format(experiment.number))
    
    checkpoints = os.listdir(model_dir)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[-1] == ".tar"]
    max_cp = max([int(cp.split(".")[0].split("_")[1]) for cp in checkpoints])
    #if experiment.episode == -1, load latest checkpoint
    episode = max_cp if experiment.episode == -1 else experiment.episode
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    log_filename = os.path.join(model_dir, "model_log.txt")
    msg =  f"\n*********** checkpoint {episode} was loaded **************" 
    log_data(msg, log_filename)
    
    leo = LEO(config)
    optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)
    leo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mode = checkpoint['mode']
    total_val_loss = checkpoint['total_val_loss']
    kl_loss = checkpoint['kl_loss']

    stats = {
        "mode": mode,
        "episode": episode,
        "kl_loss": kl_loss,
        "total_val_loss": total_val_loss
        }

    return leo, optimizer, stats

