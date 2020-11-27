import os
import torch
import gc
import numpy as np
from torch import nn
from torch.distributions import Normal
from torch.nn import CrossEntropyLoss
from torchvision import models
from torch.nn import functional as F
# from .aspp import build_aspp
from leo_segmentation.utils import display_data_shape, get_named_dict, calc_iou_per_class, \
    log_data, load_config, list_to_tensor, numpy_to_tensor, tensor_to_numpy

config = load_config()
hyp = config.hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available()
                                  and config.use_gpu else "cpu")


class EncoderBlock(nn.Module):
    """ Encoder with pretrained backbone """

    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList(list(models.mobilenet_v2(pretrained=True)
                                         .features))
        self.in_channels = 0
        self.squeeze_conv_l3 = nn.Conv2d(in_channels=24, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=1)
        self.squeeze_conv_l6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=1)
        self.squeeze_conv_l13 = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=1)
        self.squeeze_conv_l17 = nn.Conv2d(in_channels=320, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, d_train=False, we=None):
        features = []
        cnt = 0
        we = [] if we is None else we            
        output_layers = [3, 6, 13, 17]  # 56, 28, 14, 7
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in output_layers:
                if d_train == True:
                    features.append(x)
                    squeeze_conv_layer = getattr(self, f"squeeze_conv_l{i}")
                    we.append(self.sigmoid(squeeze_conv_layer(x)))
                elif len(we) > 0:
                    x = torch.mul(x, we[cnt])
                    features.append(x)
                    cnt += 1
                else:
                    features.append(x)
        latents = x
        if d_train == True:
            return features, latents, we
        return features, latents


class RelationNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(RelationNetwork, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels, momentum=1, affine=True),
                  nn.ReLU(inplace=False),
                  nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(in_channels, momentum=1, affine=True),
                  nn.ReLU(inplace=False)]
        if dropout:
            layers.append(nn.Dropout(inplace=False))
        self.relation = nn.Sequential(*layers)

    def forward(self, concat_features):
        out = self.relation(concat_features)
        return out


def decoder_block(conv_in_size, conv_out_size):
    """ Sequentical group formimg a decoder block """
    layers = [

        nn.Conv2d(conv_in_size, conv_out_size,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Dropout(hyp.dropout_rate),
        nn.BatchNorm2d(conv_out_size),
        nn.Conv2d(conv_out_size, conv_out_size,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(conv_out_size, conv_out_size,
                           kernel_size=4, stride=2, padding=1)
    ]
    conv_block = nn.Sequential(*layers)
    return conv_block


class DecoderBlock(nn.Module):
    """
    Leo Decoder
    """

    def __init__(self, skip_features, latents):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = decoder_block(latents.shape[1],
                                   hyp.base_num_covs * 4)
        self.conv2 = decoder_block(skip_features[-1].shape[1] + hyp.base_num_covs * 4,
                                   hyp.base_num_covs * 3)
        self.conv3 = decoder_block(skip_features[-2].shape[1] + hyp.base_num_covs * 3,
                                   hyp.base_num_covs * 2)
        self.conv4 = decoder_block(skip_features[-3].shape[1] + hyp.base_num_covs * 2,
                                   hyp.base_num_covs * 1)
        self.up_final = nn.ConvTranspose2d(skip_features[-4].shape[1] + hyp.base_num_covs * 1,
                                           hyp.base_num_covs, kernel_size=4, stride=2, padding=1)

        self.squeeze_conv_latent = nn.Conv2d(in_channels=1280, out_channels=1, kernel_size=(1, 1), padding=(0, 0),
                                             stride=1)
        self.squeeze_conv1_out = nn.Conv2d(in_channels=352, out_channels=1, kernel_size=(1, 1), padding=(0, 0),
                                           stride=1)  # 32
        self.squeeze_conv2_out = nn.Conv2d(in_channels=120, out_channels=1, kernel_size=(1, 1), padding=(0, 0),
                                           stride=1)  # 24
        self.squeeze_conv3_out = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=(1, 1), padding=(0, 0),
                                           stride=1)  # 16
        self.squeeze_conv4_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), padding=(0, 0),
                                           stride=1)  # 8
        self.squeeze_final_out = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(1, 1), padding=(0, 0), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_features, latents, d_train=False, wd=None):
        wd = [] if wd is None else wd
        if d_train == True:
            wd.append(self.sigmoid(self.squeeze_conv_latent(latents)))
        elif len(wd) > 0:
            if latents.shape[0] == 1:
                latents_ = latents.clone().repeat(5, 1, 1, 1)
                latents_ = torch.mul(latents_, wd[0])
            else:
                latents = torch.mul(latents, wd[0])

        if latents.shape[0] == 1:
            o = self.conv1(latents_)
        else:
            o = self.conv1(latents)
        if latents.shape[0] == 1:
            skip_f = self.upsample(skip_features[-1]).clone().repeat(5, 1, 1, 1)
            o = torch.cat([o, skip_f], dim=1)
        else:
            o = torch.cat([o, self.upsample(skip_features[-1])], dim=1)
        if d_train == True:
            # self.in_channels = o.shape[1]
            wd.append(self.sigmoid(self.squeeze_conv1_out(o)))
        elif len(wd) > 0:
            o = torch.mul(o, wd[1])
        o = self.conv2(o)
        # print(o.shape)
        if latents.shape[0] == 1:
            skip_f = self.upsample(skip_features[-2]).clone().repeat(5, 1, 1, 1)
            o = torch.cat([o, skip_f], dim=1)
        else:
            o = torch.cat([o, self.upsample(skip_features[-2])], dim=1)
        if d_train == True:
            # self.in_channels = o.shape[1]
            wd.append(self.sigmoid(self.squeeze_conv2_out(o)))
        elif len(wd) > 0:
            o = torch.mul(o, wd[2])
        o = self.conv3(o)
        # print(o.shape)
        if latents.shape[0] == 1:
            skip_f = self.upsample(skip_features[-3]).clone().repeat(5, 1, 1, 1)
            o = torch.cat([o, skip_f], dim=1)
        else:
            o = torch.cat([o, self.upsample(skip_features[-3])], dim=1)
        if d_train == True:
            # self.in_channels = o.shape[1]
            wd.append(self.sigmoid(self.squeeze_conv3_out(o)))
        elif len(wd) > 0:
            o = torch.mul(o, wd[3])
        o = self.conv4(o)
        # print(o.shape)
        if latents.shape[0] == 1:
            skip_f = self.upsample(skip_features[-4]).clone().repeat(5, 1, 1, 1)
            o = torch.cat([o, skip_f], dim=1)
        else:
            o = torch.cat([o, self.upsample(skip_features[-4])], dim=1)
        if d_train == True:
            # self.in_channels = o.shape[1]
            wd.append(self.sigmoid(self.squeeze_conv4_out(o)))
        elif len(wd) > 0:
            o = torch.mul(o, wd[4])
        o = self.up_final(o)
        if latents.shape[0] == 1:
            o = torch.mean(o, dim=0).unsqueeze(0)
            #print(o.shape)
        if d_train == True:
            # self.in_channels = o.shape[1]
            wd.append(self.sigmoid(self.squeeze_final_out(o)))
        elif len(wd) > 0:
            latents = torch.mul(o, wd[5])
        # print(o.shape)
        # print('---------------------------')
        if d_train == True:
            return o, wd
        return o


class LEO(nn.Module):
    """
    contains functions to perform latent embedding optimization
    """

    def __init__(self, mode="meta_train"):
        super(LEO, self).__init__()
        self.mode = mode
        self.encoder = EncoderBlock()
        # self.aspp = build_aspp('mobilenet', 16, nn.BatchNorm2d)
        # self.RelationNetwork = RelationNetwork(512, 256)
        seg_network = nn.Conv2d(hyp.base_num_covs + 3, 2, kernel_size=3, stride=1, padding=1)
        self.seg_weight = seg_network.weight.detach().to(device)
        self.seg_weight.requires_grad = True
        self.loss_fn = CrossEntropyLoss()
        self.optimizer_seg_network = torch.optim.Adam(
            [self.seg_weight], lr=hyp.outer_loop_lr)

    def freeze_encoder(self):
        """ Freeze encoder weights """
        for param in self.encoder.parameters():
            param.requires_grad = False
        # for param in self.RelationNetwork.parameters():
        #    param.requires_grad = False
        # for param in self.aspp.parameters():
        #    param.requires_grad = False

    def unfreeze_encoder(self):
        """ UnFreeze encoder weights """
        for param in self.encoder.parameters():
            param.requires_grad = True
        # for param in self.RelationNetwork.parameters():
        #    param.requires_grad = True
        # for param in self.aspp.parameters():
        #    param.requires_grad = True

    def forward_encoder(self, x, mode, d_train=False, we=None):
        """ Performs forward pass through the encoder """
        if d_train == True:
            skip_features, latents, we = self.encoder(x, d_train=d_train)
        else:
            skip_features, latents = self.encoder(x, we)
        # aspp_latents = self.aspp(latents)
        # relation_network_outputs, total_num_examples = self.relation_network(aspp_latents)
        # print('relation_network_outputs', relation_network_outputs.shape)
        # latent_dist_params = self.average_codes_per_class(relation_network_outputs, total_num_examples)
        if not latents.requires_grad:
            latents.requires_grad = True
        if d_train == True:
            return skip_features, latents, we
        return skip_features, latents

    def forward_decoder(self, skip_features, latents, d_train=False, wd=None):
        """Performs forward pass through the decoder"""
        if d_train == True:
            output = self.decoder(skip_features, latents, d_train=d_train)
        elif wd != None:
            output = self.decoder(skip_features, latents, wd=wd)
        else:
            output = self.decoder(skip_features, latents)
        return output

    def forward_segnetwork(self, decoder_out, x, weight):
        """  Receives features from the decoder
             Concats the features with input image
             Convolution layer acts on the concatenated input
            Args:
                decoder_out (torch.Tensor): decoder output features
                x (torch.Tensor): input images
                weight(tf.tensor): kernels for the segmentation network
            Returns:
                pred(tf.tensor): predicted logits
        """
        o = torch.cat([decoder_out, x], dim=1)
        pred = F.conv2d(o, weight, padding=1)
        return pred

    def forward(self, x, d_train=False, latents=None, weight=None, we=None, wd=None):
        """ Performs a forward pass through the entire network
            - The Autoencoder generates features using the inputs
            - Features are concatenated with the inputs
            - The concatenated features are segmented
            Args:
                x (torch.Tensor): input image
                latents(torch.Tensor): output of the bottleneck
            Returns:
                latents(torch.Tensor): output of the bottleneck
                features(torch.Tensor): output of the decoder
                pred(torch.Tensor): predicted logits
                weight(torch.Tensor): segmentation weights
        """

        if latents is None:
            if d_train == True:
                skip_features, latents, we = self.forward_encoder(x, self.mode, d_train=d_train)
            else:
                skip_features, latents = self.forward_encoder(x, self.mode, we=we)
            self.skip_features = skip_features
        else:
            skip_features = self.skip_features

        if weight is not None:
            seg_weight = weight
        else:
            seg_weight = self.seg_weight

        if d_train == True:
            features, wd = self.forward_decoder(skip_features, latents, d_train=d_train)
        elif wd is not None:
            features = self.forward_decoder(skip_features, latents, wd=wd)
        else:
            features = self.forward_decoder(skip_features, latents)
        pred = self.forward_segnetwork(features, x, seg_weight)

        if d_train == True and we is not None:
            return latents, features, pred, we
        elif d_train == True and wd is not None:
            return latents, features, pred, wd

        return latents, features, pred

    def leo_inner_loop(self, x, y):
        """
        Performs innerloop optimization
            - It updates the latents taking gradients wrt the training loss
            - It generates better features after the latents are updated
            Args:
                x(torch.Tensor): input training image
                y(torch.Tensor): input training mask
            Returns:
                seg_weight_grad(torch.Tensor): The last gradient of the
                    training loss wrt to the segmenation weights
                features(torch.Tensor): The last generated features
        """
        inner_lr = hyp.inner_loop_lr
        latents, _, pred, w_e = self.forward(x, d_train=True)
        tr_loss = self.loss_fn(pred, y.long())
        for _ in range(hyp.num_adaptation_steps):
            latents_grad = torch.autograd.grad(tr_loss, [latents], retain_graph=True, create_graph=False)[0]
            with torch.no_grad():
                latents -= inner_lr * latents_grad
            latents, features, pred, w_d = self.forward(x, latents=latents, d_train=True)
            tr_loss = self.loss_fn(pred, y.long())
        seg_weight_grad = torch.autograd.grad(tr_loss, [self.seg_weight], retain_graph=True, create_graph=False)[0]

        return seg_weight_grad, features, w_e, w_d

    def finetuning_inner_loop(self, data_dict, tr_features, seg_weight_grad, transformers, mode, we=None, wd=None):
        """ Finetunes the segmenation weights/kernels by performing MAML
            Args:
                data_dict (dict): contains tr_imgs, tr_masks, val_imgs, val_masks
                tr_features (torch.Tensor): tensor containing decoder features
                segmentation_grad (torch.Tensor): gradients of the training
                                                loss to the segmenation weights
            Returns:
                val_loss (torch.Tensor): validation loss
                seg_weight_grad (torch.Tensor): gradient of validation loss
                                                wrt segmentation weights
                decoder_grads (torch.Tensor): gradient of validation loss
                                                wrt decoder weights
                transformers(tuple): tuple of image and mask transformers
                weight (torch.Tensor): segmentation weights
        """
        img_transformer, mask_transformer = transformers
        weight = self.seg_weight - hyp.finetuning_lr * seg_weight_grad
        for _ in range(hyp.num_finetuning_steps - 1):
            pred = self.forward_segnetwork(tr_features, data_dict.tr_imgs, weight)
            tr_loss = self.loss_fn(pred, data_dict.tr_masks.long())
            seg_weight_grad = torch.autograd.grad(tr_loss, [weight], retain_graph=True, create_graph=False)[0]
            weight -= hyp.finetuning_lr * seg_weight_grad
    
        if mode == "meta_train":
            _, _, prediction = self.forward(data_dict.val_imgs, weight=weight, we=we, wd=wd)
            val_loss = self.loss_fn(prediction, data_dict.val_masks.long())
            grad_output = torch.autograd.grad(val_loss,
                                              [weight] + list(self.decoder.parameters()), retain_graph=True,
                                              create_graph=False, allow_unused=True)
            seg_weight_grad, decoder_grads = grad_output[0], grad_output[1:]
            mean_iou = calc_iou_per_class(prediction, data_dict.val_masks)
            return val_loss, seg_weight_grad, decoder_grads, mean_iou, weight
        else:
            with torch.no_grad():
                mean_ious = []
                val_losses = []
                val_img_paths = data_dict.val_imgs
                val_mask_paths = data_dict.val_masks
                for _img_path, _mask_path in zip(val_img_paths, val_mask_paths):
                    input_img = numpy_to_tensor(list_to_tensor(_img_path, img_transformer))
                    input_mask = numpy_to_tensor(list_to_tensor(_mask_path, mask_transformer))
                    _, _, prediction = self.forward(input_img, weight=weight, we=we, wd=wd)
                    val_loss = self.loss_fn(prediction, input_mask.long()).item()
                    mean_iou = calc_iou_per_class(prediction, input_mask)
                    mean_ious.append(mean_iou)
                    val_losses.append(val_loss)
                mean_iou = np.mean(mean_ious)
                val_loss = np.mean(val_losses)
            return val_loss, None, None, mean_iou, weight


def compute_loss(leo, metadata, train_stats, transformers, mode="meta_train"):
    """ Performs meta optimization across tasks
        returns the meta validation loss across tasks
        Args:
            metadata(dict): dictionary containing training data
            train_stats(object): object that stores training statistics
            transformers(tuple): tuple of image and mask transformers
            mode(str): meta_train, meta_val or meta_test
        Returns:
            total_val_loss(float32): meta-validation loss
            train_stats(object): object that stores training statistics
    """
    num_tasks = len(metadata[0])
    # initialize decoder on the first episode
    if train_stats.episode == 1:
        data_dict = get_named_dict(metadata, 0)
        skip_features, latents = leo.forward_encoder(data_dict.tr_imgs, mode)
        leo.decoder = DecoderBlock(skip_features, latents).to(device)
        leo.optimizer_decoder = torch.optim.Adam(
            leo.decoder.parameters(), lr=hyp.outer_loop_lr)

    if train_stats.episode % config.display_stats_interval == 1:
        display_data_shape(metadata)

    classes = metadata[4]
    total_val_loss = []
    mean_iou_dict = {}
    total_grads = None
    leo.forward_encoder
    for batch in range(num_tasks):
        data_dict = get_named_dict(metadata, batch)
        seg_weight_grad, features, we, wd = leo.leo_inner_loop(data_dict.tr_imgs, data_dict.tr_masks)
        val_loss, seg_weight_grad, decoder_grads, mean_iou, _ = \
            leo.finetuning_inner_loop(data_dict, features, seg_weight_grad,
                                      transformers, mode, we=we, wd=wd)
        if mode == "meta_train":
            decoder_grads_ = []
            for grad in decoder_grads:
                if grad is not None:
                    decoder_grads_.append(grad / num_tasks)
                else:
                    decoder_grads_.append(0)
            # decoder_grads = [grad / num_tasks for grad in decoder_grads]
            if total_grads is None:
                total_grads = decoder_grads_
                seg_weight_grad = seg_weight_grad / num_tasks
            else:
                total_grads = [total_grads[i] + decoder_grads_[i] \
                               for i in range(len(decoder_grads_))]
                seg_weight_grad += seg_weight_grad / num_tasks
        mean_iou_dict[classes[batch]] = mean_iou
        total_val_loss.append(val_loss)
    
    if mode == "meta_train":
        leo.optimizer_decoder.zero_grad()
        leo.optimizer_seg_network.zero_grad()

        for i, params in enumerate(leo.decoder.parameters()):
            try:
                params.grad = total_grads[i]
            except:
                params.grad = None
        leo.seg_weight.grad = seg_weight_grad
        leo.optimizer_decoder.step()
        leo.optimizer_seg_network.step()
    total_val_loss = float(sum(total_val_loss) / len(total_val_loss))
    stats_data = {
        "mode": mode,
        "total_val_loss": total_val_loss,
        "mean_iou_dict": mean_iou_dict
    }
    train_stats.update_stats(**stats_data)
    return total_val_loss, train_stats


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
        'total_val_loss': stats.total_val_loss,
        'mean_iou_dict': stats.mean_iou_dict,

    }

    experiment = config.experiment
    model_root = os.path.join(os.path.dirname(__file__), config.data_path, "models")
    model_dir = os.path.join(model_root, "experiment_{}" \
                             .format(experiment.number))
    if not os.path.exists(os.path.join(model_dir, "config.txt")):
        torch.save(config, os.path.join(model_dir, "config.txt"))
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


def load_model():
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
    model_dir = os.path.join(os.path.dirname(__file__), config.data_path, "models", "experiment_{}" \
                             .format(experiment.number))

    checkpoints = os.listdir(model_dir)
    checkpoints = [i for i in checkpoints if os.path.splitext(i)[-1] == ".tar"]
    max_cp = max([int(cp.split(".")[0].split("_")[1]) for cp in checkpoints])
    # if experiment.episode == -1, load latest checkpoint
    episode = max_cp if experiment.episode == -1 else experiment.episode
    checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    log_filename = os.path.join(model_dir, "model_log.txt")
    msg = f"\n*********** checkpoint {episode} was loaded **************"
    log_data(msg, log_filename)

    leo = LEO(config).to(device)
    optimizer = torch.optim.Adam(leo.parameters(), lr=config.hyperparameters.outer_loop_lr)
    leo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    mode = checkpoint['mode']
    total_val_loss = checkpoint['total_val_loss']
    # kl_loss = checkpoint['kl_loss']
    mean_iou_dict = checkpoint['mean_iou_dict']

    stats = {
        "mode": mode,
        "episode": episode,
        "total_val_loss": total_val_loss,
        "mean_iou_dict": mean_iou_dict
    }

    return leo, optimizer, stats
