def forward_encoder(self, data, mode):
        """
        This function generates latent codes  from the input data pass it through relation network and
        computes kl_loss after sampling
        Args:
            data (tensor): input or tr_imgs shape ((num_classes * num_eg_per_class), channel, H, W)
            mode (str): overwrites the default mode "meta_train" with one among ("meta_train", "meta_val", "meta_test")
        Returns:
            latent_leaf (tensor) : shape ((num_classes * num_eg_per_class), latent_channels, H, W) leaf node where
            backpropagation of LEO ends.
            kl_loss (tensor_0shape): how much the distribution sampled from latent code deviates from normal distribution
        """
        self.mode = mode
        latent = self.encoder(data)

        relation_network_outputs = self.relation_network(latent)
        latent_dist_params = self.average_codes_per_class(relation_network_outputs)
        latents, kl_loss = self.possibly_sample(latent_dist_params)
        latent_leaf = latents.clone().detach() #to make latents the leaf node to perform backpropagation until latents
        latent_leaf.requires_grad_(True)
        return latent_leaf, kl_loss

    def forward_decoder(self, data, latents):
        """
        This function decodes the latent codes to get the segmentation weights that has same shape as input tensor
        and computes the leo cross entropy segmentation loss.
        Args:
            data (dict) : contains tr_imgs, tr_masks, val_imgs, val_masks
            latents (tensor) : shape ((num_classes * num_eg_per_class), latent_channels, H, W)
        Returns:
            loss (tensor_0shape): computed as crossentropyloss (groundtruth--> tr/val_imgs_mask, prediction--> einsum(tr/val_imgs, segmentation_weights))
            segmentation_weights (tensor) : shape(num_classes, num_eg_per_class, channels, H, W)
        """
        #removed kl divergence sampling from decoder
        segmentation_weights = self.decoder(latents)
        dim_list = list(segmentation_weights.size())
        segmentation_weights= segmentation_weights.permute(1, 2, 3, 0)
        segmentation_weights = segmentation_weights.view(dim_list[1], dim_list[2], dim_list[3], self.config.data_params.num_classes, -1 )
        segmentation_weights = segmentation_weights.permute(3, 4, 0, 1, 2)
        loss = self.calculate_inner_loss(data["tr_imgs_orig"], data["tr_masks"], segmentation_weights)
        return loss, segmentation_weights
