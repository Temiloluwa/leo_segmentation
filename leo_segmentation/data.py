# contains data preprocessing functions
from utils import load_data, numpy_to_tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import collections, random
import numpy as np


class MetaDataset(Dataset):
    def __init__(self, dataset, config, data_type):
        self._config = config
        self._data_type = data_type
        self._dataset = self._index_data(dataset)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset

    def _index_data(self, dataset):
        """
        Converts the dataset dictionary to a mappping between
        classes in the data and the list of filenames and
        a mapping between filenames and both their embeddings and masks

        Args:
            dataset(str): name of dataset

        Returns:
            - image_class_mapping, image_mask_embeddings(tuple)
            - image_class_mapping(dictionary): keys are class, values are list of filenames in class
            - image_mask_embeddings(tuple): keys are filenames, values are a tuple of embeddings
                                     and masks for that image

        """
        data_dict = load_data(self._config, dataset, self._data_type)
        image_class_mapping = collections.OrderedDict()
        all_classes = []
        image_mask_embeddings = {}
        for i, fn in enumerate(data_dict["filenames"]):
            class_name, _ = fn.split("_")
            image_mask_embeddings[fn] = (data_dict["embeddings"][i], data_dict["masks"][i])
            if class_name in list(image_class_mapping.keys()):
                image_class_mapping[class_name].append(fn)
                all_classes.append(class_name)
            else:
                image_class_mapping[class_name] = [fn]
                all_classes.append(class_name)

        total_files = sum([len(image_class_mapping[i]) for i in image_class_mapping])
        error_msg = "Not all classes represented in the mapping"
        assert set(image_class_mapping.keys()) - set(all_classes) == set(), error_msg
        assert (len(data_dict["embeddings"]) == total_files == len(image_mask_embeddings))
        return image_class_mapping, image_mask_embeddings


class Datagenerator():
    """Data Generator class"""

    def __init__(self, dataset, config, data_type):
        self.config = config["data_type"][data_type]
        self.dataset = MetaDataset(dataset, config, data_type)
        self._data_loader = DataLoader(self.dataset, batch_size=None, shuffle=False, \
                                       num_workers=0, collate_fn=self.collation_fn)

    def get_batch_data(self):
        return next(iter(self._data_loader))

    def collation_fn(self, data):
        """
        Collation function for data loader to generate train and val batch data
        Args:
            data (tuple): contains _all_class_images dict and _image_mask_embeddings

        Returns:
            A tuple of pytorch tensors:
            DIMS = Image or Mask shape
            - tr_data: (batch_size, num_classes, tr_size, DIMS): training image embeddings
            - tr_masks: (batch_size, num_classes, tr_size, DIMS): training image masks
            - val_data: (batch_size, num_classes, val_size, DIMS): validation image embeddings
            - val_masks: (batch_size, num_classes, val_size, DIMS): validation image masks
        """
        _all_class_images, _image_mask_embeddings = data
        for i in range(self.config["num_tasks"]):
            class_list = list(_all_class_images.keys())
            tr_size = self.config["n_train_per_class"]
            val_size = self.config["n_val_per_class"]
            num_classes = self.config["num_classes"]
            sample_count = (tr_size + val_size)
            random.shuffle(class_list)
            shuffled_list = class_list[:num_classes]
            error_message = f"len(shuffled_list) {len(shuffled_list)} is not num_classes: {num_classes}"
            assert len(shuffled_list) == num_classes, error_message
            image_paths = []
            for _, class_name in enumerate(shuffled_list):
                all_images = _all_class_images[class_name]
                all_images = np.random.choice(all_images, sample_count, replace=False)
                error_message = f"{len(all_images)} == {sample_count} failed"
                assert len(all_images) == sample_count, error_message
                image_paths.append(all_images)

            path_array = np.array(image_paths)

            embedding_array = np.array([[_image_mask_embeddings[image_path][0]
                                         for image_path in class_paths]
                                        for class_paths in path_array])

            mask_array = np.array([[_image_mask_embeddings[image_path][1]
                                    for image_path in class_paths]
                                   for class_paths in path_array])

            if i == 0:
                batch_embeddings = np.empty((self.config["num_tasks"],) + embedding_array.shape)
                batch_masks = np.empty((self.config["num_tasks"],) + mask_array.shape)
            batch_embeddings[i] = embedding_array
            batch_masks[i] = mask_array

        tr_data = batch_embeddings[:, :, :tr_size, :, :]
        tr_data_masks = batch_masks[:, :, :tr_size, :, :]
        val_data = batch_embeddings[:, :, tr_size:, :, :]
        val_masks = batch_masks[:, :, tr_size:, :, :]
        return numpy_to_tensor(tr_data), numpy_to_tensor(tr_data_masks), \
               numpy_to_tensor(val_data), numpy_to_tensor(val_masks)


















