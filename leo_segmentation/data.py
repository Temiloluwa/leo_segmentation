#contains data preprocessing functions
from .utils import load_data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import collections, random
import numpy as np

class MetaDataset(Dataset):
    def __init__(self, dataset, config, data_type):
        self._dataset = dataset
        self._config  = config
        self._data_type = data_type

    def __getitem__(self, idx):
       return self._index_data()

    def _index_data(self):
        # TO-DO return masks
        data_dict = load_data(self._config, self._dataset, self._data_type)
        image_class_mapping = collections.OrderedDict()
        all_classes = []
        image_embeddings = {}
        for i, fn in enumerate(data_dict["filenames"]):
            class_name, _ = fn.split("_")
            image_embeddings[fn] = data_dict["embeddings"][i]
            if class_name in list(image_class_mapping.keys()):
                image_class_mapping[class_name].append(fn)
                all_classes.append(class_name)
            else:
                image_class_mapping[class_name] = [fn]
                all_classes.append(class_name)

        total_files = sum([len(image_class_mapping[i]) for i in image_class_mapping])
        assert(len(data_dict["embeddings"]) == total_files == len(image_embeddings))
        return image_class_mapping, image_embeddings


class Datagenerator():
    def __init__(self, dataset, config, data_type):
        self.config = config["data_type"][data_type]
        self._all_class_images, self._image_embedding = MetaDataset(Dataset, config, data_type)
        self._data_loader = DataLoader(dataset, batch_size=None, shuffle=False,\
                            num_workers=0, collate_fn=self.collation_fn)

    @property
    def get_batch_data(self):
        return next(iter(self._data_loader))

    def collation_fn(self):
        for i in range(self.config["num_tasks"]):
            class_list = list(self._all_class_images.keys())
            tr_size = self.config["n_train_per_class"]
            val_size = self.config["n_val_per_class"]
            num_classes = self.config["num_classes"]
            sample_count = (tr_size + val_size)
            random.shuffle(class_list)
            shuffled_list = class_list[:num_classes]
            error_message = f"len(shuffled_list) {len(shuffled_list)} is not num_classes: {num_classes}"
            assert(len(shuffled_list) == num_classes, error_message)
            image_paths = []
            class_ids = []
            for class_id, class_name in enumerate(shuffled_list):
                all_images = self._all_class_images[class_name]
                all_images = np.random.choice(all_images, sample_count, replace=False)
                error_message = f"{len(all_images)} == {sample_count} failed"
                assert len(all_images) == sample_count, error_message
                image_paths.append(all_images)
                class_ids.append([[class_id]]*sample_count)
                #mask_array = np.array(class_ids, dtype=np.int32)
                path_array = np.array(image_paths)
                if i == 0:
                    embedding_array = np.array([[self._image_embedding[image_path]
                                            for image_path in class_paths]
                                            for class_paths in path_array])
                else:
                    embedding_array = np.concatenate(embedding_array, 
                                    np.array([[self._image_embedding[image_path]
                                                            for image_path in class_paths]
                                                            for class_paths in path_array]))
        return embedding_array



    
    
    
        
        

     


    



    

