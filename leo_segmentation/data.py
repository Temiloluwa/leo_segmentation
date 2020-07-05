#contains data preprocessing functions
from utils import load_data
import collections


class DataLoader():
    def __init__(self, dataset, config, data_type):
        self._dataset = dataset
        self._config  = config
        self._data_dict = load_data(config, dataset, data_type)

    def _index_data(self):
        self._image_class_mapping = collections.OrderedDict()
        self._all_classes = collections.OrderedDict()
        filenames = self._data_dict["filenames"]
        for fn in filenames:
            class_name, _ = fn.split("-")
            if class_name in list(self._image_class_mapping.keys()):
                self._image_class_mapping[class_name].append(fn)
            else:
                self._image_class_mapping[class_name] = [fn]

        total_files = sum([len(self._image_class_mapping[i]) for i in self._image_class_mapping])
        assert(len(self._data_dict["embeddings"]) == total_files)

    def _get_task_data(self):
        outerloop_config = self._config["outerloop"]
        num_classes = outerloop_config["num_classes"]
        num_tr_examples_per_class = outerloop_config["num_tr_examples_per_class"]
        num_val_examples_per_class = outerloop_config["num_val_examples_per_class"]


    



    

