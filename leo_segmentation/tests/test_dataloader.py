import unittest
from ..data import Datagenerator
from ..utils import load_config
from collections import Counter

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = "pascal_voc_raw"
        self.config = load_config()
        self.dataloader = Datagenerator(self.config, self.dataset, data_type="meta_train")
        self.tr_paths = []
        self.val_paths = []
        self.num_of_sampling = 100
        for i in range(self.num_of_sampling):
            metadata = self.dataloader.get_batch_data()
            self.val_paths.extend(metadata[-1])
            self.tr_paths.extend(metadata[-2])
            if i%10 == 0:
                print(f"added {i+1} batches of paths")
    
    def test_images_and_masks_match(self):
        num_train = self.config.data_params.n_train_per_class.meta_train
        num_val = self.config.data_params.n_val_per_class.meta_train
        imgs_paths = []
        masks_paths = []
        for i in range(self.num_of_sampling):
            imgs_paths.extend(self.tr_paths[i*num_train:(i + 1)*num_train])
            imgs_paths.extend(self.val_paths[(i + 1)*num_train: (i + 2)*num_train])
            masks_paths.extend(self.tr_paths[i*num_val:(i + 1)*num_val])
            masks_paths.extend(self.val_paths[(i + 1)*num_val:(i + 2)*num_val])

        imgs_paths_to_masks = [i.replace("images", "masks") for i in imgs_paths]
        self.assertEqual(set(imgs_paths_to_masks) - set(masks_paths), set(), \
                "masks and images don't match")

    def test_img_selection_frequency(self):
        count_of_most_selected_img = Counter(self.tr_paths).most_common()[0][1]
        print(f"The most frequently sampled image was sampled\
             {count_of_most_selected_img} times")    
        self.assertTrue(count_of_most_selected_img < 10 ,
            "Some images were sampled too frequently")
  