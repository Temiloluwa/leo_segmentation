import pytest
import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from collections import Counter
from easydict import EasyDict as edict
from pascal_embeddings_data import DataGenerator, CLASSES, FewShotDataGenerator

def read_file(filename):
    with open(filename, "r") as f:
        temp_list = f.readlines()
        temp_list = [i.strip("\n").split("__") for i in temp_list]
        temp_list = [(i[0], int(i[1])) for i in temp_list]
    return temp_list

@pytest.fixture(scope="module")
def hyper_params():
    pytest.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),\
        "leo_segmentation", "data", "pascal_5i")
    pytest.classes = CLASSES
    pytest.classes_dict = {i:pytest.classes[i] for i in range(len(pytest.classes))}
    pytest.config = edict({
                    "classes": CLASSES,
                    "batch_size": 32,
                    "num_channels": 3,
                    "img_height": 384,
                    "img_width": 512,
                    "n_query": 1,
                    "n_support": 5,
                    "train_batch_size": 15,
                    "val_batch_size": 750})

"""
def test_num_images_per_class(hyper_params):
    CLASSES_DICT = pytest.classes_dict
       
    for fold in range(4):
        print(f"fold {fold}")
        data_generator = DataGenerator(fold=fold)
        train_class_names = []
        train_class_names_from_file = []
        train_folds = [0, 1, 2, 3]
        train_folds.remove(fold)
        for (batch_imgs, batch_masks, batch_classnames) in tqdm(data_generator):
            train_class_names.extend(batch_classnames)
        
        for fold_number in train_folds:
            textfile_path = os.path.join(pytest.data_dir, 'Binary_map_aug', 'train', f'split{fold_number}_train.txt')
            train_files_list = read_file(textfile_path)
            for fname, class_idx in train_files_list:
                train_class_names_from_file.append(CLASSES_DICT[class_idx-1])

        assert len(train_class_names) == len(train_class_names_from_file), \
                    f"Fold:{fold}, Not all train images generated"
        assert dict(Counter(train_class_names)) == dict(Counter(train_class_names_from_file)),\
                     f"Fold:{fold}, propotion per class in train wrong"

        data_generator.setmode("val")
        val_class_names = []
        val_class_names_from_file = []
        for (batch_imgs, batch_masks, batch_classnames) in tqdm(data_generator):
            val_class_names.extend(batch_classnames)
        textfile_path = os.path.join(pytest.data_dir, 'Binary_map_aug', 'val', f'split{fold}_val.txt')
        val_files_list = read_file(textfile_path)
        for fname, class_idx in val_files_list:
            val_class_names_from_file.append(CLASSES_DICT[class_idx-1])
        
        assert len(val_class_names) == len(val_class_names_from_file), \
                    "Fold:{fold}, Not all val images generated"
        assert dict(Counter(val_class_names)) == dict(Counter(val_class_names_from_file)),\
                     "Fold:{fold}, propotion per class in val wrong"
"""
"""
def test_train_few_shot_generator(hyper_params):
    for fold in range(4):
        train_datagenerator = FewShotDataGenerator(fold, config=pytest.config)
        num_channels = pytest.config.num_channels
        img_height = pytest.config.img_height
        img_width = pytest.config.img_width
        n_query = pytest.config.n_query
        n_support = pytest.config.n_support
        train_batch_size = pytest.config.train_batch_size
        val_batch_size = pytest.config.val_batch_size

        for i in range(10):
            batch_data = train_datagenerator.get_batch_data()
            assert batch_data[0].shape == (train_batch_size, n_support, img_height, img_width, num_channels)
            assert batch_data[1].shape == (train_batch_size, n_query, img_height, img_width, num_channels)
            assert batch_data[2].shape == (train_batch_size, n_support, img_height, img_width)
            assert batch_data[3].shape == (train_batch_size, n_query, img_height, img_width)
            for class_batch in batch_data[4] + batch_data[5]:
                class_name, fn_name = class_batch.split("-")
                assert fn_name in train_datagenerator.train_path_map[class_name]
"""

def test_val_few_shot_generator(hyper_params):
    for fold in range(4):
        print(f"processing fold {fold}")
        val_datagenerator = FewShotDataGenerator(fold, mode="val", config=pytest.config)
        num_channels = pytest.config.num_channels
        img_height = pytest.config.img_height
        img_width = pytest.config.img_width
        n_query = pytest.config.n_query
        n_support = pytest.config.n_support
        val_batch_size = pytest.config.val_batch_size
        query_fnames = []

        for i, (batch_data) in enumerate(val_datagenerator.get_batch_data()):
            assert batch_data[0].shape == (n_support, img_height, img_width, num_channels)
            assert batch_data[1].shape == (n_query, img_height, img_width, num_channels)
            assert batch_data[2].shape == (n_support, img_height, img_width)
            assert batch_data[3].shape == (n_query, img_height, img_width)
            for class_batch in batch_data[4] + batch_data[5]:
                class_name, fn_name = class_batch.split("-")
                assert fn_name in val_datagenerator.val_path_map[class_name]
            
            query_fnames.extend(batch_data[5])

        assert i == val_batch_size - 1
        assert len(set(query_fnames)) == len(set(val_datagenerator.val_paths))
            