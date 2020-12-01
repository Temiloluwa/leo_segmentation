import pytest
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from leo_segmentation.data import PascalDatagenerator, GeneralDatagenerator, TrainingStats
from leo_segmentation.model import LEO, load_model, save_model, compute_loss
from leo_segmentation.utils import load_config, update_config, get_named_dict, tensor_to_numpy
from collections import Counter

@pytest.fixture
def train_model():
    dataset = "pascal_5i"
    fold = 0
    pytest.config = load_config()
    pytest.device = torch.device("cuda:0" if torch.cuda.is_available()
                                      and pytest.config.use_gpu else "cpu")
    update_config({'fold': fold})
    dataloader = PascalDatagenerator(dataset, data_type="meta_train")
    leo = LEO().to(pytest.device)
    train_stats = TrainingStats()
    train_stats.set_episode(1)
    train_stats.set_mode("meta_train")
    leo.mode = 'meta_train'
    metadata = dataloader.get_batch_data()
    transformers = (dataloader.transform_image, dataloader.transform_mask)
    _, train_stats = compute_loss(leo, metadata, train_stats, transformers)
    pytest.dataloader = PascalDatagenerator(dataset, data_type="meta_val")
    train_stats.set_mode("meta_val")
    pytest.metadata = pytest.dataloader.get_batch_data()
    leo.mode = "meta_val"
    _, train_stats = compute_loss(leo, pytest.metadata, train_stats,
                                    transformers, mode="meta_val")
    pytest.leo = leo
    pytest.train_stats = train_stats
    return 


def test_dataloading(train_model, allclose):
    """
    make sure you set a new experiment number
    and low base_num_covs e.g 8
    """
    leo = pytest.leo
    train_stats = pytest.train_stats
    config = pytest.config
    metadata = pytest.metadata
    data_dict = get_named_dict(metadata, 0)
    save_model(leo, config, train_stats)
    loaded_leo, _  = load_model(pytest.device, data_dict)
    
    for (name_leo, param_leo), (name_loaded_leo, param_loaded_leo) \
            in zip(leo.named_parameters(), loaded_leo.named_parameters()):
        param_leo = tensor_to_numpy(param_leo)
        param_loaded_leo = tensor_to_numpy(param_loaded_leo)
        print(f"Testing name {name_leo}")
        assert name_leo == name_loaded_leo
        assert allclose(param_leo, param_loaded_leo, atol=1e-7)
    print("Testing name segweight")
    assert allclose(tensor_to_numpy(leo.seg_weight),\
             tensor_to_numpy(loaded_leo.seg_weight), atol=1e-7)
