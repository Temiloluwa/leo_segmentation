import pytest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from leo_segmentation.data import PascalDatagenerator, GeneralDatagenerator, TrainingStats
from leo_segmentation.utils import load_config, update_config
from collections import Counter

@pytest.fixture
def general_dataloader_train():
    dataset = "pascal_5i"
    fold = 0
    config = load_config()
    update_config({'fold': fold})
    dataloader = PascalDatagenerator(dataset, data_type="meta_train")
    pytest.tr_paths = []
    pytest.val_paths = []
    num_of_sampling = 100
    for i in range(num_of_sampling):
        metadata = dataloader.get_batch_data()
        pytest.val_paths.extend(metadata[-1])
        pytest.tr_paths.extend(metadata[-2])
        if i%10 == 0:
            print(f"added {i+1} batches of paths")
    return 


def test_img_selection_frequency(general_dataloader_train):
    count_of_most_selected_img = Counter(pytest.tr_paths).most_common()[0][1]
    print(f"The most frequently sampled image was sampled\
            {count_of_most_selected_img} times")    
    assert(count_of_most_selected_img < 10 == True ,
        "Some images were sampled too frequently")