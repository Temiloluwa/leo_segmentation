
def test_saved_model(encoder, decoder, chosen_encoder, input_imgs, img_dims):
    oracle = forward_model(encoder, decoder, input_imgs)
    retrieved_encoder = chosen_encoder(img_dims)
    retrieved_encoder.load_weights("./data/pascal_voc/embedding_ckpt/encoder_weights")
    retrieved_decoder = Decoder()
    retrieved_decoder.load_weights("./data/pascal_voc/embedding_ckpt/decoder_weights")
    retrieved_outputs = forward_model(retrieved_encoder, retrieved_decoder, input_imgs)
    test_types = ["final output", "embeddings"]
    for _test, _oracle, _retrieved_output in zip(test_types, oracle, retrieved_outputs):
        print(f"{_test} testing, oracle shape: {_oracle.shape}, retr. shape: {_retrieved_output.shape}")
        try:
            np.testing.assert_allclose(_retrieved_output, _oracle, rtol=0.7, atol=1e-5)
            print("Reconstruction okay")
            diff = _retrieved_output - _oracle 
            print(f"max absolute diff: {np.max(np.absolute(diff))}, max relative diff: {np.nanmax(diff/_oracle)}")
        except AssertionError as e:
                print(f"Error occured in reconstructions\n{e}")


def test_saved_embeddings():
    emb_path_root = os.path.join(os.path.dirname(__file__), "data", "pascal_voc")
    data_path_root = os.path.join(os.path.dirname(__file__), "data", "grouped_by_classes")
    data_types = ["train", "val"]
    data_categories = ["images", "masks"]
    
    for data_type, data_category in itertools.product(data_types, data_categories):
        data_type_root = os.path.join(data_path_root, f"{data_type}", f"{data_category}")
        emb_type_root = os.path.join(emb_path_root, f"{data_type}", f"{data_category}")
        
        for selected_class in os.listdir(data_type_root):
            data_class_root = os.path.join(data_type_root, selected_class)
            emb_class_root = os.path.join(emb_type_root, selected_class)
            data = [i[:-4] for i in os.listdir(data_class_root)]
            emb = [i[:-4] for i in os.listdir(emb_class_root)]
            assert set(data) - set(emb) == set(), "not all files generated"
            print(f"class: {selected_class}, data_type: {data_type}, data_category: {data_category} is ok")


def save_embeddings(encoder, decoder, data_type, **kwargs):
    path_root = os.path.join(os.path.dirname(__file__), "data", "pascal_voc")
    for selected_class in kwargs[f"{data_type}_classes"]:
        images_save_path_data_type_root = os.path.join(path_root, f"{data_type}", "images", selected_class)
        masks_save_path_data_type_root = os.path.join(path_root, f"{data_type}", "masks", selected_class)
        class_one = SampleOneClass(selected_class, data_type, **kwargs)
        for j in range(len(class_one)):
            inp_img, target = class_one[j]
            target = np.squeeze(target)
            fn = class_one.file_names[j]
            output_embedding = np.squeeze(forward_model(encoder, decoder, inp_img)[1].numpy())
        
            if not os.path.exists(images_save_path_data_type_root):
                os.makedirs(images_save_path_data_type_root, exist_ok=True)
                os.makedirs(masks_save_path_data_type_root, exist_ok=True)

            img_file_path = os.path.join(images_save_path_data_type_root, fn)
            mask_file_path = os.path.join(masks_save_path_data_type_root, fn)
            save_npy(output_embedding, img_file_path)
            save_npy(target,  mask_file_path)
