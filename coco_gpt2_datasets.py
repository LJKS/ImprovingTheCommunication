import tensorflow as tf
import transformers as hf
import tensorflow_datasets as tfds
import time
def undict_coco(ds):
    ds = ds.map(
        lambda elem: (
            elem["captions"],
            elem["image"],
            elem["image/id"],
            elem["objects"],
        )
    )
    return ds


def create_features_coco(ds, model):
    ds = undict_coco(ds).map(
        lambda captions, image, image_id, objects: (
            captions,
            image,
            tf.keras.applications.resnet50.preprocess_input(
                tf.image.resize(image, size=(224, 224))
            ),
            image_id,
            objects,
        )
    )
    ds = ds.padded_batch(32).prefetch(20)
    ds = ds.map(
        lambda captions, image, res50_img, image_id, objects: (
            image,
            model(res50_img),
            res50_img,
            captions,
            image_id,
            objects,
        )
    )
    ds = ds.unbatch()
    return ds


def ms_coco_res50_features(num=None, path="data/coco_full_features", print_bool=False):
    print_bool = False
    path_train = f"{path}_train_num_{num}"
    path_test = f"{path}_test_num_{num}"

    coco_annotated_train, coco_annotated_test, coco_annotated_val = tfds.load(
        "coco_captions", split=["train", "test", "val"], data_dir="data/tfdata"
    )
    if num != None:
        coco_annotated_train = coco_annotated_train.take(num)
        coco_annotated_test = coco_annotated_test.take(num)

    resnet_50 = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # do not cache these if you dont want to spend 150 GB hard drive space
    coco_annotated_train = create_features_coco(
        coco_annotated_train, resnet_50
    )  # .cache(filename=path_train)
    coco_annotated_test = create_features_coco(
        coco_annotated_test, resnet_50
    )  # .cache(filename=path_test)
    coco_annotated_val = create_features_coco(coco_annotated_val, resnet_50)

    return coco_annotated_train, coco_annotated_test, coco_annotated_val

def load_gpt2_TF():
    model = hf.TFGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = hf.GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def resnetfeature_gpt2tokencaption_stringid_cocodatasets(load_from_file=False, save_to_files=False, saved_element_spec=None):
    #load_from_file is the path to load the datasets if they are to be loaded from file, else False
    if load_from_file:
        train = tf.data.Dataset.load(path=load_from_file + "_train", element_spec=saved_element_spec)
        test = tf.data.Dataset.load(path=load_from_file + "_test", element_spec=saved_element_spec)
        val = tf.data.Dataset.load(path=load_from_file + "_val", element_spec=saved_element_spec)

        return train, test, val
    else:
        dstrain, dstest, dsval = ms_coco_res50_features()
        def extract_features(ds):
            ds = ds.map(lambda image, feature, res_img, captions, image_id, objects: (feature, captions, image_id, tf.shape(captions['text'])[0]))
            ds = ds.map(lambda features, captions, image_id, num_captions: (tf.repeat(tf.expand_dims(features, axis=0), repeats=num_captions, axis=0), captions["text"], (tf.repeat(tf.expand_dims(image_id, axis=0), repeats=num_captions, axis=0))))
            ds = ds.unbatch()
            return ds.prefetch(20)

        model, tokenizer = load_gpt2_TF()

        def coco_generator(ds, tokenizer):
            for elem in ds.apply(extract_features):
                feature, caption, image_id = elem
                """
                if caption.shape != (5,):
                    print('caption',caption.shape)
                if feature.shape != (5,2048):
                    print('feature',feature.shape)
                if image_id.shape != (5,):
                    print('image_id',image_id.shape)
                """
                caption = caption.numpy().decode("utf-8")
                caption_idxs = tokenizer.encode(caption, add_special_tokens=True, return_tensors="tf")
                #print(caption_ids)
                #decode the caption_ids to check if the encoding is correct
                resulting_string = tokenizer.decode(tf.squeeze(caption_idxs))
                #print(resulting_string, type(resulting_string))
                yield feature, tf.squeeze(caption_idxs), resulting_string, image_id

        coco_dataset_train = tf.data.Dataset.from_generator(lambda: coco_generator(dstrain, tokenizer), output_signature=(tf.TensorSpec(shape=(2048,), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))
        coco_dataset_test = tf.data.Dataset.from_generator(lambda: coco_generator(dstest, tokenizer), output_signature=(tf.TensorSpec(shape=(2048,), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))
        coco_dataset_val = tf.data.Dataset.from_generator(lambda: coco_generator(dsval, tokenizer), output_signature=(tf.TensorSpec(shape=(2048,), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32)))

        if save_to_files:
            coco_dataset_train = coco_dataset_train.save(path=save_to_files + "_train")
            coco_dataset_test = coco_dataset_test.save(path=save_to_files + "_test")
            coco_dataset_val = coco_dataset_val.save(path=save_to_files + "_val")
        return coco_dataset_train, coco_dataset_test, coco_dataset_val

def get_resnetfeature_gpt2tokencaption_stringid_cocodatasets_typespec():
    return (tf.TensorSpec(shape=(2048,), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32))

def get_resnetfeature_gpt2tokencaption_stringid_cocodatasets_default_path():
    return 'data/gpt2_coco_dataset'


def load_default_resnetfeature_gpt2tokencaption_stringid_cocodatasets():
    train, test, val = resnetfeature_gpt2tokencaption_stringid_cocodatasets(load_from_file=get_resnetfeature_gpt2tokencaption_stringid_cocodatasets_default_path(), saved_element_spec=get_resnetfeature_gpt2tokencaption_stringid_cocodatasets_typespec())
    return train, test, val

def resnetfeature_gpt2tokencaption_gpt2encodings_stringid_cocodatasets(load_underlying_default=True, load_from_file=False, saved_element_spec=None, save_to_files=False, gptbatchsize=256, caption_token_padding=50256):
    """
    :param load_underlying_default:
    :param load_from_file:
    :param saved_element_spec:
    :param save_to_files:
    :param gptbatchsize:
    :param caption_token_padding:
    :return: tf.data.Dataset of (img_feature, caption_tokens, caption_string, id, gpt2_hidden_state, mask)
    """
    if load_from_file:
        assert saved_element_spec is not None
        train = tf.data.Dataset.load(path=load_from_file + "/train", element_spec=saved_element_spec)
        test = tf.data.Dataset.load(path=load_from_file + "/test", element_spec=saved_element_spec)
        val = tf.data.Dataset.load(path=load_from_file + "/val", element_spec=saved_element_spec)
        return train, test, val

    if load_underlying_default:
        train, test, val = load_default_resnetfeature_gpt2tokencaption_stringid_cocodatasets()
    else:
        train, test, val = resnetfeature_gpt2tokencaption_stringid_cocodatasets()

    model, tokenizer = load_gpt2_TF()
    def gpt2encodings_generator(ds):
        ds_gen = ds.map(lambda f, c, s, i: (f, c, s, i, tf.shape(c)[0]))
        ds_gen = ds_gen.padded_batch(gptbatchsize, padding_values=(None, caption_token_padding, None, 0, -1)).prefetch(20)
        for elem in ds_gen:
            features, caption_idxs, captions_strings, image_ids, caption_len = elem
            outputs = model(caption_idxs, return_dict=True, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            yield features, caption_idxs, captions_strings, image_ids, caption_len, last_hidden_states

    batched_output_signature = (tf.TensorSpec(shape=(None,2048), dtype=tf.float32), tf.TensorSpec(shape=(None,None), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.string), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None, None, 768), dtype=tf.float32))
    coco_dataset_train = tf.data.Dataset.from_generator(lambda: gpt2encodings_generator(train), output_signature=batched_output_signature)
    coco_dataset_test = tf.data.Dataset.from_generator(lambda: gpt2encodings_generator(test), output_signature=batched_output_signature)
    coco_dataset_val = tf.data.Dataset.from_generator(lambda: gpt2encodings_generator(val), output_signature=batched_output_signature)

    coco_dataset_train = coco_dataset_train.unbatch()
    coco_dataset_train = coco_dataset_train.map(lambda feat, cap_idx, string, id, len, h_s: (feat, cap_idx[:len], string, id, h_s[:len,:], tf.ones(shape=len, dtype=tf.float32)))
    coco_dataset_test = coco_dataset_test.unbatch()
    coco_dataset_test = coco_dataset_test.map(lambda feat, cap_idx, string, id, len, h_s: (feat, cap_idx[:len], string, id, h_s[:len,:], tf.ones(shape=len, dtype=tf.float32)))
    coco_dataset_val = coco_dataset_val.unbatch()
    coco_dataset_val = coco_dataset_val.map(lambda feat, cap_idx, string, id, len, h_s: (feat, cap_idx[:len], string, id, h_s[:len,:], tf.ones(shape=len, dtype=tf.float32)))

    if save_to_files:
        coco_dataset_train.save(path=save_to_files + "/train")
        coco_dataset_test.save(path=save_to_files + "/test")
        coco_dataset_val.save(path=save_to_files + "/val")
    return coco_dataset_train, coco_dataset_test, coco_dataset_val

def get_resnetfeature_gpt2tokencaption_gpt2embeddings_stringid_cocodatasets_type_spec():
    return (tf.TensorSpec(shape=(2048,), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(None, 768), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32))

def get_resnetfeature_gpt2tokencaption_gpt2embeddings_stringid_cocodatasets_default_path():
    return 'data/gpt2_coco_dataset_with_gpt2_embeddings'

def load_default_resnetfeature_gpt2tokencaption_gpt2encodings_stringid_cocodatasets():
    train, test, val = resnetfeature_gpt2tokencaption_gpt2encodings_stringid_cocodatasets(load_from_file=get_resnetfeature_gpt2tokencaption_gpt2embeddings_stringid_cocodatasets_default_path(), saved_element_spec=get_resnetfeature_gpt2tokencaption_gpt2embeddings_stringid_cocodatasets_type_spec())
    return train, test, val





if __name__ == "__main__":
    #train, test, val = load_default_resnetfeature_gpt2tokencaption_stringid_cocodatasets()
    #for _, t, _, _ in train.take(100):
    #    print(t.shape, t[-1])
    #save the datasets to file
    train, test, val = resnetfeature_gpt2tokencaption_gpt2encodings_stringid_cocodatasets(save_to_files=get_resnetfeature_gpt2tokencaption_gpt2embeddings_stringid_cocodatasets_default_path())
