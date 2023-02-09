import tensorflow as tf
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt
import os.path
import utils
import pickle
import numpy as np



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


def feature_cap_id_ds(
    num=None, path="data/coco_features_captions_id", print_bool=False
):
    path_train = f"{path}_train_num_{num}"
    path_test = f"{path}_test_num_{num}"
    train_ds, test_ds, val_ds = ms_coco_res50_features(
        num=num, path=path, print_bool=print_bool
    )
    train_ds = train_ds.map(
        lambda image, feature, res_img, captions, image_id, objects: (
            feature,
            captions["text"],
            image_id,
        )
    )
    test_ds = test_ds.map(
        lambda image, feature, res_img, captions, image_id, objects: (
            feature,
            captions["text"],
            image_id,
        )
    )

    # every element has multiple (~5) captions: Split the dataset such that each caption is another element
    def repeat_self(tensor, repeats):
        tensor = tf.expand_dims(tensor, axis=0)
        tensor = tf.repeat(tensor, repeats, axis=0)
        return tensor

    train_ds = (
        train_ds.map(
            lambda feature, captions, id: (
                repeat_self(feature, tf.shape(captions)[0]),
                captions,
                repeat_self(id, tf.shape(captions)[0]),
            )
        )
        .unbatch()
        .cache(path_train)
    )
    test_ds = (
        test_ds.map(
            lambda feature, captions, id: (
                repeat_self(feature, tf.shape(captions)[0]),
                captions,
                repeat_self(id, tf.shape(captions)[0]),
            )
        )
        .unbatch()
        .cache(path_test)
    )
    # add end of sequence and beginning of sequence
    train_ds = train_ds.map(
        lambda feature, caption, id: (
            feature,
            tf.strings.join(
                [
                    tf.convert_to_tensor(utils.START_TOKEN, dtype=tf.string),
                    caption,
                    tf.convert_to_tensor(utils.EOS_TOKEN, dtype=tf.string),
                ]
            ),
            id,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda feature, caption, id: (
            feature,
            tf.strings.join(
                [
                    tf.convert_to_tensor(utils.START_TOKEN, dtype=tf.string),
                    caption,
                    tf.convert_to_tensor(utils.EOS_TOKEN, dtype=tf.string),
                ]
            ),
            id,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return train_ds, test_ds


def extract_vocabulary(vocab_path="data/vocabulary", vocab_size=2000):

    vocab = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    vocab_path = vocab_path + f"_size_{vocab_size}.txt"
    if os.path.isfile(vocab_path):
        vocab.set_vocabulary(vocab_path)
    else:
        train_ds, _ = feature_cap_id_ds()
        voc_ds = train_ds.map(lambda feature, caption, id: caption)
        vocab.adapt(voc_ds)
        with open(vocab_path, "wt") as file:
            vocab_list = vocab.get_vocabulary(include_special_tokens=False)
            file.writelines([word + "\n" for word in vocab_list])
        print(vocab_list)

    return vocab


def feature_capAsIdxs_id_ds(
    num=None,
    path="data/coco_features_captions_id",
    print_bool=False,
    vocab_path="data/vocabulary",
    vocab_size=2000,
):
    vocab = extract_vocabulary(vocab_path=vocab_path, vocab_size=vocab_size)
    train_ds, test_ds = feature_cap_id_ds(num=num, path=path, print_bool=print_bool)
    train_ds = train_ds.map(
        lambda feature, caption, id: (feature, vocab(caption), id),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda feature, caption, id: (feature, vocab(caption), id),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return train_ds, test_ds


class Vocabulary:
    def __init__(self, vocab):
        # vocab is list of strings, where index is respective entry in embedding
        self.tokens = vocab.get_vocabulary()
        print(self.tokens)
        self.idx2word = self.tokens
        self.word2idx = {}
        for i, word in enumerate(self.tokens):
            self.word2idx[word] = i


if __name__ == "__main__":
    train, test = feature_capAsIdxs_id_ds()
    train = [[e.numpy() for e in elem] for elem in train]
    for i, elem in enumerate(train):
        print(i, elem)
    test = [[e.numpy() for e in elem] for elem in test]
    for i, elem in enumerate(test):
        print(i, elem)
    vocab = extract_vocabulary(vocab_path="data/vocabulary", vocab_size=2000)
    vocab = Vocabulary(vocab)
    vocab = {"idx2word": vocab.idx2word, "word2idx": vocab.word2idx}
    with open(
        "np_dataset/vocab2000_feat_cap_id_train_test_andvocab.pickle", "wb"
    ) as file:
        pickle.dump([[train, test], vocab], file)
    print("dumped")

    """
    for elem in train.take(5):
        print(elem[0])
        print(tf.image.resize(elem[0], size=(224, 224)).numpy())
        print(elem[2])
        print(tf.reduce_max(elem[2]))
        print(tf.reduce_min(elem[2]))


        plt.subplot(131)
        plt.imshow(elem[0].numpy())
        plt.subplot(132)
        plt.imshow(tf.keras.applications.resnet50.preprocess_input(tf.cast(tf.image.resize(elem[0], size=(224, 224)), tf.int8)).numpy())
        plt.subplot(133)
        min = tf.reduce_min(elem[2], axis=[0,1], keepdims=True)
        delta = tf.reduce_max(elem[2], axis=[0,1], keepdims=True) - tf.reduce_min(elem[2], axis=[0,1], keepdims=True)
        print(min, delta)
        restored = (elem[2]-min)/delta
        print(tf.reduce_min(restored), tf.reduce_max(restored), restored.shape)

        plt.imshow(tf.reverse(restored, axis=[-1]).numpy())
        plt.show()
    """
    """
    train_ds, test_ds = feature_cap_id_ds()
    for ds in [train_ds, test_ds]:
        for i, elem in enumerate(ds):
            # feat, cap, id = elem
            # print(cap)
            if i % 1000 == 0:
                print(f"{i} elemens prepared")
            else:
                continue
    vocab_results = []
    for vocab_size in [1500, 2000]:
        train_ds, test_ds = feature_capAsIdxs_id_ds(vocab_size=vocab_size)
        vocab = Vocabulary(extract_vocabulary(vocab_size=vocab_size))
        unknowns = 0
        for ds in [train_ds, test_ds]:
            for i, elem in enumerate(ds):
                feature, caption, id = elem
                caption = caption.numpy()
                caption = list(caption)
                sentence = [vocab.idx2word[idx] for idx in caption]
                print(caption, sentence)
                if "[UNK]" in sentence:
                    unknowns = unknowns + 1
                # print(sentence)
                if i % 1000 == 0:
                    print(f"{i} elemens prepared")
                else:
                    continue
        vocab_results.append(
            f"vocab size: {vocab_size} with {unknowns} many OOP token sentences out of 445000"
        )
    print(vocab_results)
    """
