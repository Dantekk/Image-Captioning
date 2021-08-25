import re
import os
import numpy as np
import tensorflow as tf
from settings import *

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
AUTOTUNE = tf.data.AUTOTUNE

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def train_val_split(caption_data, train_size=0.8, shuffle=True):
    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data

def reduce_dataset_dim(captions_mapping_train, captions_mapping_valid):
    train_data = {}
    conta_train = 0
    for id in captions_mapping_train:
        if conta_train<=NUM_TRAIN_IMG:
            train_data.update({id : captions_mapping_train[id]})
            conta_train+=1
        else:
            break

    valid_data = {}
    conta_valid = 0
    for id in captions_mapping_valid:
        if conta_valid<=NUM_VALID_IMG:
            valid_data.update({id : captions_mapping_valid[id]})
            conta_valid+=1
        else:
            break

    return train_data, valid_data


def read_image(img_path, size=IMAGE_SIZE):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


trainAug = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.05, 0.15)),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
	tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
	tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.10, 0.10))
])

def conclude_image(img):
    img = tf.squeeze(img, axis=0)
    return img


def make_dataset(images, captions, data_aug, tokenizer):
    img_dataset = tf.data.Dataset.from_tensor_slices(images)

    if data_aug:
        img_dataset = (img_dataset
                       .map(read_image, num_parallel_calls=AUTOTUNE)
                       .map(lambda x: (conclude_image(trainAug(x))), num_parallel_calls=AUTOTUNE))
    else:
        img_dataset = (img_dataset
                       .map(read_image, num_parallel_calls=AUTOTUNE)
                       .map(lambda x: (conclude_image(x)), num_parallel_calls=AUTOTUNE))

    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(tokenizer, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_DIM).prefetch(AUTOTUNE)
    return dataset
