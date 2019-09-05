"""This program is a proof of concept to show whether deep learning can be
   used to classify grey matter and white matter images.
"""

from matplotlib.image import imread
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import inspect

'Load dataset'
# Load the dataset from the TFRecord file.
filename = 'Code\\WM-and-GM-dataset.tfrecord'
ds_cached = tf.data.TFRecordDataset(filename)
# Describe the nature of the dataset elements.
feature_description = {
    'slice': tf.io.FixedLenFeature([], tf.string, default_value='fail'),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=69),
}


def parse_function(example_message):
    # Parse the input tf.Example proto using the feature_description dict.
    return tf.io.parse_single_example(example_message, feature_description)


def parse_slices(element):
    """Function to be mapped to ds_serialised_slices. Returns the fully
       parsed array version of the slices.
    """
    return (tf.io.parse_tensor(element['slice'], tf.float32), element['label'])


ds_serialized_slices = ds_cached.map(parse_function)
ds = ds_serialized_slices.map(parse_slices)

'Inspect dataset'
# Give the user the option of viewing slices from the dataset.
num_slices = inspect.get_input(
    completion_message='Dataset of slices is complete.')
while not num_slices == 0:
    inspect.display_slices(num_slices)
    num_slices = inspect.get_input(
        completion_message='Finished displaying slices.')

'Prepare datasets'
# Separate dataset into training, validation and testing.
TRAIN_NUMBER = int(round(0.7 * TOTAL_NUMBER))
VAL_NUMBER = int(round(0.2 * TOTAL_NUMBER))
TEST_NUMBER = int(round(0.1 * TOTAL_NUMBER))
assert TRAIN_NUMBER + VAL_NUMBER + TEST_NUMBER == TOTAL_NUMBER,\
       'Rounding error when splitting dataset.'

ds_train = ds.take(TRAIN_NUMBER)
ds_temp = ds.skip(TRAIN_NUMBER)
ds_test = ds_temp.skip(VAL_NUMBER)
ds_val = ds_temp.take(VAL_NUMBER)
del ds_temp

# Repeat and batch dataset with prefetch. Note this is done after the
# creation of daughter datasets and inspection since making the dataset
# infinite (repeating) and separating into batches changes the nature of each
# element. The '?'s in shapes indicate that the batch dimension is not
# constant.
BATCH_SIZE = 32  # no reason to choose this number?
ds_train = ds_train.shuffle(TRAIN_NUMBER).batch(BATCH_SIZE)\
           .prefetch(AUTOTUNE)  # ?What is AUTOTUNE?, ?Should I repeat?
ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
