"""This program preprocesses all the 'WM only' and 'GM only' Needle OCT_lan
   BMP images into a large labelled dataset, where each element is a 1D
   vertical slice of the images. This dataset is then prepared for training
   and saved.
"""

from matplotlib.image import imread
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from inspect import get_input
from inspect import display_slices

'Preprocess images'
# Define the base directory of all the images and convert it into a
# pathlib.WindowsPath object. Then collect all the WM image paths into a
# list of of WindowsPath objects and convert these to strings. WM slices have
# an index of 0 and GM slices have an index of 1.
basepath = Path.cwd() / 'Needle OCT_Ian'
WM_paths = list(basepath.glob('WM only/**/*.bmp'))
GM_paths = list(basepath.glob('GM only/**/*.bmp'))
WM_paths = [str(path) for path in WM_paths]
GM_paths = [str(path) for path in GM_paths]
all_paths = WM_paths + GM_paths
label_name_to_index = {'WM only': 0, 'GM only': 1}
label_names = ['WM only', 'GM only']

WM_NUMBER = len(WM_paths)
GM_NUMBER = len(GM_paths)
TOTAL_NUMBER = len(all_paths)


def make_slice_array():
    """Load the images into arrays and extract slices from each image to build
       up a big 2D array (a list of slices). The final array has shape
       (number-of-slices, size-of-each-slice).
    """
    all_slices_list = []
    image_arrays = []
    for counter, image_path in enumerate(all_paths):
        image = Image.open(image_path)
        image_cropped = image.crop((60, 44, 827, 454))
        image_arrays.append(np.asarray(image_cropped))
        image_arrays[counter] = image_arrays[counter].transpose()
        print('Completed storing image ' + str(counter + 1) +
              ' of ' + str(TOTAL_NUMBER)
              )
    return np.vstack(image_arrays)


def make_label_list():
    label_list = []
    for counter in range(TOTAL_NUMBER):
        if counter < WM_NUMBER:
            label_list.append(label_name_to_index['WM only'])
        elif counter <= TOTAL_NUMBER:
            label_list.append(label_name_to_index['GM only'])
    return label_list


def normalize(dataset_element):
    """Function to be mapped onto slices dataset."""
    dataset_element = tf.cast(dataset_element, tf.float32)
    return dataset_element / 255


'Make datasets'
# Make a list of 1D slices (as an nd array) then a list of corresponding
# labels. Convert these lists into datasets and zip then shuffle them ready
# for training. Note: slices have to be normalized after dataset is created
# to avoid initialising array over 2GB error.
AUTOTUNE = tf.data.experimental.AUTOTUNE  # Optimum number of parallel calls
all_slices = make_slice_array()
label_list = make_label_list()
slice_dataset = tf.data.Dataset.from_tensor_slices(all_slices)
slice_dataset = slice_dataset.map(normalize, num_parallel_calls=AUTOTUNE)
label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
ds = tf.data.Dataset.zip((slice_dataset, label_dataset))
ds = ds.shuffle(buffer_size=TOTAL_NUMBER)


'Inspect dataset'
# Give the user the option of viewing x slices from the dataset.
num_slices = get_input(
    completion_message='Dataset of slices is complete.')
while not num_slices == 0:
    display_slices(num_slices, ds)
    num_slices = get_input(
        completion_message='Finished displaying slices.')


def serialize_slices(slice_, label):
    """Function to be mapped to dataset. Returns a scalar tf.Tensor of type
       String for the slice component and does not change the label
       component.
    """
    return (tf.io.serialize_tensor(slice_), label)


def bytes_feature(value):
    """Returns a tf.train.BytesList type from a string (serialized slice) to
    adhere to tf.Example message {string: tf.train.Feature} mapping.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    """Returns a tf.train.Int64List type from an int32 type (label)."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_to_example_numpy(slice_, label):
    """Main function for converting each element of the dataset into
       tf.Example-compatible data types for the dataset. Then make a
       tf.Example message and serialize it for storage.
    """
    # Make features dictionary.
    feature = {
               'slice': bytes_feature(slice_.numpy()),
               'label': int64_feature(label.numpy()),
               }
    # Create the actual tf.Example message message type (protobuf) using
    # tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(
        feature=feature))
    return example_proto.SerializeToString()


def serialize_to_example_tf(slice_, label):
    """To map the serialize_to_example_numpy function to the datasets, it must be
       able to operate in tensorflow graph mode, ie. using tf.Tensors. This
       function uses tf.Tensors so it can be mapped to a dataset, and it also
       calls serialize_to_example_numpy using tf.py_function, which converts
       the tf.Tensors to arrays so it can function.
    """
    tf_example = tf.py_function(
                                serialize_to_example_numpy,  # function
                                (slice_, label),  # inputs
                                tf.string  # return type
                                )
    return tf.reshape(tf_example, ())  # make result a scalar


'Save dataset to be used in training'
# The slice components of the dataset must be converted to a scalar tf.Tensor
# using tf.serialize_tensor in serialize_components. The dataset is then saved
# to file.
ds_serialized_slices = ds.map(serialize_slices)  # slices are now strings
ds_cached = ds_serialized_slices.map(serialize_to_example_tf)
filename = 'C:\\Users\\JayPee\\OneDrive - Imperial College London\\UROP\\\
Code\\WM-and-GM-dataset.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(ds_cached)
