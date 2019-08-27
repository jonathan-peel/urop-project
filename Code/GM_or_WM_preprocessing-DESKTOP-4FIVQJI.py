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
# from tensorflow.keras import layers


tf.compat.v1.enable_eager_execution  # lets tensorflow behave like python

'Preprocess slices for training'
'------------------------------'
'Collect slices'
# Define the base directory of all the images and convert it into a
# pathlib.WindowsPath object. Then collect all the WM image paths into a
# list of of WindowsPath objects and convert these to strings.
basedir = 'C:/Users/JayPee/OneDrive - Imperial College London/UROP/Needle \
OCT_Ian/'
basepath = Path(basedir)
WM_paths = list(basepath.glob('WM only/**/*.bmp'))
GM_paths = list(basepath.glob('GM only/**/*.bmp'))
WM_paths = [str(path) for path in WM_paths]
GM_paths = [str(path) for path in GM_paths]
all_paths = WM_paths + GM_paths
label_string_to_index = {'WM only': 1, 'GM only': 2}

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
            label_list.append(label_string_to_index['WM only'])
        elif counter <= TOTAL_NUMBER:
            label_list.append(label_string_to_index['GM only'])
    return label_list


def normalise(dataset_element):
    """Function to be mapped onto slices dataset."""
    dataset_element = tf.cast(dataset_element, tf.float32)
    return dataset_element / 255


# Make a list of 1D slices (as an nd array) then a list of corresponding
# labels. Convert these lists into datasets and zip then shuffle them ready
# for training. Note: slices had to be normalised after dataset was created
# to avoid initialising array over 2GB error.
AUTOTUNE = tf.data.experimental.AUTOTUNE  # Optimum number of parallel calls
all_slices = make_slice_array()
label_list = make_label_list()
slice_dataset = tf.data.Dataset.from_tensor_slices(all_slices)
slice_dataset.map(normalise, num_parallel_calls=AUTOTUNE)
label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
ds = tf.data.Dataset.zip((slice_dataset, label_dataset))