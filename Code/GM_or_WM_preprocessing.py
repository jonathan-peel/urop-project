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
import pickle

tf.compat.v1.enable_eager_execution()  # lets tensorflow behave like python

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


def normalise(dataset_element):
    """Function to be mapped onto slices dataset."""
    dataset_element = tf.cast(dataset_element, tf.float32)
    return dataset_element / 255


'Make datasets'
# Make a list of 1D slices (as an nd array) then a list of corresponding
# labels. Convert these lists into datasets and zip then shuffle them ready
# for training. Note: slices have to be normalised after dataset is created
# to avoid initialising array over 2GB error.
AUTOTUNE = tf.data.experimental.AUTOTUNE  # Optimum number of parallel calls
all_slices = make_slice_array()
label_list = make_label_list()
slice_dataset = tf.data.Dataset.from_tensor_slices(all_slices)
slice_dataset = slice_dataset.map(normalise, num_parallel_calls=AUTOTUNE)
label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
ds = tf.data.Dataset.zip((slice_dataset, label_dataset))
ds = ds.shuffle(buffer_size=TOTAL_NUMBER)

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


def display_slices(num_slices):
    """Plot a random sample of num_slices on the same axes. Note how ds.skip
       is used with np.random.randint to choose a sample from a random section
       of the database.
    """
    fig = plt.figure()
    ax = plt.axes()
    skip_num = np.random.randint(TOTAL_NUMBER-num_slices)

    for counter, (slice, label_index) in \
            enumerate(ds.skip(skip_num).take(num_slices)):
        slice_label = str(counter+1) + ': ' + label_names[label_index]
        plt.plot(np.linspace(0, 600, len(list(slice))), slice,
                 label=slice_label
                 )  # ?Are all the images 600?

    plt.xlabel('Frequency (Hz)')  # ?Is it Hz?
    plt.ylabel('Intensity')
    plt.title('Sample of ' + str(num_slices) + ' slices from dataset')
    plt.legend()

    plt.show()


def get_input(completion_message):
    num_slices = input(completion_message + ' How many slices would\
you like to view? (Type 0 to skip): ')
    while True:
        try:
            num_slices = int(num_slices)
        except ValueError:
            num_slices = input('Invalid entry. \
Please input a positive integer: ')
        else:
            return num_slices


'Inspect dataset'
# Give the user the option of viewing x slices from the dataset.
num_slices = get_input(completion_message='Dataset of slices is complete.')
while not num_slices == 0:
    display_slices(num_slices)
    num_slices = get_input(completion_message='Finished displaying slices.')

'Prepare datasets for training'
# Repeat and batch dataset with prefetch. Note this is done after the
# creation of daughter datasets and inspection since making the dataset
# infinite (repeating) and separating into batches changes the nature of each
# element. The '?'s in shapes indicate that the batch dimension is not
# constant.
BATCH_SIZE = 32  # no reason to choose this number?
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)  # Changes shapes to ((?, 410), (?, ))
ds = ds.prefetch(buffer_size=AUTOTUNE)


def save_object(dataset, new_file_name):
    with open(new_file_name, mode='wb') as open_file:
        pickle.dump(dataset, open_file)


'Save dataset to be used in training'
# Save the datasets as a dictionary object to the local directory to be used
# later in training.
datasets = {'complete_dataset': ds, 'training_dataset': ds_train,
            'validation_dataset': ds_val, 'testing_dataset': ds_test
            }
save_object(datasets, 'Datasets')
# !use tf_records instead!?
