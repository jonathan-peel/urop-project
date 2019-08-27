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






# inactive code below

all_image_paths = list(basepath.glob('[WM only,GM only]/*/*.bmp'))
all_image_paths = [str(path) for path in all_image_paths]

'List label indices for each image path'
# Make a list of all the classes WM and GM, make a
# dictionary for {label: index}, then make a list of labels (all_image_indices)
# corresponding to all_image_paths.
label_names = ['WM only', 'GM only']
label_string_to_index = dict((key, value) for value, key in enumerate(label_names))
all_image_indices = [
    label_string_to_index[Path(path).parent.parent.name] for path in all_image_paths]
IMAGE_COUNT = len(all_image_indices)

'Process images and extract 1D slices'
# Extract many 1D slices from each image in all_image_paths and maintain the
# correspondence with all_image_indices

for image_path in all_image_paths:
    # Extract image as np array (use pillow)
    # Take many slices of the image
    # save these slices as the colummns of a 2D np array, all_slices
    # make a corresponding list of all_slice_indices
    image = Image.open(image_path) #?
    pass











'Make image into tensor and resize/normalise'
img_shape = (511, 927)  # Original shape of images


def preprocess_image(image_file):
    """Function to preprocess the image."""
    # ??not sure if 3 output channels are needed??
    image_tensor = tf.image.decode_bmp(image_file, channels=1)
    # ??i think this line is loosing a lot of information??
    # image_tensor = tf.image.resize(image_tensor, img_shape)
    image_tensor = tf.dtypes.cast(image_tensor, dtype=tf.float32)
    image_tensor = image_tensor/255.0
    return image_tensor


def load_preprocess_image(image_path):
    # Wrapper to load the image and call preprocess_image()
    image_file = tf.io.read_file(image_path)
    return preprocess_image(image_file)


def viewimg_get_input(completed_action):
    """Ask the user whether they want to view an images or not for a given
       point in the program.
    """
    viewimg = input('Finished ' + completed_action +
                    ' image(s): View an image now? (enter [y]/[n]): ')
    while not (viewimg == 'y' or viewimg == 'n'):
        # Validate entry
        viewimg = input('Invalid entry, please try again: ')
    return viewimg


def view_preprocessed_image():
    """View a post-preprocessing image."""
    image_num = np.random.randint(0, IMAGE_COUNT)
    print('Displaying image ' + str(image_num) + ' ...')
    image_path = all_image_paths[image_num]
    image_label = all_image_indices[image_num]
    image_tensor = load_preprocess_image(image_path)
    # Convert tensor to array array using Sessions
    image_array = tf.compat.v1.Session().run(image_tensor)
    # image_array = image_array.reshape(img_shape) # necessary?
    image_array = np.squeeze(image_array)  # Remove dimensions of size 1
    plt.imshow(image_array)
    plt.title('Image number {}: {}'.format(image_num,
              label_names[image_label])
              )
    plt.xlabel('Image path: ' + str(image_path))
    plt.show()


# Give the user the option of viewing one of the images after preprocessing.
viewimg = viewimg_get_input(completed_action='processing')  # Get user input
while viewimg == 'y':
    view_preprocessed_image()
    viewimg = viewimg_get_input(completed_action='displaying')


'Make databases'
# Create a dataset of the strings to each path.
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# Dynamically set the number of parallel calls based on available CPUs.
AUTOTUNE = tf.data.experimental.AUTOTUNE
# Map the load_preprocess_image function onto all the elements making a
# dataset of images.
image_ds = path_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
# Create a dataset of the labels.
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_indices, tf.int64)
    )
# Zip the two datasets together to form a dataset of (image, label) pairs.
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

'Prepare input pipeline'
# Shuffle the entire dataset, ensure it repeats forever and separate it into
# batches. Then let the model fetch batches in the background during training.
BATCH_SIZE = 32
ds = image_label_ds.shuffle(buffer_size=IMAGE_COUNT)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

# Split the dataset into training and validation

# Give the user the option of veiwing one of the images from the datasets
viewimg = viewimg_get_input(completed_action='building dataset')

'Build model'
'-----------'
# start with a simple model with no convolutions, might need to use transfer
# learning
model = tf.keras.Sequential([
    layers.Flatten(input_shape=img_shape),  # make input images array 1D
    layers.Dense(128, activation='relu'),  # hidden layer
    # classification layer
    layers.Dense(len(label_names), activation='softmax')
])

# compile it for catagorisation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

'train model'
'-----------'
# steps_per_epoch = tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
# model.fit(ds, epochs=5, steps)