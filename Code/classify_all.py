"""This program intends to classify all the OCT scans as each of the 6 folder
   classes: GM Bottom, GM Gelatine, GM only, WM Bottom, WM Gelatine and WM
   only.
"""

# from PIL import Image
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution  # lets tensorflow behave like python

'preprocess images for training'
'------------------------------'
'collect image paths'
# Define the base directory of all the images and convert it into a
# pathlib.WindowsPath object. Then collect all the image paths into a
# list of of WindowsPath objects, convert to strings and shuffle.
basedir = 'C:/Users/JayPee/OneDrive - Imperial College London/UROP/Needle \
OCT_Ian'
basepath = Path(basedir)
all_image_paths = list(basepath.glob('*/*/*.bmp'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

'list label indices for each image path'
label_names = sorted(
    item.name for item in basepath.glob('*/') if item.is_dir())
# lists all the subfolder names
label_to_index = dict((key, value) for value, key in enumerate(label_names))
# creates dictionary {label: index}
all_image_labels = [
    label_to_index[Path(path).parent.parent.name] for path in all_image_paths]
# creates a list with all the corresponding label indices of each item in
# all_image_paths
IMAGE_COUNT = len(all_image_labels)


'Make image into tensor and resize/normalise'
img_shape = (511, 927)  # Original shape of images


def preprocess_image(image_file):
    """Function to preprocess the image."""
    # ??not sure if 3 output channels are needed??
    image_tensor = tf.image.decode_bmp(image_file, channels=1)
    # ??i think this line is loosing a lot of information??
    # image_tensor = tf.image.resize(image_tensor, img_shape)
    image_tensor = tf.dtypes.cast(image_tensor, dtype=tf.float32)
    # image_tensor = image_tensor/255.0
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
    image_label = all_image_labels[image_num]
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
    tf.cast(all_image_labels, tf.int64)
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
