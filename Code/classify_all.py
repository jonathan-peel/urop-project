# from PIL import Image
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution   # lets tensorflow behave like python

'preprocess images for training'
'------------------------------'
'collect image paths'
basedir = 'C:/Users/JayPee/OneDrive - Imperial College London/UROP/Needle OCT_Ian'
# the base directory of all the images
# makes the directory into a pathlib.WindowsPath object
basepath = Path(basedir)
all_image_paths = list(basepath.glob('*/*/*.bmp'))
# collects all the images' directories into a long list of of WindowsPath objects
all_image_paths = [str(path) for path in all_image_paths]
# converts all the WindowsPath objects into strings
random.shuffle(all_image_paths)  # shuffles list of image paths

'list label indices for each image path'
label_names = sorted(
    item.name for item in basepath.glob('*/') if item.is_dir())
# lists all the subfolder names
label_to_index = dict((key, value) for value, key in enumerate(label_names))
# creates dictionary {label: index}
all_image_labels = [
    label_to_index[Path(path).parent.parent.name] for path in all_image_paths]
# creates a list with all the corresponding label indices of each item in all_image_paths
global image_count
image_count = len(all_image_labels)

'make image into tensor and resize/normalise'
img_shape = (255, 255)


def preprocess_image(image_file):    # function to preprocess the image
    # ??not sure if 3 output channels are needed??
    image_tensor = tf.image.decode_bmp(image_file, channels=1)
    # ??i think this line is loosing a lot of information??
    image_tensor = tf.image.resize(image_tensor, img_shape)
    image_tensor = image_tensor/255.0
    return image_tensor


# wrapper to load image the call preprocess_image()
def load_preprocess_image(image_dir):
    image_file = tf.io.read_file(image_dir)
    return preprocess_image(image_file)


'display an image'


# possibly change this function slightly later to make the code more reusable
def displayimg(image_num):
    image_dir = all_image_paths[image_num]
    image_label = all_image_labels[image_num]
    image_tensor = load_preprocess_image(image_dir)
    # !converts tf.Tensor object into np.array!
    image_arr = tf.compat.v1.Session().run(image_tensor)
    image_arr = image_arr.reshape(img_shape)
    plt.imshow(image_arr)
    plt.xlabel('Image number ' + str(image_num) +
               ': ' + label_names[image_label].title())
    plt.show()


'give the user the option of viewing one of the images after preprocessing'


def viewimgfctn(action):
    viewimg = input('Finished ' + action +
                    ' images: View an image now? (enter [y]/[n]): ')
    while not viewimg = 'y' or 'n':  
        # validate entry
        viewimg = input('Invalid entry, please try again: ')
    return viewimg


def view_preprocessed_image(viewimg):
    while viewimg == 'y':
        image_num = np.random.randint(0, image_count)
        print('Displaying image ' + str(image_num) + ' ...')
        displayimg(image_num)
        viewimg = input(
            'Would you like to view another image? (enter: [y]/[n]): ')


viewimg = viewimgfctn('processing')  # get input from user
view_preprocessed_image(viewimg)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# creates a dataset of the strings to each path
AUTOTUNE = tf.data.experimental.AUTOTUNE
# this dynamically sets the number of parallel calls based on available CPU
image_ds = path_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
# map the load_preprocess_image function onto all the elements making a dataset of images
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))
# creates a dataset of the labels
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# zips the 2 datasets together to form a dataset of (image, label) pairs

'prepare input pipeline'
BATCH_SIZE = 32
# shuffles the entire dataset
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()    # ensures the dataset repeats forever
ds = ds.batch(BATCH_SIZE)   # separates the data into batches
ds = ds.prefetch(buffer_size=AUTOTUNE)
# lets the dataset fetch batches in the background while the model is training

'build model'
'-----------'
# start with a simple model with no convolutions, might need to use transfer learning
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
