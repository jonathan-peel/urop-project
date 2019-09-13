"""This program preprocesses all the 'WM only' (white matter), 'GM only'
   (grey matter), 'GM bottom' and 'WM bottom' Needle OCT_lan BMP images into a
   large labelled dataset, with each image preprocessed into many vertical
   slices. A neural network is then trained to classify this data as 'GM only',
   'WM only' or 'GM and WM' and tested.
"""

from matplotlib.image import imread
from PIL import Image
import matplotlib.pyplot as plt
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers
import inspect_dataset
import time

print('\nFinished importing libraries.')

'Preprocess images'
# Define the base directory of all the images and convert it into a
# pathlib.WindowsPath object. Then collect all the WM image paths into a
# list of of WindowsPath objects and convert these to strings. WM slices have
# an index of 0, GM slices have an index of 1 and slices with both GM and WM
# (referred to as 'GMWM') have an index of 2.
basepath = Path.cwd() / 'Needle OCT_Ian'
WM_paths = list(basepath.glob('WM only/**/*.bmp'))
GM_paths = list(basepath.glob('GM only/**/*.bmp'))
GM_bottom_paths = list(basepath.glob('GM Bottom/**/*.bmp'))
WM_bottom_paths = list(basepath.glob('WM Bottom/**/*.bmp'))
GMWM_paths = GM_bottom_paths + WM_bottom_paths
WM_paths = [str(path) for path in WM_paths]
GM_paths = [str(path) for path in GM_paths]
GMWM_paths = [str(path) for path in GMWM_paths]
all_paths = WM_paths + GM_paths + GMWM_paths
label_names = ['WM only', 'GM only', 'GM and WM']

WM_IMAGES = len(WM_paths)
GM_IMAGES = len(GM_paths)
GMWM_IMAGES = len(GMWM_paths)
TOTAL_IMAGES = len(all_paths)
print('\nCollected {} OCT scan image paths consisting of {} Grey Matter, {} \
White Matter and {} Grey and White Matter images.'
      .format(TOTAL_IMAGES, GM_IMAGES, WM_IMAGES, GMWM_IMAGES))


def make_image_array_list(all_paths):
    """Load the images into numpy arrays."""
    image_array_list = []
    for counter, image_path in enumerate(all_paths):
        image = Image.open(image_path)
        image_cropped = image.crop((60, 44, 827, 454))
        image_array_list.append(np.asarray(image_cropped))
        image_array_list[counter] = image_array_list[counter].transpose()
    return image_array_list


def make_label_list():
    """Make a list where each element is the class corresponding to each
       element of the image_array_list.
    """
    label_list = []
    for counter in range(TOTAL_IMAGES):
        if counter <= WM_IMAGES:
            label_list.append(0)  # 0: white matter
        elif counter <= WM_IMAGES+GM_IMAGES:
            label_list.append(1)  # 1: grey matter
        elif counter <= TOTAL_IMAGES:
            label_list.append(2)  # 2: both grey and white matter
    return label_list


def image_list_to_slice_list(image_list):
    """Split each of the images into 767 1D slices while preserving each
       image's label.
    """
    images, labels = zip(*image_list)  # unzips list
    image_array = np.asarray(images)
    slice_array = np.vstack(images)  # 662,688 slices of 410 pixels
    slice_array = slice_array/255  # normalise

    slices_per_image = image_array.shape[1]  # 767
    label_list_new = []

    for counter, label in enumerate(labels):
        append_list = [label] * slices_per_image
        for append_item in append_list:
            label_list_new.append(append_item)

    label_list = label_list_new  # 662,688 labels
    return (slice_array, label_list)


'Make datasets'
# First generate a list containing a numpy array for each image, then label
# each image with its class in a list of tuples. Shuffle this list then
# separate it into separate training, validation and testing lists. For each
# list, separate and combine each image into an array of all their constituent
# slices and generate a corresponding list of labels. Make GMWM of these into
# datasets then zip them together.
# Each image contains 767 slices.
image_array_list = make_image_array_list(all_paths)
print('\nCompeted storing {} images as arrays.'.format(TOTAL_IMAGES))
label_list = make_label_list()
labelled_image_list = list(zip(image_array_list, label_list))
random.shuffle(labelled_image_list)
print('\nCollected and shuffled list of all images with labels.')

TRAIN_IMAGES = int(round(0.7 * TOTAL_IMAGES))
VAL_IMAGES = int(round(0.2 * TOTAL_IMAGES))
TEST_IMAGES = int(round(0.1 * TOTAL_IMAGES))
# assert TRAIN_IMAGES + VAL_IMAGES + TEST_IMAGES\
#     == TOTAL_IMAGES, 'Rounding error when splitting dataset.'

train_images = labelled_image_list[:TRAIN_IMAGES]
val_images = labelled_image_list[TRAIN_IMAGES:TRAIN_IMAGES+VAL_IMAGES]
test_images = labelled_image_list[TRAIN_IMAGES+VAL_IMAGES:TOTAL_IMAGES]
print('\nSeparated image list into training, validation and testing.')

train_slices, train_labels = image_list_to_slice_list(train_images)
val_slices, val_labels = image_list_to_slice_list(val_images)
test_slices, test_labels = image_list_to_slice_list(test_images)
SLICES_PER_IMAGE = int(len(train_labels) / len(train_images))  # 767
TRAIN_SLICES = len(train_labels)  # 662,688
VAL_SLICES = len(val_labels)  # 189,449
TEST_SLICES = len(test_labels)  # 98,108
print('\nFinished splitting each image within each list into {} 1D slices.'
      .format(SLICES_PER_IMAGE))
# Note the slices are now normalised

train_slices_ds = Dataset.from_tensor_slices(train_slices)
val_slices_ds = Dataset.from_tensor_slices(val_slices)
test_slices_ds = Dataset.from_tensor_slices(test_slices)
print('\nCompleted slice datasets.')
train_labels_ds = Dataset.from_tensor_slices(train_labels)
val_labels_ds = Dataset.from_tensor_slices(val_labels)
test_labels_ds = Dataset.from_tensor_slices(test_labels)
print('Completed label datasets.')
train_ds = Dataset.zip((train_slices_ds, train_labels_ds))
val_ds = Dataset.zip((val_slices_ds, val_labels_ds))
test_ds = Dataset.zip((test_slices_ds, test_labels_ds))
print('\nCompleted datasets of labelled slices.')

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256
train_ds = train_ds.shuffle(TRAIN_SLICES).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.shuffle(VAL_SLICES).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)  # ds_test now has same shape as others
print('\nFinished batching and shuffling datasets.')

'Build model'
# Start simple with just a single dense layer.
model = keras.models.Sequential([
    layers.Dense(128, activation='relu', batch_input_shape=(None, 410)),
    layers.Dense(3, activation='softmax')
])
# Compile with normal optimizer and loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print('\nCompleted building model:')
model.summary()

'Define callbacks'
tensorboard_cbk = \
    keras.callbacks.TensorBoard(histogram_freq=1, write_images=True,
                                embeddings_freq=1)  # too much detail?
PATIENCE = 2
early_stop_cbk = \
    keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001,
                                  patience=PATIENCE, restore_best_weights=True)
callbacks = [tensorboard_cbk, early_stop_cbk]
print('\nAnalysis on tensorboard enabled. After running, \
type \'tensorboard --logdir log\' from the command line to activate.\n')

'Train model'
EPOCHS = 2
BATCH_NUM = TRAIN_SLICES / BATCH_SIZE
print('Training {} batches over {} epoch(s) ...\n'.format(BATCH_NUM, EPOCHS))
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                    callbacks=callbacks)

'Analyse training history'
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
num_epochs = len(acc)
epochs_range = range(num_epochs)
print('\nTraining finished in {} epoch(s).\n'.format(num_epochs))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='Validation accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.title('Training and validation accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.title('Training and validation loss')
plt.show()

'Test model performance on unseen data'
print('\nTesting model on unseen data.')
results = model.evaluate(test_ds)
success_rate = round(results[1], 3)
print('\nFinal test accuracy: {:.1%}\n'.format(success_rate))

'Generate predictions'
# 0 is white matter, 1 is grey matter, 2 is both.
rand_int = random.randint(0, TEST_IMAGES-1)
rand_element = next(iter(test_ds.skip(rand_int).take(1)))
actual_class = label_names[rand_element[1][0].numpy()]
time_start = time.time()
predictions = model.predict(rand_element)[0]
time_end = time.time()
x_pos = np.arange(len(predictions))
plt.bar(x_pos, predictions)
plt.xticks(x_pos, labels=label_names)
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.title('Class prediction for a {} scan'.format(actual_class))
plt.caption
print('\nPrediction generated in {} seconds.'.format(time_end-time_start))
plt.show()
pass
