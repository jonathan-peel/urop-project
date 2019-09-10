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
from tensorflow import keras
from tensorflow.keras import layers
import inspect_dataset

label_names = ['WM only', 'GM only']

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


def set_dataset_shape(slice, label):
    """Set the shape of the slice feature to (410,) as it was not inferred by
       tensorflow while due to the use of py_func in preprocessing
    """
    slice.set_shape((410,))
    return (slice, label)


ds_serialized_slices = ds_cached.map(parse_function)
ds = ds_serialized_slices.map(parse_slices)
# Set shape of tensors lost in processing
ds = ds.map(set_dataset_shape)

'Inspect dataset'
# Give the user the option of viewing slices from the dataset.
# num_slices = inspect_dataset.get_input(
#     completion_message='Dataset of slices is complete.')
num_slices = 0  # Skip this step
while not num_slices == 0:
    inspect_dataset.display_slices(num_slices, ds)
    num_slices = inspect_dataset.get_input(
        completion_message='Finished displaying slices.')

'Prepare datasets'
# Separate dataset into training, validation and testing.
TOTAL_NUMBER = len(list(ds))
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
# element. The Nones in shapes indicate that the batch dimension is not
# constant.
# batch during fit/evaluate/predict as needed instead of batching the dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32  # no reason to choose this number?
ds_train = ds_train.shuffle(TRAIN_NUMBER).batch(BATCH_SIZE)\
           .prefetch(AUTOTUNE)  # What is AUTOTUNE?, Should I repeat?
ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds_test = ds_test.batch(1)  # ds_test now has same shape as other datasets

'Build model'
# Start simple with just a couple of dense layers.
model = keras.models.Sequential([
    layers.Dense(128, activation='relu', batch_input_shape=(None, 410)),
    layers.Dense(2, activation='softmax')
])
# Compile with normal opt5imizer and loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

'Define callbacks'
# Enable analysis on tensorboard
tensorboard_cbk = \
    keras.callbacks.TensorBoard(histogram_freq=1, write_images=True,
                                update_freq='batch', embeddings_freq=1)
PATIENCE = 5
early_stop_cbk = \
    keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001,
                                  patience=5, restore_best_weights=True)
callbacks = [tensorboard_cbk, early_stop_cbk]

'Train model'
EPOCHS = 50  # number of epochs likely lower due to early stopping
history = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_val,
                    callbacks=callbacks)

'Analyse training history'
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
num_epochs = len(acc)
epochs_range = range(num_epochs)
print('\nTraining finished in {} epochs as increase in validation accuracy \
not recorded for {} epochs.\n'.format(num_epochs, PATIENCE))

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
# Doesn't work
results = model.evaluate(ds_test)
success_rate = round(results[1], 3)
print('\nFinal test accuracy: {:.1%}\n'.format(success_rate))

'Generate predictions'
# 0 is white matter, 1 is grey matter.
rand_int = random.randint(0, TEST_NUMBER-1)
rand_element = next(iter(ds_test.skip(rand_int).take(1)))
actual_class = label_names[rand_element[1][0].numpy()]
predictions = model.predict(rand_element)[0]
x_pos = np.arange(len(predictions))
plt.bar(x_pos, predictions)
plt.xticks(x_pos, labels=label_names)
plt.xlabel('Classes')
plt.ylabel('Probability')
plt.title('Class prediction for a {} scan'.format(actual_class))
plt.show()
