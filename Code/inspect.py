"""This is a module which holds the functions necessary to inspect a dataset
"""

from matplotlib import pyplot as plt


def get_input(completion_message):
    """Acquire input from the user after a custom action is completed."""
    num_slices = input(completion_message + ' How many slices would \
you like to view? (Type 0 to skip): ')
    while True:
        try:
            num_slices = int(num_slices)
        except ValueError:
            num_slices = input('Invalid entry. \
Please input a positive integer: ')
        else:
            return num_slices


def display_slices(num_slices, dataset):
    """Plot a random sample of num_slices on the same axes. Note how ds.skip
       is used with np.random.randint to choose a sample from a random section
       of the database.
    """
    fig = plt.figure()
    ax = plt.axes()
    skip_num = np.random.randint(TOTAL_NUMBER-num_slices)

    for counter, (slice_, label_index) in \
            enumerate(dataset.skip(skip_num).take(num_slices)):
        slice_label = str(counter+1) + ': ' + label_names[label_index]

        if label_index.numpy() == 0:  # 0 = WM
            color = 'b'
        elif label_index.numpy() == 1:  # 1 = GM
            color = 'g'

        line = plt.plot(np.linspace(0, 600, len(list(slice_))), slice_,
                        label=slice_label, color=color
                        )  # ?Are all the images 600?

    plt.xlabel('Frequency (Hz)')  # ?Is it Hz?
    plt.ylabel('Intensity')
    plt.title('Sample of ' + str(num_slices) + ' slice(s) from dataset')
    plt.legend()

    plt.show()
