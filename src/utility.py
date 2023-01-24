import argparse
import signal
import sys

import numpy as np


def find_the_distance(mutated_input, last_node):
    root = last_node
    while (root.parent != None):
        root = root.parent

    initial_input = root.state.mutated_input

    dist = np.sum((mutated_input - initial_input) ** 2) / mutated_input.size
    return dist


figure_count = 0


def init_image_plots(rows, columns, input_shape, figsize=(8, 8)):
    global figure_count
    import matplotlib.pyplot as plt
    image_size = get_image_size(input_shape)
    plt.ion()
    figure_count += 1
    fig = plt.figure(figure_count, figsize=figsize)
    fig_plots = []
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        subplot = plt.imshow(np.random.randint(0, 256, size=image_size))
        fig_plots.append(subplot)
    plt.show()
    return (fig, fig_plots, rows, columns)


def update_image_plots(f, images, title):
    (fig, fig_plots, rows, columns) = f
    if images.shape[-1] == 1:
        images = images.reshape(images.shape[:-1])
    fig.suptitle(title)
    for j in range(len(images[:rows * columns])):
        fig_plots[j].set_data(images[j])
    fig.canvas.draw()
    fig.canvas.flush_events()


def activate_ctrl_c_exit():
    def signal_handler(sig, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def merge_object(initial_obj, additional_obj):
    for property in additional_obj.__dict__:
        setattr(initial_obj, property, getattr(additional_obj, property))

    return initial_obj


def get_image_size(input_shape):
    image_size = input_shape
    if len(input_shape) == 4:
        image_size = image_size[1:]
    if image_size[-1] == 1:
        image_size = image_size[:-1]
    return image_size


def picture_preprocess(images):
    image = np.asarray(images) / 255
    return image
