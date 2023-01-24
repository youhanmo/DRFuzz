from collections import defaultdict
import datetime
import numpy as np
import os
import argparse
import pickle
import tensorflow.keras as keras


def scale(intermediate_layer_output, args, rmax=1, rmin=0):
    """standardized"""
    try:
        X_scaled = intermediate_layer_output.reshape(args.b, -1)
        x_max = X_scaled.max(axis=-1)
        x_min = X_scaled.min(axis=-1)
        x_sub = x_max - x_min
        X_scaled = X_scaled.T
        X_scaled = (X_scaled - x_min) / x_sub
        X_scaled = X_scaled.T

    except RuntimeWarning:
        print(X_scaled)

    return X_scaled


def get_batch(input_data, batch_size=100):
    print(len(input_data))
    batch_num = int(np.ceil(len(input_data) / batch_size))
    for i in range(batch_num):
        yield input_data[i * batch_size:(i + 1) * batch_size]


def get_boundry(model, input_data, args, intermediate_layer_model, boundry_dict=None):
    if boundry_dict == None:
        layers = [layer for layer in model.layers if
                  'flatten' not in layer.name and 'input' not in layer.name and 'bn' not in layer.name]
        print(layers)
        boundry_dict = defaultdict(list)
        for i, layer in enumerate(layers):
            for index in range(layer.output_shape[-1]):
                boundry_dict[(i, index)] = [None, None]
    start = datetime.datetime.now()

    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output, args)
        scaled = scaled.reshape(intermediate_layer_output.shape)
        for num_neuron in range(intermediate_layer_output.shape[-1]):
            mean = np.mean(scaled[..., num_neuron].reshape(args.b, -1), axis=-1)
            min = mean.min()
            max = mean.max()

            if (boundry_dict[(i, num_neuron)][0] is None) or (min < boundry_dict[(i, num_neuron)][0]):
                boundry_dict[(i, num_neuron)][0] = min

            if (boundry_dict[(i, num_neuron)][1] is None) or (max > boundry_dict[(i, num_neuron)][1]):
                boundry_dict[(i, num_neuron)][1] = max
    end = (datetime.datetime.now() - start)
    print(end)

    return boundry_dict


if __name__ == "__main__":
    basedir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="fm")
    parser.add_argument("--m", help="line", type=str, default='lenet5')
    parser.add_argument("--b", help="batch_num", type=int, default=1)
    args = parser.parse_args()
    if args.d == "mnist":
        model = keras.models.load_model('models/mnist_lenet5_advtrain/apricot/mnist_apricot_0.9812.hdf5')
        (x_train, y_train), (x_test1, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test1 = x_test1.reshape(-1, 28, 28, 1)
        x_train = (x_train / 255.0)
        x_test1 = (x_test1 / 255.0)
        x_train = x_train
        y_train = y_train
        layers = [layer for layer in model.layers if
                  'flatten' not in layer.name and 'input' not in layer.name]
        layer_names = [layer.name
                       for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name]
        print(layer_names)
        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=[model.get_layer(layer_name).output
                                                        for layer_name in layer_names])

        boundry_dict = get_boundry(model, input_data=x_train, args=args,
                                   intermediate_layer_model=intermediate_layer_model, boundry_dict=None)
        with open("mnist_lenet_apricot", "wb")as file:
            pickle.dump(boundry_dict, file)

    if args.d == "cifar10":
        model = keras.models.load_model('models/cifar10_vgg16_advtrain/prune/cifar10_prune_0.7627.h5')
        (x_train, y_train), (x_test1, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.reshape(-1, 32, 32, 3)
        y_train = y_train
        x_test1 = x_test1.reshape(-1, 32, 32, 3)
        x_train = (x_train / 255.0)
        x_test1 = (x_test1 / 255.0)
        x_train = x_train
        y_train = y_train
        layers = [layer for layer in model.layers if
                  'flatten' not in layer.name and 'input' not in layer.name and 'bn' not in layer.name]
        layer_names = [layer.name
                       for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name and 'bn' not in layer.name]
        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=[model.get_layer(layer_name).output
                                                        for layer_name in layer_names])
        boundry_dict = None
        for b_n, onbatch in enumerate(get_batch(x_train, args.b)):
            boundry_dict = get_boundry(model=model, input_data=onbatch, args=args, boundry_dict=boundry_dict,
                                       intermediate_layer_model=intermediate_layer_model)

        with open("cifar10_vgg16_prune", "wb")as file:
            pickle.dump(boundry_dict, file)
    if args.d == "fm":
        model = keras.models.load_model('models/fm_alexnet_advtrain/prune/fm_prune_0.9154.h5')
        (x_train, y_train), (x_test1, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test1 = x_test1.reshape(-1, 28, 28, 1)
        x_train = (x_train / 255.0)
        x_test1 = (x_test1 / 255.0)
        import math

        len_x = math.ceil(len(x_train) * 0.8)
        x_train = x_train
        y_train = y_train
        layers = [layer for layer in model.layers if
                  'flatten' not in layer.name and 'Input' not in layer.name and 'bn' not in layer.name]
        layer_names = [layer.name
                       for layer in model.layers if
                       'flatten' not in layer.name and 'Input' not in layer.name and 'bn' not in layer.name]
        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=[model.get_layer(layer_name).output
                                                        for layer_name in layer_names])
        boundry_dict = None
        for b_n, onbatch in enumerate(get_batch(x_train, args.b)):
            boundry_dict = get_boundry(model=model, input_data=onbatch, args=args, boundry_dict=boundry_dict,
                                       intermediate_layer_model=intermediate_layer_model)

        with open("fm_alexnet_prune", "wb")as file:
            pickle.dump(boundry_dict, file)
    if args.d == "svhn":
        model = keras.models.load_model('models/svhn_resnet18_advtrain/prune/svhn_prune_0.9100.h5')
        x_train = np.load('models/svhn_resnet18_0.8tr/svhn_x_train_dc.npy')
        y_train = np.load('models/svhn_resnet18_0.8tr/svhn_y_train_dc.npy')
        x_test1 = np.load('models/svhn_resnet18_0.8tr/svhn_x_test_dc.npy')
        y_test1 = np.load('models/svhn_resnet18_0.8tr/svhn_y_test_dc.npy')
        x_train = x_train.reshape(-1, 32, 32, 3)
        x_test1 = x_test1.reshape(-1, 32, 32, 3)
        x_train = (x_train / 255.0)
        x_test1 = (x_test1 / 255.0)
        import math

        len_x = math.ceil(len(x_train) * 0.8)
        x_train = x_train
        y_train = y_train
        print(x_train)
        layers = [layer for layer in model.layers if
                  'flatten' not in layer.name and 'input' not in layer.name and 'bn' not in layer.name]
        layer_names = [layer.name
                       for layer in model.layers if
                       'flatten' not in layer.name and 'input' not in layer.name and 'bn' not in layer.name]
        intermediate_layer_model = keras.Model(inputs=model.input,
                                               outputs=[model.get_layer(layer_name).output
                                                        for layer_name in layer_names])
        boundry_dict = None
        for b_n, onbatch in enumerate(get_batch(x_train, args.b)):
            boundry_dict = get_boundry(model=model, input_data=onbatch, args=args, boundry_dict=boundry_dict,
                                       intermediate_layer_model=intermediate_layer_model)

        with open("svhn_resnet18_prune", "wb")as file:
            pickle.dump(boundry_dict, file)
