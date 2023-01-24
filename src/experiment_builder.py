import time

import keras
import numpy as np

import change_measure_utils as ChangeMeasureUtils


class Experiment:
    pass


def get_experiment(params):
    experiment = Experiment()
    experiment.dataset = _get_dataset(params, experiment)
    experiment.model = _get_model(params, experiment)
    experiment.modelv2 = _get_model_v2(params, experiment)
    experiment.coverage = _get_coverage(params, experiment)
    experiment.start_time = time.time()
    experiment.iteration = 0
    experiment.termination_condition = generate_termination_condition(experiment, params)
    experiment.time_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                            210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360]
    return experiment


def generate_termination_condition(experiment, params):
    start_time = experiment.start_time
    time_period = params.time_period

    def termination_condition():
        c2 = time.time() - start_time > time_period
        return c2

    return termination_condition


def _get_dataset(params, experiment):
    if params.dataset == "MNIST":
        from keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape(-1, 28, 28, 1).astype(np.int16)
        test_images = test_images.reshape(-1, 28, 28, 1).astype(np.int16)
        print('xxxxxxxx', np.max(test_images))
    elif params.dataset == "CIFAR10":
        from keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.reshape(-1, 32, 32, 3).astype(np.int16)
        test_images = test_images.reshape(-1, 32, 32, 3).astype(np.int16)
    elif params.dataset == "FM":
        from keras.datasets import fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images.reshape(-1, 28, 28, 1).astype(np.int16)
        test_images = test_images.reshape(-1, 28, 28, 1).astype(np.int16)
    elif params.dataset == "SVHN":
        train_images = np.load('./dataset/svhn_x_train_dc.npy')
        train_labels = np.load('./dataset/svhn_y_train_dc.npy')
        test_images = np.load('./dataset/svhn_x_test_dc.npy')
        test_labels = np.load('./dataset/svhn_y_test_dc.npy')
        train_images = train_images.reshape(-1, 32, 32, 3).astype(np.int16)
        test_images = test_images.reshape(-1, 32, 32, 3).astype(np.int16)
    else:
        raise Exception("Unknown Dataset:" + str(params.dataset))
    return {
        "train_inputs": train_images,
        "train_outputs": train_labels,
        "test_inputs": test_images,
        "test_outputs": test_labels
    }


def _get_model_v2(params, experiment):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if params.model == "LeNet5":
        model = keras.models.load_model('models/mnist_lenet5_supply/train_on_0.2tr/mnist_lenet5_supply_v2.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "LeNet5_adv_cw":
        model = keras.models.load_model('models/mnist_lenet5_advtrain/adv_train/keras_mnist_lenet5_cw_0.9830.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == 'LeNet5_adv_bim':
        model = keras.models.load_model('models/mnist_lenet5_advtrain/adv_train/keras_mnist_lenet5_bim_0.9750.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "vgg16":
        model = keras.models.load_model(
            'models/cifar10_vgg16_0.8tr/train_on_0.2tr/keras_cifar10_vgg16_model.017-0.8788.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "vgg16_adv_cw":
        model = keras.models.load_model('models/cifar10_vgg16_advtrain/adv_train/keras_cifar10_vgg16_cw_0.8800.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == 'vgg16_adv_bim':
        model = keras.models.load_model('models/cifar10_vgg16_advtrain/adv_train/keras_cifar10_vgg16_bim_0.8751.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "resnet18":
        model = keras.models.load_model(
            'models/svhn_resnet18_0.8tr/train_on_0.2tr/keras_svhn_resnet18_model.003-0.9193.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "resnet18_adv_cw":
        model = keras.models.load_model(
            'models/svhn_resnet18_advtrain/adv_train/keras_svhn_resnet18_cw_model.002-0.9201_v2.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "resnet18_adv_bim":
        model = keras.models.load_model(
            'models/svhn_resnet18_advtrain/adv_train/keras_svhn_resnet18_bim_model.005-0.9190_v2.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "Alexnet":
        model = keras.models.load_model('models/fm_alexnet_0.8tr/train_on_0.2tr/keras_fm_alexnet_model.016-0.9034.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet_adv_bim":
        model = keras.models.load_model('models/fm_alexnet_advtrain/adv_train/keras_fm_alexnet_bim.034-0.9096_v2.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet_adv_cw":
        model = keras.models.load_model('models/fm_alexnet_advtrain/adv_train/keras_fm_alexnet_cw.002-0.9187_v2.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "LeNet5_quant":
        import tensorflow as tf
        model = tf.lite.Interpreter(
            model_path="models/mnist_lenet5_advtrain/quantization/mnist_lenet5_converted.tflite")
    elif params.model == "LeNet5_prune":
        import tensorflow
        model = tensorflow.keras.models.load_model('models/mnist_lenet5_advtrain/prune/mnist_prune_0.9812.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "vgg16_quant":
        import tensorflow as tf
        model = tf.lite.Interpreter(
            model_path="models/cifar10_vgg16_advtrain/quantization/cifar10_vgg16_converted_int8.tflite")
    elif params.model == "Alexnet_quant":
        import tensorflow as tf
        model = tf.lite.Interpreter(
            model_path="models/fm_alexnet_advtrain/quantization/fm_alexnet_converted_int8.tflite")
    elif params.model == "resnet18_quant":
        import tensorflow as tf
        model = tf.lite.Interpreter(
            model_path="models/svhn_resnet18_advtrain/quantization/svhn_resnet18_converted_int8.tflite")
    elif params.model == "vgg16_prune":
        import tensorflow
        model = tensorflow.keras.models.load_model('models/cifar10_vgg16_advtrain/prune/cifar10_prune_0.7627.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet_prune":
        import tensorflow
        model = tensorflow.keras.models.load_model('models/fm_alexnet_advtrain/prune/fm_prune_0.9154.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "resnet18_prune":
        import tensorflow
        model = tensorflow.keras.models.load_model('models/svhn_resnet18_advtrain/prune/svhn_prune_0.9100.h5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet_apricot":
        model = keras.models.load_model('models/fm_alexnet_advtrain/apricot/fashion_mnist_apricot_0.929.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "LeNet5_apricot":
        model = keras.models.load_model('models/mnist_lenet5_advtrain/apricot/mnist_apricot_0.9812.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "resnet18_apricot":
        model = keras.models.load_model('models/svhn_resnet18_advtrain/apricot/svhn_apricot_0.921.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "vgg16_apricot":
        model = keras.models.load_model('models/cifar10_vgg16_advtrain/apricot/cifar10_apricot_0.884.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        raise Exception("Unknown Model:" + str(params.model))

    return model


def _get_model(params, experiment):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    if params.model == "LeNet5":

        model = keras.models.load_model('models/mnist_lenet5_supply/mnist_lenet5_supply_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "LeNet5_adv_cw" or params.model == "LeNet5_adv_bim" or params.model == "LeNet5_quant" or params.model == "LeNet5_prune" or params.model == "LeNet5_apricot":

        model = keras.models.load_model(
            'models/mnist_lenet5_advtrain/keras_Mon-Dec-27-09-39-09-2021.model.011-0.9807.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "vgg16":

        model = keras.models.load_model(
            'models/cifar10_vgg16_0.8tr/keras_Sat-Oct-30-01-36-17-2021.model.094-0.8767.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "vgg16_adv_cw" or params.model == "vgg16_adv_bim" or params.model == "vgg16_quant" or params.model == "vgg16_prune" or params.model == "vgg16_apricot":

        model = keras.models.load_model(
            'models/cifar10_vgg16_advtrain/keras_Wed-Mar-24-17-55-44-2021.model.135-0.8792.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif params.model == "resnet18":
        model = keras.models.load_model(
            'models/svhn_resnet18_0.8tr/keras_Tue-Dec-28-20-09-09-2021.model.004-0.8885_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "resnet18_adv_cw" or params.model == "resnet18_adv_bim" or params.model == "resnet18_quant" or params.model == "resnet18_prune" or params.model == "resnet18_apricot":
        model = keras.models.load_model('models/svhn_resnet18_advtrain/keras_svhn_resnet18_model.006-0.9205_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet":
        model = keras.models.load_model('models/fm_alexnet_0.8tr/keras_fm_alexnet.019-0.8933_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Alexnet_adv_cw" or params.model == "Alexnet_adv_bim" or params.model == "Alexnet_quant" or params.model == "Alexnet_prune" or params.model == "Alexnet_apricot":
        model = keras.models.load_model('models/fm_alexnet_advtrain/keras_fm_alexnet.098-0.9170_v1.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    else:
        raise Exception("Unknown Model:" + str(params.model))

    return model


def _get_coverage(params, experiment):
    def input_scaler(test_inputs):
        model_lower_bound = params.model_input_scale[0]
        model_upper_bound = params.model_input_scale[1]
        input_lower_bound = params.input_lower_limit
        input_upper_bound = params.input_upper_limit
        scaled_input = (test_inputs - input_lower_bound) / (input_upper_bound - input_lower_bound)
        scaled_input = scaled_input * (model_upper_bound - model_lower_bound) + model_lower_bound
        return scaled_input

    if params.coverage == "neuron":
        from coverages.neuron_cov import NeuronCoverage
        # TODO: Skip layers should be determined autoamtically
        coverage = NeuronCoverage(params, experiment.modelv2, skip_layers=params.skip_layers)  # 0:input, 5:flatten
        print(params.skip_layers)
    elif params.coverage == "kmn" or params.coverage == "nbc" or params.coverage == "snac":
        from coverages.kmn import DeepGaugePercentCoverage
        major_func_file = _get_kmnc_profile(params)
        print('params.skip_layers', params.skip_layers)
        import src.utility as ImageUtils
        train_inputs_scaled = ImageUtils.picture_preprocess(experiment.dataset["train_inputs"])
        coverage = DeepGaugePercentCoverage(params, experiment.modelv2, getattr(params, 'kmn_k', 1000),
                                            train_inputs_scaled,
                                            skip_layers=ChangeMeasureUtils.get_skiped_layer(experiment.modelv2),
                                            coverage_name=params.coverage,
                                            major_func_regions=major_func_file)  # 0:input, 5:flatten

    elif params.coverage == "change":
        from coverages.change_scorer import ChangeScorer
        # TODO: Skip layers should be determined autoamtically
        import src.utility as ImageUtils
        test_inputs = ImageUtils.picture_preprocess(experiment.dataset['test_inputs'])
        coverage = ChangeScorer(params, experiment.model, experiment.modelv2, test_inputs, threshold=0.5,
                                skip_layers=ChangeMeasureUtils.get_skiped_layer(experiment.model))  # 0:input, 5:flatten

    else:
        raise Exception("Unknown Coverage" + str(params.coverage))


    return coverage


def _get_kmnc_profile(params):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from coverages.utils import load_major_func_regions
    if params.model == "LeNet5":
        path = 'kmnc_profile/mnist_lenet5_supply'
    elif params.model == "LeNet5_adv_cw":
        path = 'kmnc_profile/mnist_lenet5_adv_cw'
    elif params.model == "LeNet5_adv_bim":
        path = 'kmnc_profile/mnist_lenet5_adv_bim'
    elif params.model == "LeNet5_apricot":
        path = 'kmnc_profile/mnist_lenet_apricot'
    elif params.model == "vgg16":
        path = 'kmnc_profile/cifar10_vgg16_p80'
    elif params.model == "vgg16_adv_cw":
        path = 'kmnc_profile/cifar10_vgg16_cw'
    elif params.model == "vgg16_adv_bim":
        path = 'kmnc_profile/cifar10_vgg16_bim'
    elif params.model == "vgg16_apricot":
        path = 'kmnc_profile/cifar10_vgg16_apricot'
    elif params.model == "LeNet5_prune":
        path = 'kmnc_profile/mnist_lenet_prune'
    elif params.model == "Alexnet":
        path = 'kmnc_profile/fm_alexnet_p80'
    elif params.model == "Alexnet_adv_bim":
        path = 'kmnc_profile/fm_alexnet_bim'
    elif params.model == "Alexnet_adv_cw":
        path = 'kmnc_profile/fm_alexnet_cw'
    elif params.model == "Alexnet_prune":
        path = 'kmnc_profile/fm_alexnet_prune'
    elif params.model == "Alexnet_apricot":
        path = 'kmnc_profile/fm_alexnet_apricot'
    elif params.model == "resnet18":
        path = 'kmnc_profile/svhn_resnet18_p80'
    elif params.model == "resnet18_adv_cw":
        path = 'kmnc_profile/svhn_resnet18_adv_cw'
    elif params.model == "resnet18_adv_bim":
        path = 'kmnc_profile/svhn_resnet18_adv_bim'
    elif params.model == "resnet18_apricot":
        path = 'kmnc_profile/svhn_resnet18_apricot'
    elif params.model == "vgg16_prune":
        path = 'kmnc_profile/cifar10_vgg16_prune'
    elif params.model == "resnet18_prune":
        path = 'kmnc_profile/svhn_resnet18_prune'
    elif params.model == "LeNet5_quant":
        return None
    else:
        raise Exception("Unknown Model:" + str(params.model))

    return load_major_func_regions(path)
