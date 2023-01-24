import importlib

from src.DrFuzz import DrFuzz
from src.deephunter import DeepHunter
from src.experiment_builder import get_experiment
from src.utility import merge_object


def load_params(params):
    for params_set in params.params_set:
        m = importlib.import_module("params." + params_set)
        print(m)
        new_params = getattr(m, params_set)
        params = merge_object(params, new_params)
    return params


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Experiments Script For DeepReFuzz")
    parser.add_argument("--params_set", nargs='*', type=str, default=["alexnet", "fm", "kmn", "deephunter"],
                        help="see params folder")
    parser.add_argument("--dataset", type=str, default="FM", choices=["MNIST", "CIFAR10", "FM", "SVHN"])
    parser.add_argument("--model", type=str, default="Alexnet_prune", choices=["vgg16", "resnet18", "LeNet5", "Alexnet",
                                                                               "vgg16_adv_bim", "vgg16_adv_cw",
                                                                               "vgg16_apricot",
                                                                               "LeNet5_adv_bim", "LeNet5_adv_cw",
                                                                               "LeNet5_apricot",
                                                                               "Alexnet_adv_bim", "Alexnet_adv_cw",
                                                                               "Alexnet_apricot",
                                                                               "resnet18_adv_bim", "resnet18_adv_cw",
                                                                               "resnet18_apricot",
                                                                               "LeNet5_quant", "vgg16_quant",
                                                                               "resnet18_quant", "Alexnet_quant",
                                                                               "LeNet5_prune", "vgg16_prune",
                                                                               "resnet18_prune", "Alexnet_prune"])

    parser.add_argument("--coverage", type=str, default="kmn", choices=["change", "neuron", "kmn", "nbc", "snac"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--time", type=int, default=1440)
    params = parser.parse_args()

    print(params)
    params = load_params(params)
    params.time_minutes = params.time
    params.time_period = params.time_minutes * 60
    experiment = get_experiment(params)
    experiment.time_list = [i * 30 for i in range(1, params.time // 30 + 1 + 1)]

    if params.framework_name == 'drfuzz':
        dh = DrFuzz(params, experiment)
    elif params.framework_name == 'deephunter':
        dh = DeepHunter(params, experiment)
    else:
        raise Exception("No Framework Provided")

    import numpy as np
    import os

    print(os.path.abspath(__file__))
    experiment_dir = str(params.coverage)
    dir_name = 'experiment_' + str(params.framework_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if not os.path.exists(os.path.join(dir_name, experiment_dir)):
        os.mkdir(os.path.join(dir_name, experiment_dir))

    both_fail, regression_faults, weaken = dh.run()

    np.save(os.path.join(dir_name, experiment_dir, "bothfail.npy"), np.asarray(both_fail))
    np.save(os.path.join(dir_name, experiment_dir, "regression_faults.npy"), np.asarray(regression_faults))
    np.save(os.path.join(dir_name, experiment_dir, "weaken.npy"), np.asarray(weaken))

    print('TOTAL BOTH:', len(both_fail))
    print('TOTAL REGRESSION:', len(regression_faults))
    print('TOTAL WEAKEN:', len(weaken))
    print('CORPUS', dh.corpus)
    np.save(os.path.join(dir_name, experiment_dir, "corpus.npy"), np.asarray(dh.corpus_list))
    print('ITERATION', dh.experiment.iteration)

    if params.framework_name == 'drfuzz':
        print('SCORE', dh.experiment.coverage.get_failure_type())
    elif params.framework_name == 'deephunter':
        print('SCORE', dh.experiment.coverage.get_current_coverage())

    import matplotlib.pyplot as plt

    plt.imshow(regression_faults[0].input)
    plt.show()