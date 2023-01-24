import numpy as np

episilon = 1e-6


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + episilon)


def normalization_by_line(data):
    Zmax, Zmin = data.max(axis=0), data.min(axis=0)
    result = (data - Zmin) / ((Zmax - Zmin) + episilon)
    return result


def get_skiped_layer(model_m1):
    need_skip_layer_id = []
    layer_name = []
    for layer_id in range(len(model_m1.layers)):
        layer_m1 = model_m1.layers[layer_id].name
        if 'flatten' in layer_m1 or 'bn' in layer_m1 or 'input' in layer_m1:
            need_skip_layer_id.append(layer_id)
            layer_name.append(layer_m1)
    print('need_skip_layer_id', need_skip_layer_id)
    print(layer_name)
    return need_skip_layer_id
