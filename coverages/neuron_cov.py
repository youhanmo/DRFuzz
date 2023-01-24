import sys

sys.path.append('../')
import copy
import numpy as np
from coverages.utils import get_layer_outs_new, percent_str, percent
from collections import defaultdict
import change_measure_utils as ChangeMeasureUtils


def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def normalization_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    return X_std


from coverages.coverage import AbstractCoverage


class NeuronCoverage(AbstractCoverage):
    def __init__(self, params, model, scaler=default_scale, threshold=0.75, skip_layers=None):
        self.params = params
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.skip_layers = ChangeMeasureUtils.get_skiped_layer(self.model)

        self.start = 0
        self.bytearray_len = 1
        self.layer_neuron_num = []
        self.layer_start_index = []
        print('skipped layers:', self.skip_layers)
        num = 0
        layer_id = 0
        for layer in self.model.layers:
            if layer_id not in self.skip_layers:
                self.layer_start_index.append(num)
                self.layer_neuron_num.append(int(layer.output.shape[-1]))
                num += int(layer.output.shape[-1] * self.bytearray_len)
            layer_id += 1

        self.total_neuron_num = np.sum(self.layer_neuron_num)
        self.activation_table = np.zeros(self.total_neuron_num, dtype=np.uint8)

    def calc_reward(self, activation_table):
        activation_values = np.array(list(activation_table.values()))
        covered_positions = activation_values == 1
        covered = np.sum(covered_positions)
        reward = covered
        return reward, covered

    def get_measure_state(self):
        return [self.activation_table]

    def set_measure_state(self, state):
        self.activation_table = state[0]

    def reset_measure_state(self):
        self.activation_table = defaultdict(float)


    def test(self, test_inputs):
        ptr = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        outs = get_layer_outs_new(self.params, self.model, test_inputs, self.skip_layers)
        for layer_index, layer_out in enumerate(outs):
            seed_id = 0
            for out_for_input in layer_out:
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    if np.mean(out_for_input[..., neuron_index]) > self.threshold:
                        id = self.start + self.layer_start_index[layer_index] + neuron_index * self.bytearray_len + 0
                        ptr[seed_id][id] = 1
                seed_id += 1
        return ptr

    def initial_seed_list(self, test_inputs):
        ptr = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        print(ptr.shape)
        outs = get_layer_outs_new(self.params, self.model, test_inputs, self.skip_layers)
        for layer_index, layer_out in enumerate(outs):
            seed_id = 0
            for out_for_input in layer_out:
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    if np.mean(out_for_input[..., neuron_index]) > self.threshold:
                        id = self.start + self.layer_start_index[layer_index] + neuron_index * self.bytearray_len + 0
                        ptr[seed_id][id] = 1
                seed_id += 1
        for ptr_seed in ptr:
            self.activation_table = self.activation_table | ptr_seed
        return ptr

    def get_current_coverage(self):
        covered_positions = self.activation_table > 0
        covered = np.sum(covered_positions)
        return percent(covered, len(self.activation_table))
