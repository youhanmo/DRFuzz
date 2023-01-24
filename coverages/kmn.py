# -*- coding: utf-8 -*-
import numpy as np
from coverages.utils import get_layer_outs, get_layer_outs_new, calc_major_func_regions, percent_str, percent
from math import floor
import math
from coverages.coverage import AbstractCoverage


def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


class DeepGaugePercentCoverage(AbstractCoverage):
    def __init__(self, params, model, k=1000, train_inputs=None, major_func_regions=None, skip_layers=None,
                 coverage_name="kmn", scaler=default_scale):
        self.coverage_name = coverage_name
        self.params = params
        self.model = model
        self.k = k
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.scaler = scaler
        self.start = 0
        self.bytearray_len = k
        self.layer_start_index = []
        self.layer_neuron_num = []
        section = 0
        layer_id = 0
        num = 0
        for layer in self.model.layers:
            if layer_id not in skip_layers:
                self.layer_start_index.append(section)
                print(layer.name)
                print(layer.output.shape)
                self.layer_neuron_num.append(int(layer.output.shape[-1]))
                num += int(layer.output.shape[-1])
                section += int(layer.output.shape[-1] * self.bytearray_len)
            layer_id += 1
        print(self.layer_start_index)
        self.total_neuron_num = np.sum(self.layer_neuron_num)
        self.activation_table_by_section = np.zeros(self.total_neuron_num * self.k, dtype=np.uint8)
        self.activation_table_snac = np.zeros(self.total_neuron_num, dtype=np.uint8)
        self.activation_table_nbc = np.zeros(self.total_neuron_num * 2, dtype=np.uint8)

        if major_func_regions is None:
            if train_inputs is None:
                raise ValueError("Training inputs must be provided when major function regions are not given")

            self.major_func_regions = calc_major_func_regions(params, model, train_inputs, skip_layers)
        else:
            self.major_func_regions = major_func_regions

    def get_measure_state(self):
        if self.coverage_name == 'kmn':
            return [self.activation_table_by_section]
        elif self.coverage_name == 'snac':
            return [self.activation_table_snac]
        elif self.coverage_name == 'nbc':
            return [self.activation_table_nbc]

    def calc_reward(self, activation_table):
        activation_values = np.array(list(activation_table.values()))
        covered_positions = activation_values == 1
        covered = np.sum(covered_positions)
        reward = covered
        return reward, covered

    def set_measure_state(self, state):
        if self.coverage_name == 'kmn':
            self.activation_table_by_section = state[0]
        elif self.coverage_name == 'snac':
            self.activation_table_snac = state[0]
        elif self.coverage_name == 'nbc':
            self.activation_table_nbc = state[0]

    def reset_measure_state(self):
        self.activation_table_by_section, self.upper_activation_table, self.lower_activation_table = {}, {}, {}

    def initial_seed_list(self, test_inputs):
        ptr_section_by_section = np.tile(np.zeros(self.total_neuron_num * self.k, dtype=np.uint8),
                                         (len(test_inputs), 1))
        ptr_section_higher = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        ptr_section_lower = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        print(ptr_section_by_section.shape)
        outs = get_layer_outs_new(self.params, self.model, test_inputs, skip=self.skip_layers)
        print('neuron_num', self.total_neuron_num)
        print(len(self.major_func_regions))
        for layer_index, layer_out in enumerate(outs):
            seed_id = 0
            for out_for_input in layer_out:
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    neuron_out = np.mean(out_for_input[..., neuron_index])
                    global_neuron_index = (layer_index, neuron_index)

                    neuron_low = self.major_func_regions[global_neuron_index][0]
                    neuron_high = self.major_func_regions[global_neuron_index][1]
                    section_length = (neuron_high - neuron_low) / self.k
                    section_index = floor((neuron_out - neuron_low) / section_length) if section_length > 0 else 0

                    nid = self.start + (self.layer_start_index[layer_index] // self.bytearray_len) + neuron_index
                    if section_index > self.k - 1:
                        sid = self.start + self.layer_start_index[
                            layer_index] + neuron_index * self.bytearray_len + self.k - 1
                    else:
                        sid = self.start + self.layer_start_index[
                            layer_index] + neuron_index * self.bytearray_len + section_index

                    if sid >= 0:
                        ptr_section_by_section[seed_id][sid] = 1

                    if neuron_out < neuron_low:
                        ptr_section_lower[seed_id][nid] = 1
                    elif neuron_out > neuron_high:
                        ptr_section_higher[seed_id][nid] = 1

                seed_id += 1

        ptr_nbc = np.concatenate((ptr_section_lower, ptr_section_higher), axis=1)
        for i in range(len(ptr_section_by_section)):
            self.activation_table_by_section = self.activation_table_by_section | ptr_section_by_section[i]
            self.activation_table_snac = self.activation_table_snac | ptr_section_higher[i]
            self.activation_table_nbc = self.activation_table_nbc | ptr_nbc[i]

        multisection_activated = np.sum(self.activation_table_by_section)
        snac_activated = np.sum(self.activation_table_snac)
        nbc_activated = np.sum(self.activation_table_nbc)
        print('KMNC', percent_str(multisection_activated, self.k * self.total_neuron_num))
        print('NBC', percent_str(nbc_activated, self.total_neuron_num * 2))
        print('SNAC', percent_str(snac_activated, self.total_neuron_num))

    def get_current_coverage(self, with_implicit_reward=False):
        multisection_activated = np.sum(self.activation_table_by_section)
        snac_activated = np.sum(self.activation_table_snac)
        nbc_activated = np.sum(self.activation_table_nbc)
        total = self.total_neuron_num
        if self.coverage_name == "kmn":
            if multisection_activated == 0:
                return 0
            return percent(multisection_activated, self.k * total)
        elif self.coverage_name == "nbc":
            if nbc_activated == 0:
                return 0
            return percent(nbc_activated, 2 * total)
        elif self.coverage_name == "snac":
            if snac_activated == 0:
                return 0
            return percent(snac_activated, total)
        else:
            raise Exception("Unknown coverage: " + str(self.coverage_name))

    def test(self, test_inputs, with_implicit_reward=False):
        ptr_section_by_section = np.tile(np.zeros(self.total_neuron_num * self.k, dtype=np.uint8),
                                         (len(test_inputs), 1))
        ptr_section_higher = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        ptr_section_lower = np.tile(np.zeros(self.total_neuron_num, dtype=np.uint8), (len(test_inputs), 1))
        outs = get_layer_outs_new(self.params, self.model, test_inputs, skip=self.skip_layers)
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            seed_id = 0
            for out_for_input in layer_out:
                out_for_input = self.scaler(out_for_input)
                for neuron_index in range(out_for_input.shape[-1]):
                    neuron_out = np.mean(out_for_input[..., neuron_index])
                    global_neuron_index = (layer_index, neuron_index)
                    neuron_low = self.major_func_regions[global_neuron_index][0]
                    neuron_high = self.major_func_regions[global_neuron_index][1]

                    section_length = (neuron_high - neuron_low) / self.k
                    nid = self.start + (self.layer_start_index[layer_index] // self.bytearray_len) + neuron_index

                    if neuron_high == neuron_low:
                        continue
                    if neuron_out < neuron_low:
                        ptr_section_lower[seed_id][nid] = 1
                        continue
                    elif neuron_out > neuron_high:
                        ptr_section_higher[seed_id][nid] = 1
                        continue
                    if math.isnan((neuron_out - neuron_low) / section_length):
                        section_index = 0
                    else:
                        section_index = floor((neuron_out - neuron_low) / section_length) if section_length > 0 else 0

                    try:
                        if section_index > self.k - 1:
                            sid = self.start + self.layer_start_index[
                                layer_index] + neuron_index * self.bytearray_len + self.k - 1
                        else:
                            sid = self.start + self.layer_start_index[
                                layer_index] + neuron_index * self.bytearray_len + section_index
                        if sid >= 0:
                            ptr_section_by_section[seed_id][sid] = 1
                    except Exception as e:
                        print(e)
                        print(neuron_out)
                        print(neuron_low)
                        print(neuron_high)
                        print(self.layer_start_index[layer_index])
                        print(neuron_index)
                        print(section_index)

                seed_id += 1

        if self.coverage_name == 'kmn':
            return ptr_section_by_section
        elif self.coverage_name == 'nbc':
            ptr_nbc = np.concatenate((ptr_section_lower, ptr_section_higher), axis=1)
            return ptr_nbc
        elif self.coverage_name == 'snac':
            return ptr_section_higher


def measure_k_multisection_cov(params, model, test_inputs, k, train_inputs=None, major_func_regions=None, skip=None,
                               outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_inputs, skip=skip)

    if major_func_regions is None:
        if train_inputs is None:
            raise ValueError("Training inputs must be provided when major function regions are not given")

        major_func_regions = calc_major_func_regions(params, model, train_inputs, skip)

    activation_table_by_section, upper_activation_table, lower_activation_table = {}, {}, {}
    neuron_set = set()

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        print(layer_index)
        for out_for_input in layer_out[0]:  # out_for_input is output of layer for single input
            for neuron_index in range(out_for_input.shape[-1]):
                neuron_out = np.mean(out_for_input[..., neuron_index])
                global_neuron_index = (layer_index, neuron_index)

                neuron_set.add(global_neuron_index)

                neuron_low = major_func_regions[layer_index][0][neuron_index]
                neuron_high = major_func_regions[layer_index][1][neuron_index]
                section_length = (neuron_high - neuron_low) / k
                section_index = floor((neuron_out - neuron_low) / section_length) if section_length > 0 else 0

                activation_table_by_section[(global_neuron_index, section_index)] = True

                if neuron_out < neuron_low:
                    lower_activation_table[global_neuron_index] = True
                elif neuron_out > neuron_high:
                    upper_activation_table[global_neuron_index] = True

    multisection_activated = len(activation_table_by_section.keys())
    lower_activated = len(lower_activation_table.keys())
    upper_activated = len(upper_activation_table.keys())

    total = len(neuron_set)

    return (percent_str(multisection_activated, k * total),  # kmn
            multisection_activated,
            percent_str(upper_activated + lower_activated, 2 * total),  # nbc
            percent_str(upper_activated, total),  # snac
            lower_activated, upper_activated, total,
            multisection_activated, upper_activated, lower_activated, total, outs)
