import os
import time

import numpy as np


class INFO:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, i):
        _i = str(i)
        if _i in self.dict:
            return self.dict[_i]
        else:
            I0, I0_new, state = np.copy(i), np.copy(i), 0
            return I0, I0_new, state

    def __setitem__(self, i, s):
        _i = str(i)
        self.dict[_i] = s
        return self.dict[_i]


def testcaselist2labels(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].label))
    new_input = np.asarray(new_input)
    return new_input


class TestCase:
    def __init__(self, input, ground_truth, source_id):
        self.input = input
        self.label = ground_truth
        self.source_id = source_id


def testcaselist2nparray(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].input))
    new_input = np.asarray(new_input)
    return new_input


def testcaselist2sourceid(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].source_id))
    new_input = np.asarray(new_input)
    return new_input


def timing(start_time, duriation):
    MIN = 60
    duriation_sec = duriation * MIN
    now = time.time()
    return now - start_time > duriation_sec


import src.utility as ImageUtils


class DeepHunter:
    def __init__(self, params, experiment):
        self.params = params
        self.experiment = experiment
        self.info = INFO()
        self.last_coverage_state = None
        self.input_shape = params.input_shape
        self.corpus = 0
        self.corpus_list = []
        self.both_failure_case = []
        self.regression_faults = []
        self.weaken_case = []
        print('params')
        print(params)

    def run(self):
        starttime = time.time()
        I_input = self.experiment.dataset["test_inputs"]
        I_label = self.experiment.dataset["test_outputs"]
        model_v2 = self.experiment.modelv2
        result = np.argmax(model_v2.predict(ImageUtils.picture_preprocess(I_input)), axis=1)
        I_label = np.squeeze(I_label)
        result = np.squeeze(result)
        good_idx = np.where(I_label == result)
        I_list = []
        for i in range(len(I_label)):
            if i in good_idx[0]:
                tc = TestCase(I_input[i], I_label[i], i)
                I_list.append(tc)
                self.corpus += 1
                self.corpus_list.append(tc)
        print('seed num', self.corpus)
        initial_seed_preprocessed = ImageUtils.picture_preprocess(testcaselist2nparray(I_list))
        for isp in initial_seed_preprocessed:
            xisp = isp.reshape(-1, isp.shape[0], isp.shape[1], isp.shape[2])
            self.experiment.coverage.initial_seed_list(xisp)
        print(self.experiment.coverage.get_current_coverage())

        time_list = self.experiment.time_list
        T = self.Preprocess(I_list)
        B, B_id = self.SelectNext(T)
        self.experiment.iteration = 0
        while not self.experiment.termination_condition():
            S = B

            count_both_fail = []
            count_regression = []
            count_weaken = []
            if len(time_list) > 0:
                if timing(starttime, time_list[0]):
                    f = open('txt_1.txt', 'a')
                    f.writelines('\nAT:' + str(time_list[0]))
                    f.writelines('\nTOTAL BOTH:' + str(len(self.both_failure_case)))
                    f.writelines('\nTOTAL REGRESSION:' + str(len(self.regression_faults)))
                    f.writelines('\nTOTAL WEAKEN:' + str(len(self.weaken_case)))
                    f.writelines('\nITERATION:' + str(self.experiment.iteration))
                    f.writelines('\nCORPUS:' + str(self.corpus))
                    f.writelines('\nSCORE:' + str(self.experiment.coverage.get_current_coverage()))
                    f.close()
                    print('AT ' + str(time_list[0]) + ' MIN')
                    print('TOTAL BOTH', len(self.both_failure_case))
                    print('TOTAL REGRESSION', len(self.regression_faults))
                    print('TOTAL WEAKEN', len(self.weaken_case))
                    print('CORPUS', self.corpus)
                    print('ITERATION', self.experiment.iteration)
                    print('SCORE', self.experiment.coverage.get_current_coverage())

                    experiment_dir = str(self.params.coverage)
                    dir_name = 'experiment_' + str(self.params.framework_name)
                    if not os.path.exists(os.path.join(dir_name, experiment_dir)):
                        os.mkdir(os.path.join(dir_name, experiment_dir))
                    np.save(os.path.join(dir_name, experiment_dir, str(time_list[0]) + '_rf'), self.regression_faults)
                    np.save(os.path.join(dir_name, experiment_dir, str(time_list[0]) + '_bf'), self.both_failure_case)
                    time_list.remove(time_list[0])

            B_new = []
            Mutants = []
            for s_i in range(len(S)):
                I = S[s_i]
                for i in range(1, 20 + 1):
                    I_new = self.Mutate(I)
                    if I_new != None and self.isChanged(I, I_new):
                        Mutants.append(I_new)

            if len(Mutants) > 0:
                bflist, rflist, wklist, hwklist, B_new, dangerous_source_id = self.isFailedTestList(Mutants)
                self.both_failure_case.extend(bflist)
                count_both_fail.extend(bflist)
                self.regression_faults.extend(rflist)
                count_regression.extend(rflist)
                self.weaken_case.extend(wklist)
                count_weaken.extend(wklist)

            if len(B_new) > 0:
                cov, activation_dicts = self.Predict(B_new)

                for t in range(len(B_new)):
                    new_activation_values = activation_dicts[t]
                    now_activation_table = self.last_coverage_state[0]
                    x = np.bitwise_or(new_activation_values, now_activation_table)
                    new_covered_positions = x == 1
                    now_covered_positions = now_activation_table == 1

                    if np.sum(new_covered_positions) - np.sum(now_covered_positions) > 0:
                        self.experiment.coverage.step(ImageUtils.picture_preprocess(B_new[t].input), update_state=True,
                                                      coverage_state=[new_covered_positions])
                        self.last_coverage_state = [new_covered_positions]
                        B_c, Bs = T
                        B_c += [0]
                        selected_test_case_list_i = np.asarray(B_new)[t:t + 1]
                        Bs += [selected_test_case_list_i]
                        self.corpus += 1
                        self.corpus_list.append(selected_test_case_list_i[0])
                        self.BatchPrioritize(T, B_id)

            B, B_id = self.SelectNext(T)
            print('iteration:', self.experiment.iteration)
            self.experiment.iteration += 1
        return self.both_failure_case, self.regression_faults, self.weaken_case

    def Preprocess(self, I):
        _I = np.random.permutation(I)
        Bs = np.array_split(_I, range(self.params.batch1, len(_I), self.params.batch1))
        return list(np.zeros(len(Bs))), Bs

    def calc_priority(self, B_ci):
        if B_ci < (1 - self.params.p_min) * self.params.gamma:
            return 1 - B_ci / self.params.gamma
        else:
            return self.params.p_min

    def SelectNext(self, T):
        B_c, Bs = T
        B_p = [self.calc_priority(B_c[i]) for i in range(len(B_c))]
        c = np.random.choice(len(Bs), p=B_p / np.sum(B_p))
        return Bs[c], c

    def Sample(self, B):
        c = np.random.choice(len(B), size=min(len(B), self.params.batch2), replace=False)
        return B[c]

    def PowerSchedule(self, S, K):
        potentials = []
        for i in range(len(S)):
            I = S[i].input
            I0, I0_new, state = self.info[I]
            p = self.params.beta * 255 * np.sum(I > 0) - np.sum(np.abs(I - I0_new))
            potentials.append(p)
        potentials = np.array(potentials) / np.sum(potentials)

        def Ps(I_id):
            p = potentials[I_id]
            return int(np.ceil(p * K))

        return Ps

    def isFailedTest(self, I_new):
        model_v1 = self.experiment.model
        I_new_input = I_new.input.reshape(-1, 28, 28, 1)

        I_new_input_preprocess = ImageUtils.picture_preprocess(I_new_input)
        temp_result_v1 = model_v1.predict(I_new_input_preprocess)

        predict_result_v1 = np.argmax(temp_result_v1, axis=1)
        y_prob_vector_max_confidence_m1 = np.max(temp_result_v1, axis=1)
        ground_truth = I_new.label

        model_v2 = self.experiment.modelv2
        temp_result_v2 = model_v2.predict(I_new_input_preprocess)
        predict_result_v2 = np.argmax(temp_result_v2, axis=1)

        y_m2_at_m1_max_pos = []
        for i in range(len(temp_result_v2)):
            y_m2_at_m1_max_pos.append(temp_result_v2[i][predict_result_v1[i]])

        difference = (y_prob_vector_max_confidence_m1 - y_m2_at_m1_max_pos)

        if predict_result_v1 != ground_truth and predict_result_v2 != ground_truth:
            return 1
        elif predict_result_v1 == ground_truth and predict_result_v2 != ground_truth:
            return 2
        elif predict_result_v1 == ground_truth and predict_result_v2 == ground_truth and difference > 0.3:
            return 3
        else:
            return 0

    def isChanged(self, I, I_new):
        return np.any(I.input != I_new.input)

    def Predict(self, B_new):
        B_new_input = testcaselist2nparray(B_new)
        B_new_input_preprocess = ImageUtils.picture_preprocess(B_new_input)
        activation_dicts, self.last_coverage_state, cov = self.experiment.coverage.step(B_new_input_preprocess,
                                                                                        update_state=False)
        return cov, activation_dicts

    def CoverageGain(self, cov):
        return cov > 0

    def BatchPrioritize(self, T, B_id):
        B_c, Bs = T
        B_c[B_id] += 1

    def Mutate(self, I):
        G, P = self.params.G, self.params.P
        I0, I0_new, state = self.info[I.input]
        for i in range(1, self.params.TRY_NUM):
            if state == 0:
                t, p = self.randomPick(G + P)
            else:
                t, p = self.randomPick(P)
            I_new = t(np.copy(I.input), p).reshape(*(self.input_shape[1:]))
            I_new = np.clip(I_new, 0, 255)
            if (t, p) in G:
                state = 1
                I0_new = t(np.copy(I0), p)
                self.info[I_new] = (np.copy(I0), np.copy(I0_new), state)
                return TestCase(I_new, I.label, I.source_id)

            if self.f(I0_new, I_new, I0, state):

                return TestCase(I_new, I.label, I.source_id)
        return None

    def randomPick(self, A):
        c = np.random.randint(0, len(A))
        return A[c]

    def isFailedTestList(self, I_new_list):
        model_v1 = self.experiment.model
        I_new_list_inputs = testcaselist2nparray(I_new_list)
        ground_truth_list = testcaselist2labels(I_new_list)
        I_new_input = I_new_list_inputs.reshape(-1, self.params.input_shape[1], self.params.input_shape[2],
                                                self.params.input_shape[3])
        I_new_input_preprocess = ImageUtils.picture_preprocess(I_new_input)

        temp_result_v1 = model_v1.predict(I_new_input_preprocess)
        predict_result_v1 = np.argmax(temp_result_v1, axis=1)
        y_prob_vector_max_confidence_m1 = np.max(temp_result_v1, axis=1)

        model_v2 = self.experiment.modelv2
        temp_result_v2 = model_v2.predict(I_new_input_preprocess)
        predict_result_v2 = np.argmax(temp_result_v2, axis=1)

        y_m2_at_m1_max_pos = []
        for i in range(len(temp_result_v2)):
            y_m2_at_m1_max_pos.append(temp_result_v2[i][predict_result_v1[i]])

        difference = (y_prob_vector_max_confidence_m1 - y_m2_at_m1_max_pos)

        both_file_list = []
        regression_faults_list = []
        weaken_faults_list = []
        half_weaken_faults_list = []
        rest_case_list = []
        potential_source_id = []
        for i in range(len(I_new_list)):
            if predict_result_v1[i] != ground_truth_list[i] and predict_result_v2[i] != ground_truth_list[i]:
                both_file_list.append(I_new_list[i])
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] != ground_truth_list[i]:
                regression_faults_list.append(I_new_list[i])
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and \
                    difference[i] > 0.3:
                weaken_faults_list.append(I_new_list[i])
                rest_case_list.append(I_new_list[i])
            elif predict_result_v1[i] == ground_truth_list[i] and predict_result_v2[i] == ground_truth_list[i] and \
                    difference[i] > 0.15:
                half_weaken_faults_list.append(I_new_list[i])
                rest_case_list.append(I_new_list[i])
            else:
                rest_case_list.append(I_new_list[i])
        return both_file_list, regression_faults_list, weaken_faults_list, half_weaken_faults_list, rest_case_list, potential_source_id

    def f(self, I, I_new, I_origin, state):
        if state == 0:
            l0_ref = (np.sum((I - I_new) != 0))
            linf_ref = np.max(np.abs(I - I_new))
        else:
            l0_ref = (np.sum((I - I_new) != 0)) + (np.sum((I - I_origin) != 0))
            linf_ref = max(np.max(np.abs(I - I_new)), np.max(np.abs(I - I_origin)))
        if (l0_ref < self.params.alpha * np.size(I)):
            return linf_ref <= 255
        else:
            return linf_ref < self.params.beta * 255
