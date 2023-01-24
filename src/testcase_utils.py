import numpy as np


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


def testcaselist2labels(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].label))
    new_input = np.asarray(new_input)
    return new_input


def testcaselist2generation(test_case_list):
    new_input = []
    for i in range(len(test_case_list)):
        new_input.append(np.asarray(test_case_list[i].generation))
    new_input = np.asarray(new_input)
    return new_input


def testcaselist2basicfeatures(test_case_list):
    new_input = []
    input_ids = []
    input_ground_truth = []
    m1_predict_results = []
    m2_predict_results = []

    for i in range(len(test_case_list)):
        new_input.append(test_case_list[i].input)
        input_ids.append(test_case_list[i].source_id)
        input_ground_truth.append(test_case_list[i].label)
        m1_predict_results.append(test_case_list[i].m1_predict_label)
        m2_predict_results.append(test_case_list[i].m2_predict_label)
    new_input = np.asarray(new_input)

    input_ids = np.asarray(input_ids)
    input_ground_truth = np.asarray(input_ground_truth)
    m1_predict_results = np.asarray(m1_predict_results)
    m2_predict_results = np.asarray(m2_predict_results)
    return new_input, input_ids, input_ground_truth, m1_predict_results, m2_predict_results


def testcaselist2all(test_case_list):
    new_input = []
    input_ids = []
    input_ground_truth = []
    m1_predict_results = []
    m2_predict_results = []
    new_m1_traces = []
    new_m2_traces = []

    for i in range(len(test_case_list)):
        new_input.append(test_case_list[i].input)
        input_ids.append(test_case_list[i].source_id)
        input_ground_truth.append(test_case_list[i].label)
        m1_predict_results.append(test_case_list[i].m1_predict_label)
        m2_predict_results.append(test_case_list[i].m2_predict_label)
        new_m1_traces.append(test_case_list[i].m1_trace)
        new_m2_traces.append(test_case_list[i].m2_trace)
    new_input = np.asarray(new_input)

    input_ids = np.asarray(input_ids)
    input_ground_truth = np.asarray(input_ground_truth)
    m1_predict_results = np.asarray(m1_predict_results)
    m2_predict_results = np.asarray(m2_predict_results)
    new_m1_traces = np.asarray(new_m1_traces)
    new_m2_traces = np.asarray(new_m2_traces)
    return new_input, input_ids, input_ground_truth, m1_predict_results, m2_predict_results, new_m1_traces, new_m2_traces


def testcaselist2failureid(test_case_list):
    failure_ids = []
    for i in range(len(test_case_list)):
        failure_ids.append(
            (test_case_list[i].source_id, test_case_list[i].m1_predict_label, test_case_list[i].m2_predict_label))

    return failure_ids


def testcaselist2pastandfailureid(test_case_list):
    failure_ids = []
    past_mutop = []
    history_ids_in_tree = []
    for i in range(len(test_case_list)):
        failure_tuple = (
            test_case_list[i].source_id, test_case_list[i].m1_predict_label, test_case_list[i].m2_predict_label)
        failure_ids.append(failure_tuple)
        past_trace = test_case_list[i].mutation_trace
        history_ids_in_tree.extend(test_case_list[i].last_corpus_trace)
        if len(past_trace) > 0:
            past_mutop.append((past_trace[-1], failure_tuple))
    return failure_ids, past_mutop, history_ids_in_tree
