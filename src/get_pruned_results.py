"""Module with implementations of different pruning methods"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import models_pruned


def numerical_order():
    """Return numerical order of neurons"""
    return np.arange(512)


def plot_pruned_results(results_dict, save_path=None):
    """Plot the results of the pruning"""
    
    tb_tr_accs, tb_te_accs, in12_tr_accs, in12_te_accs = [], [], [], []
    keys = sorted(list(results_dict.keys()))
    for key in keys:
        tb_tr_accs.append(results_dict[key][0])
        tb_te_accs.append(results_dict[key][1])
        in12_tr_accs.append(results_dict[key][2])
        in12_te_accs.append(results_dict[key][3])

    fig, ax = plt.subplots(figsize=(16, 9), layout='tight')
    ax.plot(results_dict.keys(), tb_tr_accs, "bo--", label="Toybox Train")
    ax.plot(results_dict.keys(), tb_te_accs, "bo-", label="Toybox Test")
    ax.plot(results_dict.keys(), in12_tr_accs, "r+--", label="IN-12 Train")
    ax.plot(results_dict.keys(), in12_te_accs, "r+-", label="IN-12 Test")
    ax.legend(loc="upper right")
    ax.set_xlabel("Number of neurons pruned")
    ax.set_ylabel("Absolute Accuracy")
    ax.set_ylim(0, 100)
    if save_path is not None:
        plt.savefig(save_path+"_absolute.png")
    
    plt.close()

    tb_tr_accs = list(map(lambda x: x * 100. / tb_tr_accs[0], tb_tr_accs))
    tb_te_accs = list(map(lambda x: x * 100. / tb_te_accs[0], tb_te_accs))
    in12_tr_accs = list(map(lambda x: x * 100. / in12_tr_accs[0], in12_tr_accs))
    in12_te_accs = list(map(lambda x: x * 100. / in12_te_accs[0], in12_te_accs))
    fig, ax = plt.subplots(figsize=(16, 9), layout='tight')
    ax.plot(results_dict.keys(), tb_tr_accs, "bo--", label="Toybox Train")
    ax.plot(results_dict.keys(), tb_te_accs, "bo-", label="Toybox Test")
    ax.plot(results_dict.keys(), in12_tr_accs, "r+--", label="IN-12 Train")
    ax.plot(results_dict.keys(), in12_te_accs, "r+-", label="IN-12 Test")
    ax.legend(loc="upper right")
    ax.set_xlabel("Number of neurons pruned")
    ax.set_ylabel("Accuracy relative to unpruned model")
    ax.set_ylim(0, 100)
    if save_path is not None:
        plt.savefig(save_path + "_relative.png")

    plt.close()
    
    
def prune_model_and_plot(dir_path, order, step_size, pref):
    """Prune the model neurons in provided order using specified step size"""
    
    step_sizes = list(range(0, len(order) + 1, step_size))
    if step_sizes[-1] != len(order):
        step_sizes.append(len(order))
    # results_dict = models_pruned.test_prune_eval(model_path=dir_path,
    #                                              ordered_neurons=order,
    #                                              step_sizes=step_sizes)
    #
    out_path = dir_path + "analysis/pruning/"
    os.makedirs(out_path, exist_ok=True)
    out_pref = out_path + pref
    #
    res_out_path = out_pref + ".pkl"
    # res_out_fp = open(res_out_path, "wb")
    # pickle.dump(results_dict, res_out_fp)
    # res_out_fp.close()
    
    res_in_fp = open(res_out_path, "rb")
    res_dict = pickle.load(res_in_fp)
    
    plot_pruned_results(results_dict=res_dict, save_path=out_pref)
    
    
def test_2():
    """Test method"""
    dir_path = "../ICLR_OUT/DUAL_SUP/exp_Aug_10_2023_21_29/"
    preferred_neurons = np.load(dir_path+"activations/from_hooks/data/preferred_domain.npy")
    tb_preferred_neurons = [i for i in range(512) if preferred_neurons[i] == 0]
    in12_preferred_neurons = [i for i in range(512) if preferred_neurons[i] == 1]
    print(len(tb_preferred_neurons), len(in12_preferred_neurons))
    
    prune_model_and_plot(dir_path=dir_path, order=tb_preferred_neurons, step_size=128, pref="tb_preferred")
    prune_model_and_plot(dir_path=dir_path, order=in12_preferred_neurons, step_size=128, pref="in12_preferred")
    
    
def test():
    """Test method"""
    dir_path = "../ICLR_OUT/DUAL_SUP/exp_Aug_10_2023_21_29/"
    ordered_neurons = numerical_order()
    prune_model_and_plot(dir_path=dir_path, order=ordered_neurons, step_size=64, pref="numerical")
    

if __name__ == "__main__":
    test_2()
    