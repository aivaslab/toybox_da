"""Module containing code to analyze the activations"""
import os
import argparse
import pickle
import collections
import numpy as np
import matplotlib.pyplot as plt

TOYBOX_CLASSES = ["airplane", "ball", "car", "cat", "cup", "duck", "giraffe", "helicopter", "horse", "mug", "spoon",
                  "truck"]

DOMAINS = ["tb", "in12"]


def get_parser():
    """Returns parser for the run"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-path", type=str, required=True, help="Path for the saved model")
    return vars(parser.parse_args())


def analyze_domains(exp_args, layer):
    """Code to analyze neurons for toybox activations"""
    tb_dir_path = exp_args['dir_path'] + "activations/from_hooks/Toybox/"
    in12_dir_path = exp_args['dir_path'] + "activations/from_hooks/IN-12/"
    img_out_dir = exp_args['dir_path'] + "activations/from_hooks/images/"
    os.makedirs(img_out_dir, exist_ok=True)
    assert os.path.isdir(tb_dir_path) and os.path.isdir(in12_dir_path)
    tb_activations = []
    in12_activations = []
    for i, cl in enumerate(TOYBOX_CLASSES):
        tb_act_path = tb_dir_path + cl + ".pkl"
        tb_act_fp = open(tb_act_path, "rb")
        tb_cl_act = pickle.load(tb_act_fp)
        tb_act_fp.close()
        tb_act = tb_cl_act[layer]
        tb_act = np.concatenate(tb_act, axis=0)
        
        tb_activations.append(tb_act)

        in12_act_path = in12_dir_path + cl + ".pkl"
        in12_act_fp = open(in12_act_path, "rb")
        in12_cl_act = pickle.load(in12_act_fp)
        in12_act_fp.close()
        in12_act = in12_cl_act[layer]
        in12_act = np.concatenate(in12_act, axis=0)

        in12_activations.append(in12_act)
        
    tb_activations = np.concatenate(tb_activations, axis=0)
    in12_activations = np.concatenate(in12_activations, axis=0)
    tb_activations_min, tb_activations_max = \
        np.min(tb_activations, axis=0, keepdims=True), np.max(tb_activations, axis=0, keepdims=True)
    in12_activations_min, in12_activations_max = \
        np.min(in12_activations, axis=0, keepdims=True), np.max(in12_activations, axis=0, keepdims=True)
    min_activation, max_activation = \
        np.min(np.concatenate([tb_activations_min, in12_activations_min], axis=0), axis=0, keepdims=True), \
        np.max(np.concatenate([tb_activations_max, in12_activations_max], axis=0), axis=0, keepdims=True),
    
    tb_activations = (tb_activations - min_activation) / (max_activation - min_activation)
    in12_activations = (in12_activations - min_activation) / (max_activation - min_activation)
    tb_mean_activations = np.mean(tb_activations, axis=0, keepdims=True)
    in12_mean_activations = np.mean(in12_activations, axis=0, keepdims=True)
    all_mean_activations = np.concatenate([tb_mean_activations, in12_mean_activations], axis=0)
    preferred_dom = np.argmax(all_mean_activations, axis=0)
    preferred_dom_cntr = collections.Counter(preferred_dom)
    print(preferred_dom_cntr)
    preferred_doms = [preferred_dom_cntr[0], preferred_dom_cntr[1]]
    fig, ax = plt.subplots(figsize=(8, 9), layout='tight')
    ax.bar(x=np.arange(2), height=preferred_doms, tick_label=DOMAINS)
    ax.set_title("Number of preferred neurons for each category")
    plt.savefig(img_out_dir + "pref_neurons_domain.png")
    plt.close()
    
    mean_activations_mins = np.min(all_mean_activations, axis=0, keepdims=True)
    mean_activations_maxs = np.max(all_mean_activations, axis=0, keepdims=True)
    selectivities = np.squeeze((mean_activations_maxs - mean_activations_mins) /
                               (mean_activations_maxs + mean_activations_mins))
    print(selectivities.shape, np.min(selectivities), np.max(selectivities))
    fig, ax = plt.subplots(figsize=(16, 9), layout='tight')
    ax.hist(selectivities, bins=50)
    ax.set_title("Histogram of selectivities in last conv layer for all neurons")
    plt.savefig(img_out_dir + "domain_selectivities_hist.png")
    plt.close()
    print(preferred_dom.shape, selectivities.shape)
    tb_selectivities = selectivities[preferred_dom == 0]
    in12_selectivities = selectivities[preferred_dom == 1]
    # in12_selectivities = np.where(preferred_dom == 1, selectivities)
    print(tb_selectivities.shape, in12_selectivities.shape)

    fig, ax = plt.subplots(figsize=(16, 9), layout='tight')
    ax.hist(tb_selectivities, bins=50)
    ax.set_title("Histogram of selectivities in last conv layer for toybox")
    plt.savefig(img_out_dir + "toybox_selectivities_hist.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 9), layout='tight')
    ax.hist(in12_selectivities, bins=50)
    ax.set_title("Histogram of selectivities in last conv layer for in12")
    plt.savefig(img_out_dir + "in12_selectivities_hist.png")
    plt.close()
    

def analyze_tb(exp_args, layer, cl):
    """Code to analyze neurons for toybox activations"""
    assert cl in ['tb', 'in12']
    if cl == 'tb':
        dir_path = exp_args['dir_path'] + "activations/from_hooks/Toybox/"
        cl_name = "toybox"
    else:
        dir_path = exp_args['dir_path'] + "activations/from_hooks/IN-12/"
        cl_name = "in-12"
    img_out_dir = exp_args['dir_path'] + "activations/from_hooks/images/"
    os.makedirs(img_out_dir, exist_ok=True)
    assert os.path.isdir(dir_path)
    activations = {}
    mean_activations = {}
    all_mean_activations = None
    for i, cl in enumerate(TOYBOX_CLASSES):
        act_path = dir_path + cl + ".pkl"
        act_fp = open(act_path, "rb")
        cl_act = pickle.load(act_fp)
        act_fp.close()
        ll_act = cl_act[layer]
        ll_act = np.concatenate(ll_act, axis=0)
        activations[cl] = ll_act
        mean_act = np.mean(ll_act, axis=0, keepdims=True)
        mean_activations[cl] = mean_act
        # print(mean_act.shape)
        if all_mean_activations is None:
            all_mean_activations = mean_act
        else:
            all_mean_activations = np.concatenate([all_mean_activations, mean_act])
        # print(all_mean_activations.shape)
        
    min_act, max_act = np.min(all_mean_activations), np.max(all_mean_activations)
    
    all_mean_activations = (all_mean_activations - min_act) / (max_act - min_act)
    preferred_cl = np.argmax(all_mean_activations, axis=0)
    preferred_cl_cntr = collections.Counter(preferred_cl)
    print(preferred_cl_cntr)
    preferred_cl_ll = []
    for i in range(len(TOYBOX_CLASSES)):
        preferred_cl_ll.append(preferred_cl_cntr[i])
    fig, ax = plt.subplots(figsize=(16, 9), layout='tight')
    ax.bar(x=np.arange(12), height=preferred_cl_ll, tick_label=TOYBOX_CLASSES)
    plt.savefig(img_out_dir+cl_name+"_pref_neurons.png")
    plt.close()
    
    pref_act = []
    selectivities = []
    avg_nonpref_act = []
    
    for i in range(preferred_cl.shape[0]):
        pref_cl = preferred_cl[i]
        pref_cl_act = all_mean_activations[pref_cl][i]
        all_act = all_mean_activations[:, i]
        sum_act = np.sum(all_act)
        nonpref_act_sum = sum_act - pref_cl_act
        nonpref_act_avg = nonpref_act_sum / (len(TOYBOX_CLASSES) - 1)
        pref_act.append(sum_act)
        avg_nonpref_act.append(nonpref_act_avg)
        selectivity = (pref_cl_act - nonpref_act_avg) / (pref_cl_act + nonpref_act_avg)
        selectivities.append(selectivity)
        
        # print(pref_cl_act, nonpref_act_avg, selectivity)
    
    fig, ax = plt.subplots(figsize=(16, 9), layout='tight')
    ax.hist(selectivities, bins=50)
    ax.set_title("Histogram of selectivities in last conv layer for " + cl_name)
    plt.savefig(img_out_dir+cl_name+"_selectivities_hist.png")
    plt.close()
    

def main():
    """Main method"""
    args = get_parser()
    # analyze_tb(exp_args = args, layer="layer4.1.conv2", cl='tb')
    # analyze_tb(exp_args = args, layer="layer4.1.conv2", cl='in12')
    analyze_domains(exp_args=args, layer='layer4.1.conv2')
    

if __name__ == "__main__":
    main()
