"""calculate ccmmd loss between activations"""
import argparse
import os
import numpy as np
import torch

import ccmmd

FC_SIZES = {
        'l0': 4096,
        'l1': 4096,
        'l2': 4608,
        'l3': 4096,
        'l4': 4608,
        'avg': 512
    }


def get_ccmmd_from_activations_layer(act_path, layer: str):
    """Calculate the ccmmd loss layer by layer from activations"""
    ccmmd_loss = ccmmd.ClassConditionalMMDLoss(
        kernels=([ccmmd.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
    ).cuda()
    
    tb_labels = np.load(act_path+"tb_tr_labels.npy")
    in12_labels = np.load(act_path+"in12_tr_labels.npy")
    
    tb_feats_path = act_path + "tb_tr_" + layer + "_f.npy"
    in12_feats_path = act_path + "in12_tr_" + layer + "_f.npy"
    tb_feats_l = np.load(tb_feats_path)
    in12_feats_l = np.load(in12_feats_path)
    
    print(layer, len(tb_labels), len(in12_labels))
    total_loss = 0.
    for cl in range(12):
        tb_count = 0
        in12_count = 0
        for i in range(len(tb_labels)):
            if tb_labels[i] == cl:
                tb_count += 1
        for j in range(len(in12_labels)):
            if in12_labels[j] == cl:
                in12_count += 1
        tb_count_curr = 0
        in12_count_curr = 0
        tb_feats = torch.zeros(size=(tb_count, FC_SIZES[layer]))
        in12_feats = torch.zeros(size=(in12_count, FC_SIZES[layer]))
        for i in range(len(tb_labels)):
            if tb_labels[i] == cl:
                tb_feats[tb_count_curr] = torch.from_numpy(tb_feats_l[i])
                tb_count_curr += 1
        for i in range(len(in12_labels)):
            if in12_labels[i] == cl:
                in12_feats[in12_count_curr] = torch.from_numpy(in12_feats_l[i])
                in12_count_curr += 1
        # print(cl, tb_count, in12_count, tb_count_curr, in12_count_curr)
        assert tb_count == 1500
        assert in12_count == 1500
        assert tb_count_curr == 1500
        assert in12_count_curr == 1500
        labels = torch.tensor([cl] * tb_count)
        tb_feats, in12_feats, labels = tb_feats.cuda(), in12_feats.cuda(), labels.cuda()
        
        with torch.no_grad():
            loss = ccmmd_loss(z_s=tb_feats, z_t=in12_feats, l_s=labels, l_t=labels)
            total_loss += loss.item()
        # print(loss.item())
    print(layer, ":", total_loss)
    return total_loss


def get_ccmmd_from_activations(act_path):
    """Calculate ccmmd loss for all layers"""
    assert os.path.isdir(act_path)
    ccmmds = {}
    for layer in ['l0', 'l1', 'l2', 'l3', 'l4', 'avg']:
        ccmmd_loss = get_ccmmd_from_activations_layer(act_path=act_path, layer=layer)
        ccmmds[layer] = ccmmd_loss
    print(ccmmds)
    import pickle
    pkl_fname = act_path + "ccmmds.pkl"
    pkl_fp = open(pkl_fname, "wb")
    pickle.dump(obj=ccmmds, file=pkl_fp)
    pkl_fp.close()
    

def get_args():
    """Parser with arguments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dir-path", "-d", type=str, required=True, help="Path where model is stored.")
    return vars(parser.parse_args())


if __name__ == "__main__":
    # test_network()
    # get_activations_sup(model_path="", out_path="")
    args = get_args()
    dir_name = args['dir_path']
    assert os.path.isdir(dir_name)
    get_ccmmd_from_activations(act_path=dir_name+"/all_activations/")
    
    # {'l0': 3.391178995370865, 'l1': 2.619435392320156, 'l2': 2.4206863418221474, 'l3': 3.78637433052063, 'l4': 4.222728535532951, 'avg': 7.880559921264648}
    