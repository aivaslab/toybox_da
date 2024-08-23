import argparse
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

import torch

UMAP_FILENAMES = ["tb_train.csv", "tb_test.csv", "in12_train.csv", "in12_test.csv"]
NORM_UMAP_FILENAMES = ["tb_train_norm.csv", "tb_test_norm.csv", "in12_train_norm.csv", "in12_test_norm.csv"]
TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck',
              'giraffe', 'horse', 'helicopter', 'mug', 'spoon', 'truck']

SUPER_CATEGORIES = {
    'airplane': 'vehicle',
    'ball': 'household',
    'car': 'vehicle',
    'cat': 'animal',
    'cup': 'household',
    'duck': 'animal',
    'giraffe': 'animal',
    'horse': 'animal',
    'helicopter': 'vehicle',
    'mug': 'household',
    'spoon': 'household',
    'truck': 'vehicle'
}

DATASETS = ["tb_train", "tb_test", "in12_train", "in12_test"]
DSET_CUTOFFS = {
    'tb_train': 0,
    'tb_test': 12,
    'in12_train': 24,
    'in12_test': 36
}

SRC_FILES = {
    'tb_train': "../data/data_12/Toybox/toybox_data_interpolated_cropped_dev.csv",
    'tb_test': "../data/data_12/Toybox/toybox_data_interpolated_cropped_test.csv",
    'toybox_train': "../data/data_12/Toybox/toybox_data_interpolated_cropped_dev.csv",
    'toybox_test': "../data/data_12/Toybox/toybox_data_interpolated_cropped_test.csv",
    'in12_train': "../data/data_12/IN-12/dev.csv",
    'in12_test': "../data/data_12/IN-12/test.csv"
}


def get_dist_matrix(feats):
    n, dim = feats.size(0), feats.size(1)
    all_dists = torch.zeros((n, n), dtype=torch.float32).cuda()

    chunk_size = n // 12

    chunks = torch.chunk(feats, 12, dim=0)
    for chunk in chunks:
        assert chunk.shape == (chunk_size, 512)

    for i, chunk_1 in enumerate(chunks):
        for j, chunk_2 in enumerate(chunks):
            dist = ((chunk_1.unsqueeze(1) - chunk_2.unsqueeze(0)) ** 2).sum(dim=-1) ** 0.5
            assert dist.shape == (chunk_size, chunk_size)
            all_dists[i*chunk_size: i*chunk_size + chunk_size, j*chunk_size: j*chunk_size + chunk_size] = dist

    all_dists = all_dists.cpu().numpy()
    assert np.all(all_dists.transpose(0, 1) == all_dists)
    return all_dists


def get_activation_points(path, dataset):
    """Get a 2d matrix of datapoints for the given path, dataset and cl"""
    act_fname = dataset + "_activations.npy"
    act_fpath = path + act_fname

    idx_fname = dataset + "_indices.npy"
    idx_fpath = path + idx_fname

    assert os.path.isfile(act_fpath) and os.path.isfile(idx_fpath), f"Could not find {act_fpath} and {idx_fpath}"
    act_data = np.load(act_fpath)
    idx_data = np.load(idx_fpath)
    idx_key = 'ID' if "toybox" in dataset else "Index"

    src_data_fpath = SRC_FILES[dataset]
    src_data_fp = open(src_data_fpath, "r")
    src_data = list(csv.DictReader(src_data_fp))

    cl_data = []

    for i, idx in enumerate(idx_data):
        act_idx, src_cl = int(src_data[idx][idx_key]), src_data[idx]['Class']
        assert act_idx == idx
        cl_data.append(src_cl)

    cl_data_np = np.array(cl_data)
    cl_sim_mat = cl_data_np[np.newaxis, :] == cl_data_np[:, np.newaxis]

    cl_match_mat = np.eye(len(TB_CLASSES), dtype=bool)
    cl_match_mat = np.repeat(cl_match_mat, 1500, axis=1)
    cl_match_mat = np.repeat(cl_match_mat, 1500, axis=0)

    assert np.all(cl_sim_mat == cl_match_mat)

    # print(dataset, target_cl, ret_data_np.shape)
    src_data_fp.close()
    return act_data


def gen_intra_domain_dist_histograms(path, title):
    act_path = path + "analysis/final_model/backbone/activations/"
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex=True, sharey=True)
    max_val = -np.inf
    for idx, dset in enumerate(["toybox_train", "in12_train"]):
        activations = get_activation_points(path=act_path, dataset=dset)
        dist_matrix = get_dist_matrix(feats=torch.from_numpy(activations).cuda())
        q99 = np.quantile(dist_matrix, q=0.99)
        max_val = max(max_val, q99)

    cl_match_mat = np.eye(len(TB_CLASSES), dtype=bool)
    super_cl_match_mat = np.zeros((12, 12), dtype=bool)
    for i, cl_1 in enumerate(TB_CLASSES):
        for j, cl_2 in enumerate(TB_CLASSES):
            if cl_1 != cl_2 and SUPER_CATEGORIES[cl_1] == SUPER_CATEGORIES[cl_2]:
                super_cl_match_mat[i][j] = True

    cl_match_mat = np.repeat(cl_match_mat, 1500, axis=1)
    cl_match_mat = np.repeat(cl_match_mat, 1500, axis=0)
    cl_mismatch_mat = ~cl_match_mat

    super_cl_match_mat = np.repeat(super_cl_match_mat, 1500, axis=1)
    super_cl_match_mat = np.repeat(super_cl_match_mat, 1500, axis=0)
    super_cl_mismatch_mat = ~(cl_match_mat | super_cl_match_mat)
    # print(max_val)

    def plot_histogram(axis, data, n_bins, x_range, labels, ax_title):
        colors = []
        cmap_name = 'tab10'
        for data_idx in range(len(data)):
            colors.append(mpl.colormaps[cmap_name].colors[data_idx])

        axis.hist(data, stacked=False, bins=n_bins, range=x_range, label=labels, color=colors)
        for data_idx in range(len(data)):
            partial_data = data[data_idx]
            axis.axvline(np.median(partial_data), color=colors[data_idx], linestyle='dashed', linewidth=2)
        axis.legend(loc="upper right", fontsize="large")
        axis.set_title(ax_title)

    for idx, dset in enumerate(["toybox_train", "in12_train"]):
        activations = get_activation_points(path=act_path, dataset=dset)
        dist_matrix = get_dist_matrix(feats=torch.from_numpy(activations).cuda())

        cl_match_dists = dist_matrix[cl_match_mat]
        cl_mismatch_dists = dist_matrix[cl_mismatch_mat]
        assert len(cl_match_dists) == 1500 * 1500 * 12
        assert len(cl_mismatch_dists) == 1500 * 1500 * 11 * 12
        # print(cl_match_dists.shape, cl_mismatch_dists.shape)

        plot_histogram(axis=axes[idx][0], data=[cl_match_dists, cl_mismatch_dists], n_bins=100,
                       x_range=(0.0, max_val), labels=["class_match", "class_mismatch"], ax_title=dset)

        super_cl_match_dists = dist_matrix[super_cl_match_mat]
        super_cl_mismatch_dists = dist_matrix[super_cl_mismatch_mat]
        assert len(super_cl_match_dists) == 1500 * 1500 * 3 * 12
        assert len(super_cl_mismatch_dists) == 1500 * 1500 * 8 * 12
        # print(cl_match_dists.shape, super_cl_match_dists.shape, super_cl_mismatch_dists.shape)

        plot_histogram(axis=axes[idx][1], data=[cl_match_dists, super_cl_match_dists, super_cl_mismatch_dists],
                       n_bins=100, x_range=(0.0, max_val),
                       labels=["class_match", "superclass_match", "superclass_mismatch"], ax_title=dset)

    histogram_fname = f"{act_path}/intra_domain_dist_histogram.jpeg"
    fig.tight_layout(pad=2.0, h_pad=1.5)
    fig.suptitle(title, fontsize='x-large')
    plt.savefig(histogram_fname)
    plt.close()
    return histogram_fname


def plot_and_show_histogram(path, title):
    hist_fname = gen_intra_domain_dist_histograms(path=path, title=title)
    img = Image.open(hist_fname)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True,
                        help='Model path for which histogram has to be computed')
    args = vars(parser.parse_args())
    # "../out/DUAL_SSL_DOM_MMD_V1/dual_ssl_no_dom_mmd_trial_500/"
    model_path = args['model_path']
    hist_img = plot_and_show_histogram(model_path, title="dual_ssl_no_dom_mmd")
    hist_img.show()
