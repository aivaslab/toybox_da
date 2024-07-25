import os
import time

import numpy as np
import csv
import umap_metrics_utils as um
from sklearn.mixture import GaussianMixture
import pickle
import multiprocessing as mp

UMAP_FILENAMES = ["tb_train.csv", "tb_test.csv", "in12_train.csv", "in12_test.csv"]
NORM_UMAP_FILENAMES = ["tb_train_norm.csv", "tb_test_norm.csv", "in12_train_norm.csv", "in12_test_norm.csv"]
TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck',
              'giraffe', 'horse', 'helicopter', 'mug', 'spoon', 'truck']
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


def get_umap_points(path, dataset, target_cl, norm=True):
    """Get a 2d matrix of datapoints for the given path, dataset and cl"""
    if norm:
        fname = dataset + "_norm.csv"
        assert fname in NORM_UMAP_FILENAMES
    else:
        fname = dataset + ".csv"
        assert fname in UMAP_FILENAMES
    fpath = path + fname

    assert os.path.isfile(fpath)
    assert target_cl in TB_CLASSES
    fp = open(fpath, "r")
    data = list(csv.DictReader(fp))
    idx_key = 'ID' if "tb" in dataset else "Index"

    src_data_fpath = SRC_FILES[dataset]
    src_data_fp = open(src_data_fpath, "r")
    src_data = list(csv.DictReader(src_data_fp))

    ret_data = []

    for dp in data:
        idx, x, y = int(dp['idx']), float(dp['x']), float(dp['y'])
        i, src_cl = int(src_data[idx][idx_key]), src_data[idx]['Class']
        assert i == idx
        if src_cl == target_cl:
            ret_data.append([x, y])
    ret_data_np = np.array(ret_data)
    # print(dataset, target_cl, ret_data_np.shape)
    fp.close()
    src_data_fp.close()
    return ret_data_np


def get_activation_points(path, dataset, target_cl):
    """Get a 2d matrix of datapoints for the given path, dataset and cl"""
    act_fname = dataset + "_activations.npy"
    act_fpath = path + act_fname

    idx_fname = dataset + "_indices.npy"
    idx_fpath = path + idx_fname

    assert os.path.isfile(act_fpath) and os.path.isfile(idx_fpath)
    assert target_cl in TB_CLASSES
    act_data = np.load(act_fpath)
    idx_data = np.load(idx_fpath)
    idx_key = 'ID' if "toybox" in dataset else "Index"

    src_data_fpath = SRC_FILES[dataset]
    src_data_fp = open(src_data_fpath, "r")
    src_data = list(csv.DictReader(src_data_fp))

    ret_data = []

    for i, idx in enumerate(idx_data):
        act_idx, src_cl = int(src_data[idx][idx_key]), src_data[idx]['Class']
        assert act_idx == idx
        if src_cl == target_cl:
            ret_data.append(act_data[i])
    ret_data_np = np.array(ret_data)
    # print(dataset, target_cl, ret_data_np.shape)
    src_data_fp.close()
    return ret_data_np


def extract_feats_from_umap(act_path, umap_path, norm):
    assert os.path.isdir(act_path)
    assert os.path.isdir(umap_path)

    umap_points_dict = {}
    act_points_dict = {}
    cc_idxs_dict = {}
    acc_points_dict = {}
    acc_means_dict = {}
    acc_covs_dict = {}
    acc_weights_dict = {}

    for dset in DATASETS:
        for cl in TB_CLASSES:
            act_dset = dset if 'tb' not in dset else dset.replace('tb', 'toybox')
            umap_points = get_umap_points(path=umap_path, dataset=dset, norm=norm, target_cl=cl)
            act_points = get_activation_points(path=act_path, dataset=act_dset, target_cl=cl)
            assert len(umap_points) == len(act_points)
            umap_points_dict[(dset, cl)] = umap_points
            act_points_dict[(dset, cl)] = act_points

            largest_cc_idxs, all_cc_idxs, comp_means, comp_covs, all_cc_idxs_split = \
                um.build_and_compute_mst(path=umap_path, dataset=dset, target_cl=cl, points=umap_points,
                                         cc_rem_threshold=5, plot=False)

            assert len(comp_means) == len(comp_covs) and len(comp_means) == len(all_cc_idxs_split)
            assert len(all_cc_idxs) == sum([len(cc_idxs) for cc_idxs in all_cc_idxs_split])

            all_cc_points = []
            all_cc_means = []
            all_cc_covs = []
            all_cc_weights = []
            for cc_idxs in all_cc_idxs_split:
                cc_points = np.array([act_points[i] for i in cc_idxs])
                cc_mean = np.mean(cc_points, axis=0)
                cc_cov = np.cov(cc_points, rowvar=False)
                all_cc_points.append(cc_points)
                all_cc_means.append(cc_mean)
                all_cc_covs.append(cc_cov)
                all_cc_weights.append(len(cc_points) / len(all_cc_idxs))

            all_points = np.concatenate(all_cc_points)
            all_cc_means = np.array(all_cc_means)
            all_cc_covs = np.array(all_cc_covs)
            all_cc_weights = np.array(all_cc_weights)
            assert len(all_points) == len(all_cc_idxs)

            cc_idxs_dict[(dset, cl)] = all_cc_idxs_split
            acc_points_dict[(dset, cl)] = all_points
            acc_means_dict[(dset, cl)] = all_cc_means
            acc_covs_dict[(dset, cl)] = all_cc_covs
            acc_weights_dict[(dset, cl)] = all_cc_weights
    um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="cc_idxs", norm=False, data_dict=cc_idxs_dict)
    um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="act_dict", norm=False, data_dict=acc_points_dict)
    um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="act_means", norm=False, data_dict=acc_means_dict)
    um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="act_covs", norm=False, data_dict=acc_covs_dict)
    um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="act_weights", norm=False, data_dict=acc_weights_dict)


def load_pkl(fpath, norm):
    fname = fpath + "_norm.pkl" if norm else fpath + ".pkl"
    fp = open(fname, "rb")
    data_dict = pickle.load(fp)
    fp.close()
    return data_dict


def check_matrix_full(mat):
    is_full = True
    for prec in mat:
        is_full = is_full and np.allclose(prec, prec.T) and np.all(np.linalg.eigvalsh(prec)) > 0.0
    return is_full


def get_feat_gmm_likelihoods(act_path, umap_path, norm, use_umap=True):
    assert os.path.isdir(act_path)
    assert os.path.isdir(umap_path)

    acc_points_dict = load_pkl(umap_path+"/ll/acc/feats/act_dict", norm=norm)
    acc_means_dict = load_pkl(umap_path+"/ll/acc/feats/act_means", norm=norm)
    acc_covs_dict = load_pkl(umap_path+"/ll/acc/feats/act_covs", norm=norm)
    acc_weights_dict = load_pkl(umap_path+"/ll/acc/feats/act_weights", norm=norm)
    gmm_means_dict, gmm_covs_dict, gmm_weights_dict, ll_dict = {}, {}, {}, {}
    for dset in DATASETS:
        for cl in TB_CLASSES:
            key = (dset, cl)
            points, means, covs, weights = \
                acc_points_dict[key], acc_means_dict[key], acc_covs_dict[key], acc_weights_dict[key]
            perturb = 1e-3 * np.eye(covs.shape[1])
            perturb = np.expand_dims(perturb, axis=0)
            covs += perturb
            precision_matrix = np.linalg.inv(covs)
            # print(dset, cl, len(acc_means_dict[key]), check_matrix_full(covs), check_matrix_full(precision_matrix),
            #       np.linalg.matrix_rank(covs), np.linalg.cond(covs))

            for i in range(precision_matrix.shape[0]):
                prec = precision_matrix[i]
                is_symmetric = np.allclose(prec, prec.T)
                eig_vals = np.linalg.eigvalsh(prec)
                assert is_symmetric and np.all(eig_vals) > 0.0

            for i in range(precision_matrix.shape[0]):
                prec = precision_matrix[i]
                is_symmetric = np.allclose(prec, prec.T)
                assert is_symmetric
            if use_umap:
                model = GaussianMixture(n_components=len(means), covariance_type='full', means_init=means,
                                        weights_init=weights, precisions_init=precision_matrix, reg_covar=1e-5)
            else:
                model = GaussianMixture(n_components=len(means), covariance_type='full', reg_covar=1e-5)
            model.fit(points)
            gmm_means_dict[(dset, cl)] = model.means_
            gmm_covs_dict[(dset, cl)] = model.covariances_
            gmm_weights_dict[(dset, cl)] = model.weights_

            for trgt_dset in DATASETS:
                for trgt_cl in TB_CLASSES:
                    trgt_points = acc_points_dict[(trgt_dset, trgt_cl)]
                    ll = model.score_samples(trgt_points)
                    ll_dict[(dset, cl, trgt_dset, trgt_cl)] = ll
    if use_umap:
        um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="gmm_ll", norm=False, data_dict=ll_dict)
        um.save_pkl(path=umap_path + "/ll/acc/feats/", fname="gmm_means", norm=False, data_dict=gmm_means_dict)
        um.save_pkl(path=umap_path + "/ll/acc/feats/", fname="gmm_covs", norm=False, data_dict=gmm_covs_dict)
        um.save_pkl(path=umap_path + "/ll/acc/feats/", fname="gmm_weights", norm=False, data_dict=gmm_weights_dict)
    else:
        um.save_pkl(path=umap_path + "/ll/acc/feats/", fname="gmm_ll_raw", norm=False, data_dict=ll_dict)
        um.save_pkl(path=umap_path + "/ll/acc/feats/", fname="gmm_means_raw", norm=False, data_dict=gmm_means_dict)
        um.save_pkl(path=umap_path + "/ll/acc/feats/", fname="gmm_covs_raw", norm=False, data_dict=gmm_covs_dict)
        um.save_pkl(path=umap_path + "/ll/acc/feats/", fname="gmm_weights_raw", norm=False, data_dict=gmm_weights_dict)

    # um.make_ll_tbl(save_path="../temp/", dic=ll_dict)


def gen_feat_gmm_emd(umap_path, norm, use_umap=True):
    gmm_means_path = umap_path+"/ll/acc/feats/gmm_means" if use_umap else umap_path+"/ll/acc/feats/gmm_means_raw"
    gmm_covs_path = umap_path+"/ll/acc/feats/gmm_covs" if use_umap else umap_path+"/ll/acc/feats/gmm_covs_raw"
    gmm_weights_path = umap_path+"/ll/acc/feats/gmm_weights" if use_umap else umap_path+"/ll/acc/feats/gmm_weights_raw"
    gmm_means_dict = load_pkl(gmm_means_path, norm=norm)
    gmm_covs_dict = load_pkl(gmm_covs_path, norm=norm)
    gmm_weights_dict = load_pkl(gmm_weights_path, norm=norm)
    ot_dict = {}
    count = 0
    tot_time = 0.0
    for dset in DATASETS:
        for cl in TB_CLASSES:
            key = (dset, cl)
            means, covs, weights = gmm_means_dict[key], gmm_covs_dict[key], gmm_weights_dict[key]

            st_time = time.time()
            # for trgt_dset in DATASETS:
            for trgt_cl in TB_CLASSES:
                if cl == trgt_cl:
                    ot_dict[(dset, cl, dset, cl)] = [0.0]
                else:
                    dict_key = (dset, cl, dset, trgt_cl)
                    if dict_key not in ot_dict:
                        trgt_key = (dset, trgt_cl)
                        trgt_means, trgt_covs, trgt_weights = \
                            gmm_means_dict[trgt_key], gmm_covs_dict[trgt_key], gmm_weights_dict[trgt_key]
                        costs, ot_map, dist = um.GW2(weights, means, covs, trgt_weights, trgt_means, trgt_covs)
                        ot_dict[(dset, cl, dset, trgt_cl)] = [dist]
                        ot_dict[(dset, trgt_cl, dset, cl)] = [dist]

                count += 1
            tot_time += time.time() - st_time
            print(dset, cl, round(time.time() - st_time, 2), round(tot_time / count, 2))

    if use_umap:
        um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="gmm_emd", norm=False, data_dict=ot_dict)
    else:
        um.save_pkl(path=umap_path+"/ll/acc/feats/", fname="gmm_emd_raw", norm=False, data_dict=ot_dict)


def run_all_feature_metrics_on_model(model_path):
    umap_dir_path = model_path + "/analysis/final_model/backbone/umap/all_data/umap_200_0.1_euclidean/"
    activation_path = model_path + "/analysis/final_model/backbone/activations/"
    st_time = time.time()

    extract_feats_from_umap(umap_path=umap_dir_path, act_path=activation_path, norm=False)
    print(time.time() - st_time, "finished extracting features")
    st_time = time.time()

    get_feat_gmm_likelihoods(umap_path=umap_dir_path, act_path=activation_path, norm=False)
    print(time.time() - st_time, "finished gmm umap")
    st_time = time.time()

    get_feat_gmm_likelihoods(umap_path=umap_dir_path, act_path=activation_path, norm=False, use_umap=False)
    print(time.time() - st_time, "finished gmm raw")
    st_time = time.time()

    gen_feat_gmm_emd(umap_path=umap_dir_path, norm=False)
    print(time.time() - st_time, "finished gmm emd")


def run_all_feature_metrics(model_paths):
    mp.set_start_method('forkserver')
    num_procs = 4
    idx = 0
    while idx < len(model_paths):
        curr_procs = []
        while idx < len(model_paths) and len(curr_procs) < num_procs:
            model_path = model_paths[idx]
            new_proc = mp.Process(target=run_all_feature_metrics_on_model, args=(model_path, ))
            curr_procs.append(new_proc)
            idx += 1

        for proc in curr_procs:
            proc.start()

        for proc in curr_procs:
            proc.join()


if __name__ == '__main__':
    DUAL_SSL_CLASS_100_100_PATH_PRE = "../out/DUAL_SSL/dual_ssl_class_100_100/"
    DUAL_SSL_CLASS_050_050_PATH_PRE = "../out/DUAL_SSL/dual_ssl_class_050_050/"
    run_all_feature_metrics(model_paths=[DUAL_SSL_CLASS_050_050_PATH_PRE, DUAL_SSL_CLASS_100_100_PATH_PRE])

