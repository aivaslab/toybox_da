import time
import multiprocessing as mp
import numpy as np
import pickle
import statistics
from tabulate import tabulate
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata

import umap_metrics_utils as um
import get_activations as ga
import feature_space_metrics_utils as fum

DSET_CUTOFFS = {
    'tb_train': 0,
    'tb_test': 12,
    'in12_train': 24,
    'in12_test': 36
}

TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'horse', 'helicopter', 'mug', 'spoon', 'truck']

SCRATCH_PATH_PRE = "../out/random_init/"
TB_SUP_PATH_PRE = "../out/TB_SUP/tb_supervised_scratch_2/"
IN12_SUP_PATH_PRE = "../out/IN12_SUP/in12_supervised_scratch_final_1/"
IN12_DCL_PATH_PRE = "../out/IN12_SSL/dcl_ht_3/"
ILSVRC_PATH_PRE = "../out/ilsvrc_pretrained/"
DUAL_SUP_PATH_PRE = "../out/DUAL_SUP/TB_IN12/dual_sup_model_1/"
DUAL_SUP_24_PATH_PRE = "../out/DUAL_SUP_24/TB_IN12/dual_sup_24_model_1/"
DUAL_SUP_2CL_PATH_PRE = "../out/DUAL_SUP_2_CLASSIFIERS/TB_IN12/dual_sup_2cl_final_1/"
DUAL_SUP_JAN_PATH_PRE = "../out/DUAL_SUP_JAN/TB_IN12/dual_sup_jan_model_1/"
DUAL_SSL_JAN_PATH_PRE = "../out/DUAL_SSL/dual_ssl_class_trial_4/"

SCRATCH_PATH = "../out/JAN/TB_IN12/jan_scratch_final_1/"
TB_SUP_PATH = "../out/JAN/TB_IN12/jan_tb_sup_final_1/"
IN12_SUP_PATH = "../out/JAN/TB_IN12/jan_in12_sup_final_1/"
IN12_DCL_PATH = "../out/JAN/TB_IN12/jan_in12_dcl_final_1/"
ILSVRC_PATH = "../out/JAN/TB_IN12/jan_ilsvrc_final_1/"
DUAL_SUP_PATH = "../out/JAN/TB_IN12/jan_dual_sup_final_1/"
DUAL_SUP_24_PATH = "../out/JAN/TB_IN12/jan_dual_sup_24_final_1/"
DUAL_SUP_2CL_PATH = "../out/JAN/TB_IN12/jan_dual_sup_2cl_final_1/"
DUAL_SUP_JAN_PATH = "../out/JAN/TB_IN12/jan_dual_sup_jan_final_1/"
DUAL_SSL_JAN_PATH = "../out/JAN/TB_IN12/jan_dual_ssl_trial_4/"

JAN_DIFF_PRETRAINING_PATHS = {
        'scratch_pre': SCRATCH_PATH_PRE,
        'scratch': SCRATCH_PATH,
        'tb_sup_pre': TB_SUP_PATH_PRE,
        'tb_sup': TB_SUP_PATH,
        'in12_sup_pre': IN12_SUP_PATH_PRE,
        'in12_sup': IN12_SUP_PATH,
        'in12_dcl_pre': IN12_DCL_PATH_PRE,
        'in12_dcl': IN12_DCL_PATH,
        'ilsvrc_pre': ILSVRC_PATH_PRE,
        'ilsvrc': ILSVRC_PATH,
        'dual_sup_pre': DUAL_SUP_PATH_PRE,
        'dual_sup': DUAL_SUP_PATH,
        'dual_sup_24_pre': DUAL_SUP_24_PATH_PRE,
        'dual_sup_24': DUAL_SUP_24_PATH,
        'dual_sup_2cl_pre': DUAL_SUP_2CL_PATH_PRE,
        'dual_sup_2cl': DUAL_SUP_2CL_PATH,
        'dual_sup_jan_pre': DUAL_SUP_JAN_PATH_PRE,
        'dual_sup_jan': DUAL_SUP_JAN_PATH,
        'dual_ssl_jan_pre': DUAL_SSL_JAN_PATH_PRE,
        'dual_ssl_jan': DUAL_SSL_JAN_PATH
    }


def generate_activations_with_final_model(path, jan=False, btlnk=False):
    model_path = path + "final_model.pt"
    out_path = path + "/analysis/final_model/"
    ga.get_activations_sup(model_path=model_path, out_path=out_path, jan=jan, btlnk=btlnk)


def get_pairwise_correlations(file_path, norm, outlier_removal, model_types):
    assert len(model_types) == 2
    ll_dicts = []
    for model_type in model_types:
        assert model_type in ["kde", "gmm", "discrete_emd", "gmm_emd"]
        assert (outlier_removal in ["raw", "lcc", "acc"] and model_type == "kde") or (
                outlier_removal == "acc" and model_type == "gmm") or (
                       outlier_removal == "acc" and model_type == "discrete_emd") or (
                       outlier_removal == "acc" and model_type == "gmm_emd")
        if norm:
            dic_fname = file_path + f"ll/{outlier_removal}/{model_type}_ll_norm.pkl"

        else:
            dic_fname = file_path + f"ll/{outlier_removal}/{model_type}_ll.pkl"

        with open(dic_fname, "rb") as dic_fp:
            ll_dict = pickle.load(dic_fp)
        ll_dicts.append(ll_dict)
    calculate_pairwise_correlations(ll_dicts[0], ll_dicts[1])


def generate_umap_metrics(file_path, norm, outlier_removal, model_type):
    assert model_type in ["kde", "gmm", "discrete_emd", "gmm_emd"]
    assert (outlier_removal in ["raw", "lcc", "acc"] and model_type == "kde") or (
                outlier_removal == "acc" and model_type == "gmm") or (
                outlier_removal == "acc" and model_type == "discrete_emd") or (
                outlier_removal == "acc" and model_type == "gmm_emd")

    if norm:
        dic_fname = file_path + f"ll/{outlier_removal}/{model_type}_ll_norm.pkl"

    else:
        dic_fname = file_path + f"ll/{outlier_removal}/{model_type}_ll.pkl"

    dic_fp = open(dic_fname, "rb")
    ll_dic = pickle.load(dic_fp)
    display_ll_tables(dic=ll_dic)
    dic_fp.close()
    print(file_path)

    print("----------------------------------------------------------------------------------------------------")


def tabulated(df):
    """Tabulate a pandas dataframe"""
    return tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f")


def get_ll_arr(data_dict):
    ll_arr = np.zeros(shape=(48, 48), dtype=float)
    for key in data_dict.keys():
        src_d, src_cl, trgt_d, trgt_cl = key
        row, col = DSET_CUTOFFS[src_d] + TB_CLASSES.index(src_cl), DSET_CUTOFFS[trgt_d] + TB_CLASSES.index(trgt_cl)
        try:
            ll_arr[row][col] = statistics.fmean(data_dict[key])
        except TypeError:
            ll_arr[row][col] = data_dict[key]
    return ll_arr


def get_flattened_ll_arrays(data_dict):
    ll_arr = get_ll_arr(data_dict=data_dict)
    tb_train_arr = ll_arr[:len(TB_CLASSES), :len(TB_CLASSES)]
    tb_test_arr = ll_arr[len(TB_CLASSES):2 * len(TB_CLASSES), len(TB_CLASSES):2 * len(TB_CLASSES)]
    in12_train_arr = ll_arr[2 * len(TB_CLASSES):3 * len(TB_CLASSES), 2 * len(TB_CLASSES):3 * len(TB_CLASSES)]
    in12_test_arr = ll_arr[3 * len(TB_CLASSES):4 * len(TB_CLASSES), 3 * len(TB_CLASSES):4 * len(TB_CLASSES)]
    ll_arr_dict = {
        'tb_train': np.ravel(tb_train_arr),
        'tb_test': np.ravel(tb_test_arr),
        'in12_train': np.ravel(in12_train_arr),
        'in12_test': np.ravel(in12_test_arr)
    }

    return ll_arr_dict


def calculate_pairwise_correlations(data_dict_1, data_dict_2):
    ll_arr_dict_1, ll_arr_dict_2 = get_flattened_ll_arrays(data_dict_1), get_flattened_ll_arrays(data_dict_2)
    for key in ll_arr_dict_1.keys():
        print_str = f"{key}: "
        arr_1, arr_2 = ll_arr_dict_1[key], ll_arr_dict_2[key]
        pearson_corr = pearsonr(arr_1, arr_2)
        print_str += f"pearson corr: {round(pearson_corr.statistic, 3)} (p={round(pearson_corr.pvalue, 3):.3f}) "
        spearman_corr = spearmanr(arr_1, arr_2)
        print_str += f"spearman corr: {round(spearman_corr.statistic, 3)} (p={round(spearman_corr.pvalue, 3):.3f}) "
        arr_rank_1, arr_rank_2 = rankdata(arr_1, method="ordinal"), rankdata(arr_2, method="ordinal")
        kendal_tau = kendalltau(arr_rank_1, arr_rank_2)
        print_str += f"kendal tau: {round(kendal_tau.statistic, 3)} (p={round(kendal_tau.pvalue, 3):.3f}) "
        print(print_str)


def display_ll_tables(dic):
    """Beautify table"""
    assert len(list(dic.keys())) == 2304

    arr = get_ll_arr(data_dict=dic)

    pd.options.display.float_format = '{:.2f}'.format
    arr_df = pd.DataFrame(arr[:len(TB_CLASSES), :len(TB_CLASSES)], columns=TB_CLASSES)
    arr_df.index = TB_CLASSES
    print("Toybox Train")
    print(tabulated(arr_df))

    arr_df = pd.DataFrame(arr[len(TB_CLASSES):2 * len(TB_CLASSES), len(TB_CLASSES):2 * len(TB_CLASSES)],
                          columns=TB_CLASSES)
    arr_df.index = TB_CLASSES
    print("Toybox Test")
    print(tabulated(arr_df))

    arr_df = pd.DataFrame(arr[2 * len(TB_CLASSES):3 * len(TB_CLASSES), 2 * len(TB_CLASSES):3 * len(TB_CLASSES)],
                          columns=TB_CLASSES)
    arr_df.index = TB_CLASSES
    print("IN-12 Train")
    print(tabulated(arr_df))

    arr_df = pd.DataFrame(arr[3 * len(TB_CLASSES):4 * len(TB_CLASSES), 3 * len(TB_CLASSES):4 * len(TB_CLASSES)],
                          columns=TB_CLASSES)
    arr_df.index = TB_CLASSES
    print("IN-12 Test")
    print(tabulated(arr_df))


def generate_distinctness_metrics(file_path, norm, outlier_removal, model_type):
    all_models = ["kde", "gmm", "discrete_emd", "gmm_emd",
                  "feats/gmm_ll", "feats/gmm_ll_raw",  "feats/gmm_emd", "feats/gmm_emd_raw"]
    assert model_type in all_models
    assert (outlier_removal in ["raw", "lcc", "acc"] and model_type == "kde") or (
                outlier_removal == "acc" and model_type in all_models)
    if "feats" in model_type:
        if norm:
            dic_fname = file_path + f"ll/{outlier_removal}/{model_type}_norm.pkl"
        else:
            dic_fname = file_path + f"ll/{outlier_removal}/{model_type}.pkl"
    else:
        if norm:
            dic_fname = file_path + f"ll/{outlier_removal}/{model_type}_ll_norm.pkl"
        else:
            dic_fname = file_path + f"ll/{outlier_removal}/{model_type}_ll.pkl"

    dic_fp = open(dic_fname, "rb")
    ll_dic = pickle.load(dic_fp)

    dsets = ['tb_train', 'tb_test', 'in12_train', 'in12_test']
    ll_arr_dict = {}
    for dset in dsets:
        ll_arr = []
        for src_cl in TB_CLASSES:
            self_ll = round(statistics.fmean(ll_dic[(dset, src_cl, dset, src_cl)]), 3) \
                if model_type != "discrete_emd" else ll_dic[(dset, src_cl, dset, src_cl)]
            cnt = 0.
            tot = 0.0
            max_cross_ll = -np.inf
            for trgt_cl in TB_CLASSES:
                if src_cl != trgt_cl:
                    cnt += 1  # 2
                    cross_ll_1 = round(statistics.fmean(ll_dic[(dset, src_cl, dset, trgt_cl)]), 3) \
                        if model_type != "discrete_emd" else ll_dic[(dset, src_cl, dset, trgt_cl)]
                    # cross_ll_2 = round(statistics.fmean(ll_dic[(dset, trgt_cl, dset, src_cl)]), 3)
                    tot += cross_ll_1  # + cross_ll_2
                    max_cross_ll = max(max_cross_ll, cross_ll_1)  # , cross_ll_2)
            avg_cross_ll = round(tot / cnt, 3)
            if model_type in ["gmm_emd", "discrete_emd", "feats/gmm_emd", "feats/gmm_emd_raw"]:
                ll_arr.append([src_cl, self_ll, avg_cross_ll, avg_cross_ll - self_ll,
                               max_cross_ll, max_cross_ll - self_ll])
            else:
                ll_arr.append([src_cl, self_ll, avg_cross_ll, self_ll - avg_cross_ll,
                               max_cross_ll, self_ll - max_cross_ll])
        ll_arr_dict[dset] = ll_arr

    dic_fp.close()
    return ll_arr_dict


def run_all_metrics_on_model(model_path, norm):
    umap_dir_path = model_path + "/analysis/final_model/backbone/umap/all_data/umap_200_0.1_euclidean/"
    activation_path = model_path + "/analysis/final_model/backbone/activations/"
    st_time, total_time = time.time(), 0.0
    print_str = "Time taken: "

    ##########################
    # Ran the following ones #
    ##########################

    # um.remove_outliers(p=umap_dir_path, norm=norm)
    # print_str += f"outlier removal: {time.time() - st_time - total_time:.2f}s "
    # total_time = time.time() - st_time
    #
    # um.compute_kde_likelihood_dict(p=umap_dir_path, norm=norm, outlier_removal="acc")
    # print_str += f"kde_ll: {time.time() - st_time - total_time:.2f}s "
    # total_time = time.time() - st_time
    #
    # um.compute_gmm_likelihood_dict(p=umap_dir_path, norm=norm)
    # print_str += f"gmm_ll: {time.time() - st_time - total_time:.2f}s "
    # total_time = time.time() - st_time
    #
    # um.compute_gmm_emd_likelihood_dict(p=umap_dir_path, norm=norm)
    # print_str += f"gmm_emd: {time.time() - st_time - total_time:.2f}s "
    # total_time = time.time() - st_time
    #
    # fum.extract_feats_from_umap(umap_path=umap_dir_path, act_path=activation_path, norm=norm)
    # print_str += f"feats_ex: {time.time() - st_time - total_time:.2f}s "
    # total_time = time.time() - st_time
    #
    # fum.get_feat_gmm_likelihoods(umap_path=umap_dir_path, act_path=activation_path, norm=False, use_umap=norm)
    # print_str += f"feats_gmm_ll_raw: {time.time() - st_time - total_time:.2f}s "
    # total_time = time.time() - st_time

    # fum.get_feat_gmm_likelihoods(umap_path=umap_dir_path, act_path=activation_path, norm=norm)
    # print_str += f"feats_gmm_ll: {time.time() - st_time - total_time:.2f}s "
    # total_time = time.time() - st_time

    ##################################
    # Did not run the following ones #
    ##################################

    fum.gen_feat_gmm_emd(umap_path=umap_dir_path, norm=norm)
    print_str += f"feats_gmm_emd: {time.time() - st_time - total_time:.2f}s "
    total_time = time.time() - st_time

    fum.gen_feat_gmm_emd(umap_path=umap_dir_path, norm=norm, use_umap=False)
    print_str += f"feats_gmm_emd_raw: {time.time() - st_time - total_time:.2f}s "

    print_str += f"total: {time.time() - st_time:.2f}s "
    print(print_str)


def run_all_metrics(model_paths):
    # mp.set_start_method('forkserver')
    start_time = time.time()
    num_procs = 2
    idx = 0
    while idx < len(model_paths):
        curr_procs = []
        while idx < len(model_paths) and len(curr_procs) < num_procs:
            model_path = model_paths[idx]
            new_proc = mp.Process(target=run_all_metrics_on_model, args=(model_path, False))
            curr_procs.append(new_proc)
            idx += 1

        for proc in curr_procs:
            proc.start()

        for proc in curr_procs:
            proc.join()
    print(f"Ran metrics on {len(model_paths)} models, took {time.time() - start_time:.2f}s")


def get_metric_correlations_for_model(model_path):
    umap_dir_path = model_path + "/analysis/final_model/backbone/umap/all_data/umap_200_0.1_euclidean/"
    get_pairwise_correlations(file_path=umap_dir_path, norm=False, outlier_removal="acc",
                              model_types=["kde", "gmm_emd"])


def run_metrics_on_jan_models():
    run_all_metrics(model_paths=list(JAN_DIFF_PRETRAINING_PATHS.values()))


if __name__ == "__main__":
    DUAL_SSL_CLASS_100_100_PATH_PRE = "../out/DUAL_SSL/dual_ssl_class_100_100/"
    DUAL_SSL_CLASS_050_050_PATH_PRE = "../out/DUAL_SSL/dual_ssl_class_050_050/"
    DUAL_SSL_CLASS_025_025_PATH_PRE = "../out/DUAL_SSL/dual_ssl_class_025_025/"
    DUAL_SSL_CLASS_075_075_PATH_PRE = "../out/DUAL_SSL/dual_ssl_class_075_075/"
    # run_all_metrics(model_paths=[DUAL_SSL_CLASS_050_050_PATH_PRE, DUAL_SSL_CLASS_100_100_PATH_PRE])
    #                              DUAL_SSL_CLASS_025_025_PATH_PRE, DUAL_SSL_CLASS_075_075_PATH_PRE])
    # run_metrics_on_jan_models()
    for model_name, path in JAN_DIFF_PRETRAINING_PATHS.items():
        print(model_name)
        get_metric_correlations_for_model(model_path=path)
