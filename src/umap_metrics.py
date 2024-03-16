"""Code to preprocess and compute overlap metric from the umap data"""

import os
import csv
import numpy as np
import pickle
import pandas as pd
import time

UMAP_FILENAMES = ["tb_train.csv", "tb_test.csv", "in12_train.csv", "in12_test.csv"]
NORM_UMAP_FILENAMES = ["tb_train_norm.csv", "tb_test_norm.csv", "in12_train_norm.csv", "in12_test_norm.csv"]
TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck',
              'giraffe', 'horse', 'helicopter', 'mug', 'spoon', 'truck']

SRC_FILES = {
    'tb_train': "../data/data_12/Toybox/toybox_data_interpolated_cropped_dev.csv",
    'tb_test': "../data/data_12/Toybox/toybox_data_interpolated_cropped_test.csv",
    'in12_train': "../data/data_12/IN-12/dev.csv",
    'in12_test': "../data/data_12/IN-12/test.csv"
}

for k, val in SRC_FILES.items():
    assert os.path.isfile(val)


def normalize(path):
    """normalize all the umaps"""
    tb_train_data = []
    tb_test_data = []
    in12_train_data = []
    in12_test_data = []
    for fname in UMAP_FILENAMES:
        fpath = path + fname
        assert os.path.isfile(fpath)
        fp = open(fpath, "r")
        if "tb_train" in fname:
            tb_train_data = list(csv.DictReader(fp))
        elif "tb_test" in fname:
            tb_test_data = list(csv.DictReader(fp))
        elif "in12_train" in fname:
            in12_train_data = list(csv.DictReader(fp))
        else:
            in12_test_data = list(csv.DictReader(fp))
    len_tb_tr = len(tb_train_data)
    len_tb_te = len(tb_test_data)
    len_in12_tr = len(in12_train_data)
    len_in12_te = len(in12_test_data)
    assert len_tb_tr > 0 and len_tb_te > 0 and len_in12_tr > 0 and len_in12_te > 0
    # print(len_tb_tr, len_tb_te, len_in12_tr, len_in12_te, total_len)
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    all_datasets = [tb_train_data, tb_test_data, in12_train_data, in12_test_data]
    for dataset in all_datasets:
        for datapoint in dataset:
            x_min = min(x_min, float(datapoint['x']))
            x_max = max(x_max, float(datapoint['x']))
            y_min = min(y_min, float(datapoint['y']))
            y_max = max(y_max, float(datapoint['y']))
    
    # print(x_min, x_max, y_min, y_max)
    x_gap = x_max - x_min
    y_gap = y_max - y_min
    
    norm_xmin, norm_xmax = float("inf"), float("-inf")
    norm_ymin, norm_ymax = float("inf"), float("-inf")
    
    for out_fname, dataset in zip(NORM_UMAP_FILENAMES, all_datasets):
        out_fp = open(path + out_fname, "w")
        out_csv = csv.writer(out_fp)
        out_csv.writerow(["idx", "x", "y"])
        for datapoint in dataset:
            idx, x, y = datapoint['idx'], float(datapoint['x']), float(datapoint['y'])
            x_scaled = 2 * (x - x_min) / x_gap - 1
            y_scaled = 2 * (y - y_min) / y_gap - 1
            norm_xmin = min(norm_xmin, x_scaled)
            norm_xmax = max(norm_xmax, x_scaled)
            norm_ymin = min(norm_ymin, y_scaled)
            norm_ymax = max(norm_ymax, y_scaled)
            out_csv.writerow([idx, x_scaled, y_scaled])
        out_fp.close()
    # print(norm_xmin, norm_xmax, norm_ymin, norm_ymax)
    assert norm_xmin == -1.0
    assert norm_xmax == 1.0
    assert norm_ymin == -1.0
    assert norm_ymax == 1.0


def get_points(path, dataset, target_cl, norm=True):
    """Get a 2d matrix of datapoints for the given path, dataset and cl"""
    if norm:
        fname = dataset + "_norm.csv"
    else:
        fname = dataset + ".csv"
    fpath = path + fname
    if norm:
        assert fname in NORM_UMAP_FILENAMES
    else:
        assert fname in UMAP_FILENAMES
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
    try:
        assert ret_data_np.shape == (1500, 2)
    except AssertionError:
        print(f"Dataset: {dataset}  Cl: {target_cl} did not match shape. Expected: (1500, 2)  "
              f"Output: {ret_data_np.shape}")
    fp.close()
    src_data_fp.close()
    return ret_data_np


def get_all_kde(dataset, cl, all_points):
    """Get all kde"""
    train_points = all_points[(dataset, cl)]
    train_data = np.array(train_points)
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    # import time
    # st_time = time.time()
    bandwidths = 10 ** np.linspace(-1, 0, 5)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=5,
                        n_jobs=-1,
                        verbose=1)
    grid.fit(train_data)
    kernel = grid.best_estimator_
    kernel.fit(train_data)
    ll_dict = dict()
    for eval_dset in ['tb_train', "in12_train"]:
        for eval_cl in TB_CLASSES:
            eval_points = all_points[(eval_dset, eval_cl)]
            eval_data = np.array(eval_points)  # np.vstack([eval_x, eval_y])
            eval_likelihood = kernel.score_samples(eval_data)
            ll_dict[(eval_dset, eval_cl)] = eval_likelihood
    # print(dataset, cl, time.time() - st_time)
    return ll_dict


def make_ll_tbl(save_path, dic):
    """Beautify table"""
    assert len(list(dic.keys())) == 576
    
    arr = np.zeros(shape=(24, 24), dtype=float)
    for key in dic.keys():
        src_d, src_cl, trgt_d, trgt_cl = key
        row = TB_CLASSES.index(src_cl)
        if src_d == "in12_train":
            row += 12
        col = TB_CLASSES.index(trgt_cl)
        if trgt_d == "in12_train":
            col += 12
        
        arr[row][col] = sum(dic[key])
    
    np.savetxt(save_path + "ll.csv", arr, delimiter=',')
    
    arr = np.zeros(shape=(2, 12), dtype=float)
    for i, cl in enumerate(TB_CLASSES):
        lls = dic[("tb_train", cl, "in12_train", cl)]
        arr[0][i] = sum(lls)
        
        lls = dic[("in12_train", cl, "tb_train", cl)]
        arr[1][i] = sum(lls)
    
    arr_df = pd.DataFrame(arr, columns=TB_CLASSES)
    arr_df.to_csv(save_path + "ll_domain.csv")
    print(arr_df)
    
    arr_df_mean = arr_df.mean(axis=1)
    print("LL_domain: \n", arr_df_mean)
    arr_df_mean = np.array([arr_df_mean.mean(axis=0)])
    # arr_df_mean.to_csv(save_path + "ll_domain_mean.csv")
    np.savetxt(save_path + "ll_domain_mean.csv", arr_df_mean, delimiter=',')
    print("ll_domain:", arr_df_mean)
    # np.savetxt(save_path + "ll_domain.csv", arr, delimiter=',')


def main(p):
    """Main method"""
    normalize(path=p)
    datapoints_dict = {}
    likelihood_dict = {}
    for dset in ['tb_train', "in12_train"]:  # , 'tb_test', 'in12_train', 'in12_test']:
        for tb_cl in TB_CLASSES:
            datapoints = get_points(path=p, dataset=dset, target_cl=tb_cl)
            datapoints_dict[(dset, tb_cl)] = datapoints
    
    for dset in ['tb_train', 'in12_train']:
        for tb_cl in TB_CLASSES:
            dic = get_all_kde(dataset=dset, cl=tb_cl, all_points=datapoints_dict)
            for key in dic.keys():
                likelihood_dict[(dset, tb_cl, key[0], key[1])] = dic[key]
    
    print(len(list(likelihood_dict.keys())))
    
    data_path = p + "/data/"
    os.makedirs(data_path, exist_ok=True)
    datapoints_fname = data_path + "datapoints.pkl"
    datapoints_fp = open(datapoints_fname, "wb")
    pickle.dump(datapoints_dict, datapoints_fp)
    datapoints_fp.close()
    
    ll_fname = data_path + "ll.pkl"
    ll_fp = open(ll_fname, "wb")
    pickle.dump(likelihood_dict, ll_fp)
    ll_fp.close()


if __name__ == "__main__":
    start_time = time.time()
    for ep in range(0, 101, 10):
        file_path = f"../out/DUAL_SUP/TB_IN12/exp_Feb_15_2024_15_18/umap_epoch_{ep}_all_data/umap_300_0.1_euclidean/"
        main(p=file_path)
        
        # dic_fname = file_path + "data/ll.pkl"
        # dpath = file_path + "data/"
        # dic_fp = open(dic_fname, "rb")
        # ll_dic = pickle.load(dic_fp)
        # make_ll_tbl(save_path=dpath, dic=ll_dic)
        # dic_fp.close()
        print(file_path, time.time() - start_time)
        start_time = time.time()
        
        print("----------------------------------------------------------------------------------------------------")
    