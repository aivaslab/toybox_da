"""Module containing code to preprocess and load umap data"""
import os
import csv
import matplotlib.pyplot as plt


UMAP_FILENAMES = ["tb_train.csv", "tb_test.csv", "in12_train.csv", "in12_test.csv"]
NORM_UMAP_FILENAMES = ["tb_train_norm.csv", "tb_test_norm.csv", "in12_train_norm.csv", "in12_test_norm.csv"]


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
    total_len = len_tb_tr + len_tb_te + len_in12_tr + len_in12_te
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
    print(norm_xmin, norm_xmax, norm_ymin, norm_ymax)
    
    
def generate_scatter_plots(path):
    """Generate_scatter_plots for the unnormed and normed coordinates"""
    scatter_out_path = path + "images/scatter/"
    os.makedirs(scatter_out_path, exist_ok=True)
    for i, (fname, norm_fname) in enumerate(zip(UMAP_FILENAMES, NORM_UMAP_FILENAMES)):
        fpath, norm_fpath = path + fname, path + norm_fname
        fp, norm_fp = open(fpath, "r"), open(norm_fpath, "r")
        data, norm_data = list(csv.DictReader(fp)), list(csv.DictReader(norm_fp))
        assert len(data) == len(norm_data)
        data_x, data_y, norm_data_x, norm_data_y = [], [], [], []
        for dp, norm_dp in zip(data, norm_data):
            data_x.append(float(dp['x']))
            data_y.append(float(dp['y']))
            norm_data_x.append(float(norm_dp['x']))
            norm_data_y.append(float(norm_dp['y']))
        plt.scatter(data_x, data_y)
        plt.savefig(scatter_out_path+"data_{}.png".format(i))
        plt.close()
        plt.scatter(norm_data_x, norm_data_y)
        plt.savefig(scatter_out_path+"data_norm_{}.png".format(i))
        plt.close()
        
    
if __name__ == "__main__":
    p = "/home/VANDERBILT/sanyald/Documents/AIVAS/Meeting Slides/Summer 2023/Aug-04-2023/umaps/dual_sup/" \
        "umap_all_data/umap_100_0.1_euclidean/"
    normalize(path=p)
    generate_scatter_plots(path=p)
    
    