"""Util files for computing the metrics"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import pandas as pd
import math
import heapq
from matplotlib.lines import Line2D
import networkx as nx
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.neighbors import KernelDensity
from scipy import stats

import pickle

import umap_preprocess_for_metrics

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


def get_points(path, dataset, target_cl):
    """Get a 2d matrix of datapoints for the given path, dataset and cl"""
    fname = dataset + "_norm.csv"
    fpath = path + fname
    assert fname in NORM_UMAP_FILENAMES
    assert os.path.isfile(fpath)
    assert target_cl in TB_CLASSES
    fp = open(fpath, "r")
    data = list(csv.DictReader(fp))
    len_data = len(data)
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
    print(dataset, target_cl, ret_data_np.shape)
    fp.close()
    src_data_fp.close()
    return ret_data_np
    
    
def plot_points(path, dataset, target_cl, points):
    """Plot the points and save them"""
    scatter_out_path = path + "images/scatter/{}/".format(dataset)
    # print(scatter_out_path)
    os.makedirs(scatter_out_path, exist_ok=True)
    plt.scatter(points[:, 0], points[:, 1])
    plt.savefig(scatter_out_path + "{}.png".format(target_cl))
    plt.close()
    
    
def build_and_compute_mst(path, dataset, target_cl, points):
    scatter_out_path = path + "images/scatter/{}/".format(dataset)
    graph = nx.Graph()
    node_dic = {}
    min_x, max_x, min_y, max_y = float("inf"), float("-inf"), float("inf"), float("-inf")
    for i in range(len(points)):
        node_dic[i] = (points[i][0], points[i][1])
        min_x, max_x = min(min_x, points[i][0]), max(max_x, points[i][0])
        min_y, max_y = min(min_y, points[i][1]), max(max_y, points[i][1])
        
        for j in range(i+1, len(points)):
            dist = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            graph.add_edge(i, j, weight=dist)
            graph.add_edge(j, i, weight=dist)
            
    min_x -= 0.05
    max_x += 0.05
    min_y -= 0.05
    max_y += 0.05
            
    mst = nx.minimum_spanning_tree(graph)

    fig, ax = plt.subplots(1, 1)  # or what ever layout you want
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    nx.draw_networkx(mst, pos=node_dic, with_labels=False, node_size=10, ax=ax)
    plt.savefig(scatter_out_path + "{}_mst_nx.png".format(target_cl))
    plt.close()
    
    edge_weights = []
    for edge in mst.edges():
        u, v = edge
        dist = math.sqrt((points[u][0] - points[v][0]) ** 2 + (points[u][1] - points[v][1]) ** 2)
        edge_weights.append(dist)
    quantiles = np.quantile(edge_weights, [0.025, 0.975])
    threshold = quantiles[1] + 1.5 * (quantiles[1] - quantiles[0])
    removed_edges = []
    for edge in mst.edges():
        u, v = edge
        dist = math.sqrt((points[u][0] - points[v][0]) ** 2 + (points[u][1] - points[v][1]) ** 2)
        if dist > threshold:
            removed_edges.append(edge)
    print(len(removed_edges))
    for e in removed_edges:
        mst.remove_edge(e[0], e[1])
    
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    nx.draw_networkx(mst, pos=node_dic, with_labels=False, node_size=10, ax=ax)
    plt.savefig(scatter_out_path + "{}_mst_thres.png".format(target_cl))
    plt.close()
    
    largest_cc = max(nx.connected_components(mst), key=len)
    largest_cc_g = mst.subgraph(largest_cc).copy()

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    nx.draw_networkx(largest_cc_g, pos=node_dic, with_labels=False, node_size=10, ax=ax)
    plt.savefig(scatter_out_path + "{}_largest_cc.png".format(target_cl))
    plt.close()
    
    ll = set(list(largest_cc_g.nodes))
    return ll
    # print(largest_cc)


def convex_hull(path, dataset, target_cl, points):
    """Compute convex hull"""
    points_np = np.array(points)
    # print(points_np.shape)
    scatter_out_path = path + "images/scatter/{}/".format(dataset)
    hull = ConvexHull(points_np)
    hull_area = hull.volume
    plt.plot(points_np[:, 0], points_np[:, 1], 'o')

    for simplex in hull.simplices:
        plt.plot(points_np[simplex, 0], points_np[simplex, 1], 'k-')

    plt.plot(points_np[hull.vertices, 0], points_np[hull.vertices, 1], 'r--', lw=2)
    plt.plot(points_np[hull.vertices[0], 0], points_np[hull.vertices[0], 1], 'ro')

    plt.savefig(scatter_out_path + "{}_hull.png".format(target_cl))
    plt.close()
    return hull_area, hull.vertices
    
    
def get_intersection(hull_1, hull_2):
    from SH import PolygonClipper
    poly1, poly2 = np.array(hull_1), np.array(hull_2)
    clip = PolygonClipper(warn_if_empty=True)
    clipped_poly = clip(poly1, poly2)
    print(poly1.shape, poly2.shape, clipped_poly.shape)


def kde(path, dataset, target_cl, m1, m2):
    xmin = min(m1)
    xmax = max(m1)
    ymin = min(m2)
    ymax = max(m2)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap=plt.cm.get_cmap('viridis'), extent=[xmin, xmax, ymin, ymax])
    ax.plot(m1, m2, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    scatter_out_path = path + "images/scatter/{}/".format(dataset)
    plt.savefig(scatter_out_path + "{}_kde.png".format(target_cl))
    plt.close()


def compute_kde(path, dataset, target_cl, points):
    x = [points[i][0] for i in range(len(points))]
    y = [points[i][1] for i in range(len(points))]
    
    kde(path=path, dataset=dataset, target_cl=target_cl, m1=x, m2=y)
    
    
def get_all_kde(path, dataset, cl, all_points, ll_dict):
    """Get all kde"""
    train_points = all_points[(dataset, cl)]
    train_data = np.array(train_points)
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    import time
    st_time = time.time()
    bandwidths = 10 ** np.linspace(-1, 0, 5)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=5,
                        n_jobs=-1,
                        verbose=1)
    grid.fit(train_data)
    kernel = grid.best_estimator_
    kernel.fit(train_data)
    for eval_dset in ['tb_train', "in12_train"]:
        for eval_cl in TB_CLASSES:
            eval_points = all_points[(eval_dset, eval_cl)]
            eval_data = np.array(eval_points)  # np.vstack([eval_x, eval_y])
            eval_likelihood = kernel.score_samples(eval_data)
            ll_dict[(dataset, cl, eval_dset, eval_cl)] = eval_likelihood
    print(dataset, cl, time.time() - st_time)
    return ll_dict
    

def preprocess(path):
    """Prepare umap data"""
    umap_preprocess_for_metrics.normalize(path=path)
    umap_preprocess_for_metrics.generate_scatter_plots(path=path)
    
    
def main(p):
    """Main method"""
    preprocess(path=p)
    datapoints_dict = {}
    core_points_dict = {}
    hull_points_dict = {}
    likelihood_dict = {}
    hull_areas = np.zeros(shape=(2, 12), dtype=float)
    num_core_points = np.zeros(shape=(2, 12), dtype=int)
    density = np.zeros(shape=(2, 12), dtype=int)
    for dset in ['tb_train', "in12_train"]:  # , 'tb_test', 'in12_train', 'in12_test']:
        for tb_cl in TB_CLASSES:
            datapoints = get_points(path=p, dataset=dset, target_cl=tb_cl)
            datapoints_dict[(dset, tb_cl)] = datapoints
            plot_points(path=p, dataset=dset, target_cl=tb_cl, points=datapoints)
            core_point_idxs = build_and_compute_mst(path=p, dataset=dset, target_cl=tb_cl, points=datapoints)
            core_points = [datapoints[i] for i in range(len(datapoints)) if i in core_point_idxs]
            core_points_dict[(dset, tb_cl)] = core_points
            hull_area, hull_idxs = convex_hull(path=p, dataset=dset, target_cl=tb_cl, points=core_points)
            hull_points = [core_points[i] for i in range(len(core_points)) if i in hull_idxs]
            hull_points_dict[(dset, tb_cl)] = hull_points
            
            ha_row = 0 if dset == "tb_train" else 1
            ha_col = TB_CLASSES.index(tb_cl)
            hull_areas[ha_row][ha_col] = hull_area
            num_core_points[ha_row][ha_col] = len(core_point_idxs)
            density[ha_row][ha_col] = len(core_point_idxs) / hull_area
        
            # compute_kde(points=core_points, path=p, dataset=dset, target_cl=tb_cl)
            # print(hull_points)
            # print(type(hull_points), len(hull_points), hull_points[0])
            # get_intersection(hull_1=hull_points, hull_2=hull_points)
        
            # mst(path=p, dataset=dset, target_cl=tb_cl, points=datapoints)

    for dset in ['tb_train', 'in12_train']:
        for tb_cl in TB_CLASSES:
            likelihood_dict = get_all_kde(path=p, dataset=dset, cl=tb_cl,
                                          all_points=datapoints_dict, ll_dict=likelihood_dict)

    print(len(list(likelihood_dict.keys())))

    data_path = p + "/data/"

    os.makedirs(data_path, exist_ok=True)
    datapoints_fname = data_path + "datapoints.pkl"
    corepoints_fname = data_path + "corepoints.pkl"
    hullpoints_fname = data_path + "hullpoints.pkl"
    datapoints_fp = open(datapoints_fname, "wb")
    corepoints_fp = open(corepoints_fname, "wb")
    hullpoints_fp = open(hullpoints_fname, "wb")
    pickle.dump(datapoints_dict, datapoints_fp)
    pickle.dump(core_points_dict, corepoints_fp)
    pickle.dump(hull_points_dict, hullpoints_fp)
    corepoints_fp.close()
    hullpoints_fp.close()
    datapoints_fp.close()

    ll_fname = data_path + "ll.pkl"
    ll_fp = open(ll_fname, "wb")
    pickle.dump(likelihood_dict, ll_fp)
    ll_fp.close()
    hull_area_df = pd.DataFrame(hull_areas, columns=TB_CLASSES)
    hull_area_df.to_csv(data_path + "hull_area.csv")

    num_core_points_df = pd.DataFrame(num_core_points, columns=TB_CLASSES)
    num_core_points_df.to_csv(data_path + "num_core_points_.csv")
    
    density_df = pd.DataFrame(density, columns=TB_CLASSES)
    density_df.to_csv(data_path + "density.csv")
    
    mean_density_df = density_df.mean(axis=1)
    mean_density_df.to_csv(data_path + "mean_density.csv")
    
    print(density_df)
    print(mean_density_df)
    
    
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
        
    np.savetxt(save_path+"ll.csv", arr, delimiter=',')
    
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
    print("LL_domain: ", arr_df_mean)
    arr_df_mean = np.array([arr_df_mean.mean(axis=0)])
    # arr_df_mean.to_csv(save_path + "ll_domain_mean.csv")
    np.savetxt(save_path + "ll_domain_mean.csv", arr_df_mean, delimiter=',')
    print("ll_domain:", arr_df_mean)
    # np.savetxt(save_path + "ll_domain.csv", arr, delimiter=',')
    

if __name__ == "__main__":
    file_path = "../ICLR_OUT/TB_SUP_IN12_SCRAMBLED/exp_Sep_29_2023_01_09/umap_all_data/umap_200_0.1_euclidean/"
    main(p=file_path)
    
    dic_fname = file_path + "data/ll.pkl"
    data_path = file_path + "data/"
    dic_fp = open(dic_fname, "rb")
    ll_dic = pickle.load(dic_fp)
    make_ll_tbl(save_path=data_path, dic=ll_dic)
    dic_fp.close()
    print(file_path)
    
    print("----------------------------------------------------------------------------------------------------")

    file_path = "../ICLR_OUT/TB_SUP_IN12_SCRAMBLED_MMD/exp_Sep_28_2023_06_00/umap_all_data/umap_200_0.1_euclidean/"
    main(p=file_path)

    dic_fname = file_path + "data/ll.pkl"
    data_path = file_path + "data/"
    dic_fp = open(dic_fname, "rb")
    ll_dic = pickle.load(dic_fp)
    make_ll_tbl(save_path=data_path, dic=ll_dic)
    dic_fp.close()

    print(file_path)

    print("----------------------------------------------------------------------------------------------------")

    # file_path = "../ICLR_OUT/TB_IN12_R/exp_Sep_28_2023_16_32/umap_all_data/umap_200_0.1_euclidean/"
    # main(p=file_path)
    #
    # dic_fname = file_path + "data/ll.pkl"
    # data_path = file_path + "data/"
    # dic_fp = open(dic_fname, "rb")
    # ll_dic = pickle.load(dic_fp)
    # make_ll_tbl(save_path=data_path, dic=ll_dic)
    # dic_fp.close()
    #
    # print(file_path)
    #
    # print("----------------------------------------------------------------------------------------------------")
    # file_path = "../ICLR_OUT/DUAL_SUP_CCMMD/exp_Sep_27_2023_00_42/umap_all_data/umap_200_0.1_euclidean/"
    # main(p=file_path)
    #
    # dic_fname = file_path + "data/ll.pkl"
    # data_path = file_path + "data/"
    # dic_fp = open(dic_fname, "rb")
    # ll_dic = pickle.load(dic_fp)
    # make_ll_tbl(save_path=data_path, dic=ll_dic)
    # dic_fp.close()
    #
    # print(file_path)
    #
    # print("----------------------------------------------------------------------------------------------------")
    # file_path = "/home/VANDERBILT/sanyald/Documents/AIVAS/Projects/toybox_da/ICLR_OUT/IN12_SUP/" \
    #             "exp_Sep_25_2023_23_11/umap_all_data/umap_200_0.1_euclidean/"
    # main(p=file_path)
    #
    # dic_fname = file_path + "data/ll.pkl"
    # data_path = file_path + "data/"
    # dic_fp = open(dic_fname, "rb")
    # ll_dic = pickle.load(dic_fp)
    # make_ll_tbl(save_path=data_path, dic=ll_dic)
    # dic_fp.close()
    #
    # print(file_path)
    #
    # print("----------------------------------------------------------------------------------------------------")
    # file_path = "/home/VANDERBILT/sanyald/Documents/AIVAS/Projects/toybox_da/ICLR_OUT/DUAL_SUP_24/" \
    #             "exp_Aug_11_2023_10_06/umap_all_data/umap_200_0.1_euclidean/"
    # main(p=file_path)
    #
    # dic_fname = file_path + "data/ll.pkl"
    # data_path = file_path + "data/"
    # dic_fp = open(dic_fname, "rb")
    # ll_dic = pickle.load(dic_fp)
    # make_ll_tbl(save_path=data_path, dic=ll_dic)
    # dic_fp.close()
    #
    # print("----------------------------------------------------------------------------------------------------")
    # file_path = "/home/VANDERBILT/sanyald/Documents/AIVAS/Projects/toybox_da/ICLR_OUT/DUAL_SUP_CCMMD/" \
    #             "exp_Aug_21_2023_14_54/umap_all_data/umap_200_0.1_euclidean/"
    # main(p=file_path)
    #
    # dic_fname = file_path + "data/ll.pkl"
    # data_path = file_path + "data/"
    # dic_fp = open(dic_fname, "rb")
    # ll_dic = pickle.load(dic_fp)
    # make_ll_tbl(save_path=data_path, dic=ll_dic)
    # dic_fp.close()
    #
    # print("----------------------------------------------------------------------------------------------------")
