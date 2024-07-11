"""Util files for computing the metrics"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import networkx as nx
from scipy.spatial import ConvexHull
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
from tabulate import tabulate
from collections import defaultdict
from scipy.stats import pearsonr

import pickle

import umap_preprocess_for_metrics

UMAP_FILENAMES = ["tb_train.csv", "tb_test.csv", "in12_train.csv", "in12_test.csv"]
NORM_UMAP_FILENAMES = ["tb_train_norm.csv", "tb_test_norm.csv", "in12_train_norm.csv", "in12_test_norm.csv"]
TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck',
              'giraffe', 'horse', 'helicopter', 'mug', 'spoon', 'truck']
DATASETS = ["tb_train", "tb_test", "in12_train", "in12_test"]
DSET_CUTOFFS = {
    'tb_train':   0,
    'tb_test':    12,
    'in12_train': 24,
    'in12_test':  36
}

SRC_FILES = {
    'tb_train': "../data/data_12/Toybox/toybox_data_interpolated_cropped_dev.csv",
    'tb_test': "../data/data_12/Toybox/toybox_data_interpolated_cropped_test.csv",
    'in12_train': "../data/data_12/IN-12/dev.csv",
    'in12_test': "../data/data_12/IN-12/test.csv"
}

for k, val in SRC_FILES.items():
    assert os.path.isfile(val)


def get_points(path, dataset, target_cl, norm=True):
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

    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            graph.add_edge(i, j, weight=dist)
            graph.add_edge(j, i, weight=dist)
            
    min_x -= 0.05
    max_x += 0.05
    min_y -= 0.05
    max_y += 0.05
            
    mst = nx.minimum_spanning_tree(graph)

    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(24, 5))  # or what ever layout you want
    ax[0].set_xlim(min_x, max_x)
    ax[0].set_ylim(min_y, max_y)
    ax[0].set_title("MST of all datapoints")
    nx.draw_networkx(mst, pos=node_dic, with_labels=False, node_size=10, ax=ax[0])
    
    edge_weights = []
    for edge in mst.edges():
        u, v = edge
        dist = math.sqrt((points[u][0] - points[v][0]) ** 2 + (points[u][1] - points[v][1]) ** 2)
        edge_weights.append(dist)
    quantiles = np.quantile(edge_weights, [0.05, 0.95])
    threshold = quantiles[1] + 1.5 * (quantiles[1] - quantiles[0])

    removed_edges = []
    for edge in mst.edges():
        u, v = edge
        dist = math.sqrt((points[u][0] - points[v][0]) ** 2 + (points[u][1] - points[v][1]) ** 2)
        if dist > threshold:
            removed_edges.append(edge)
    # print(len(removed_edges))
    for e in removed_edges:
        mst.remove_edge(e[0], e[1])
    
    nx.draw_networkx(mst, pos=node_dic, with_labels=False, node_size=10, ax=ax[1])
    ax[1].set_title("MST of all datapoints (edges above threshold removed)")
    
    largest_cc = max(nx.connected_components(mst), key=len)
    largest_cc_g = mst.subgraph(largest_cc).copy()

    nx.draw_networkx(largest_cc_g, pos=node_dic, with_labels=False, node_size=10, ax=ax[2])
    ax[2].set_title("Largest connected component")

    new_mst = mst.copy()
    cc_rem_threshold = 5
    removed_cc = []
    all_cc = nx.connected_components(mst)
    for cc in all_cc:
        if len(cc) < cc_rem_threshold:
            removed_cc.append(cc)
    # print(removed_cc)
    for cc in removed_cc:
        for node in cc:
            new_mst.remove_node(node)

    nx.draw_networkx(new_mst, pos=node_dic, with_labels=False, node_size=10, ax=ax[3])
    ax[3].set_title("All connected components with >= 5 nodes")
    plt.savefig(scatter_out_path + "{}_scatter.png".format(target_cl), bbox_inches='tight')
    plt.close()

    # print(dataset, target_cl, mst.number_of_nodes(), largest_cc_g.number_of_nodes(), new_mst.number_of_nodes())
    largest_cc_points = set(list(largest_cc_g.nodes))
    all_cc_points = set(list(new_mst.nodes))
    return largest_cc_points, all_cc_points


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

    
def get_all_kde(model_type, dataset, cl, all_points, ll_dict, grid_xx, grid_yy, grid_points, img_path, **kwargs):
    """Get all kde"""
    assert model_type in ['kde', 'gmm']
    train_points = all_points[(dataset, cl)]
    train_data = np.array(train_points)
    if model_type == "kde":
        from sklearn.neighbors import KernelDensity
        from sklearn.model_selection import GridSearchCV
        bandwidths = 10 ** np.linspace(-2, 1, 20)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=5,
                            n_jobs=-1,
                            verbose=1, refit=True)
        grid.fit(train_data)
        print(grid.best_params_)
        model = grid.best_estimator_
        model.fit(train_data)
    else:
        means, covs = kwargs["means"], kwargs["covs"]
        precision_matrix = np.linalg.pinv(covs)
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=len(means), covariance_type='full', means_init=means,
                                    precisions_init=precision_matrix)
        model.fit(train_data)

    for eval_dset in DATASETS:
        for eval_cl in TB_CLASSES:
            eval_points = all_points[(eval_dset, eval_cl)]
            eval_data = np.array(eval_points)
            eval_likelihood = model.score_samples(eval_data)
            ll_dict[(dataset, cl, eval_dset, eval_cl)] = eval_likelihood

    zz = model.score_samples(grid_points)
    # print(grid_points.shape, zz.shape)
    zz = zz.reshape(grid_xx.shape[0], -1)
    fig, ax = plt.subplots(figsize=(16, 9))
    cntr_levels = [
        # -10000, -9000, -8000, -7000, -6000, -5000, -4000, -3000, -2000, -1000,
                   -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100,
                   -90, -80, -70, -60, -50, -40, -30, -20, -10, -7.5, -5, -2.5,
                   0, 5, 10, 20]
    a = np.linspace(0, 1, len(cntr_levels))
    colors = mpl.colormaps['coolwarm'](a)

    cont = ax.contourf(grid_xx, grid_yy, zz, levels=cntr_levels, colors=colors)
    fig.colorbar(cont, ax=ax, orientation='vertical', location='right')
    # plt.axis('scaled')
    ax.set_title(f"{model_type.upper()} ({dataset}-{cl})")

    contour_out_path = img_path + f"images/{model_type}/{dataset}/"
    os.makedirs(contour_out_path, exist_ok=True)
    plt.savefig(fname=contour_out_path + f"{cl}_{model_type}.png", bbox_inches='tight')
    ax.scatter(train_data[:, 0], train_data[:, 1], marker='o', c='white', alpha=0.2)
    plt.savefig(fname=contour_out_path + f"{cl}_{model_type}_with_points.png", bbox_inches='tight')
    plt.close()

    for eval_cl in TB_CLASSES:
        out_path = contour_out_path + cl + "/"
        fig, ax = plt.subplots(figsize=(16, 9))
        cont = ax.contourf(grid_xx, grid_yy, zz, levels=cntr_levels, colors=colors)  # , cmap='coolwarm')
        fig.colorbar(cont, ax=ax, orientation='vertical', location='right')
        # plt.axis('scaled')
        ax.set_title(f"{model_type.upper()} ({dataset}-{eval_cl})")

        os.makedirs(out_path, exist_ok=True)
        eval_points = all_points[(dataset, eval_cl)]
        eval_data = np.array(eval_points)
        ax.scatter(eval_data[:, 0], eval_data[:, 1], marker='o', c='black', alpha=0.2)
        plt.savefig(fname=out_path + f"{eval_cl}.png", bbox_inches='tight')
        plt.close()
    return ll_dict


def compute_grid_points(all_points):
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for dset, cl in all_points.keys():
        points = all_points[(dset, cl)]
        xmin = min(xmin, points[:, 0].min())
        xmax = max(xmax, points[:, 0].max())
        ymin = min(ymin, points[:, 1].min())
        ymax = max(ymax, points[:, 1].max())

    num_cells = 500
    x = np.linspace(xmin-1, xmax+1, num_cells+1)
    y = np.linspace(ymin-1, ymax+1, num_cells+1)
    xx, yy = np.meshgrid(x, y)
    xx_r, yy_r = xx.reshape(-1, 1), yy.reshape(-1, 1)
    coords = np.hstack((xx_r, yy_r))
    return xx, yy, coords


def preprocess(path):
    """Prepare umap data"""
    umap_preprocess_for_metrics.normalize(path=path)
    umap_preprocess_for_metrics.generate_scatter_plots(path=path)


def remove_outliers(p, norm=True):
    """Remove outliers and run umap metrics"""
    preprocess(path=p)
    datapoints_dict = {}
    largest_cc_points_dict = {}
    all_cc_points_dict = {}
    hull_points_dict = {}

    hull_areas = np.zeros(shape=(4, 12), dtype=float)
    num_lcc_points = np.zeros(shape=(4, 12), dtype=int)
    num_acc_points = np.zeros(shape=(4, 12), dtype=int)
    density = np.zeros(shape=(4, 12), dtype=int)
    for dset in DATASETS:
        for tb_cl in TB_CLASSES:
            datapoints = get_points(path=p, dataset=dset, target_cl=tb_cl, norm=norm)
            datapoints_dict[(dset, tb_cl)] = datapoints
            plot_points(path=p, dataset=dset, target_cl=tb_cl, points=datapoints)
            largest_cc_idxs, all_cc_idxs = build_and_compute_mst(path=p, dataset=dset, target_cl=tb_cl,
                                                                 points=datapoints)

            lcc_points = [datapoints[i] for i in range(len(datapoints)) if i in largest_cc_idxs]
            largest_cc_points_dict[(dset, tb_cl)] = np.array(lcc_points)

            acc_points = [datapoints[i] for i in range(len(datapoints)) if i in all_cc_idxs]
            all_cc_points_dict[(dset, tb_cl)] = np.array(acc_points)

            hull_area, hull_idxs = convex_hull(path=p, dataset=dset, target_cl=tb_cl, points=lcc_points)
            hull_points = [lcc_points[i] for i in range(len(lcc_points)) if i in hull_idxs]
            hull_points_dict[(dset, tb_cl)] = hull_points

            ha_row = DSET_CUTOFFS[dset] // 12
            ha_col = TB_CLASSES.index(tb_cl)
            hull_areas[ha_row][ha_col] = hull_area
            num_lcc_points[ha_row][ha_col] = len(largest_cc_idxs)
            num_acc_points[ha_row][ha_col] = len(all_cc_idxs)
            density[ha_row][ha_col] = len(largest_cc_idxs) / hull_area

    data_path = p + "/data/"
    os.makedirs(data_path, exist_ok=True)
    datapoints_fname = data_path + "datapoints.pkl" if not norm else data_path + "datapoints_norm.pkl"
    lcc_points_fname = data_path + "lcc_points.pkl" if not norm else data_path + "lcc_points_norm.pkl"
    acc_points_fname = data_path + "acc_points.pkl" if not norm else data_path + "acc_points_norm.pkl"
    hullpoints_fname = data_path + "hullpoints.pkl" if not norm else data_path + "hullpoints_norm.pkl"
    datapoints_fp = open(datapoints_fname, "wb")
    lcc_points_fp = open(lcc_points_fname, "wb")
    acc_points_fp = open(acc_points_fname, "wb")
    hullpoints_fp = open(hullpoints_fname, "wb")
    pickle.dump(datapoints_dict, datapoints_fp)
    pickle.dump(largest_cc_points_dict, lcc_points_fp)
    pickle.dump(all_cc_points_dict, acc_points_fp)
    pickle.dump(hull_points_dict, hullpoints_fp)
    lcc_points_fp.close()
    acc_points_fp.close()
    hullpoints_fp.close()
    datapoints_fp.close()

    hull_area_df = pd.DataFrame(hull_areas, columns=TB_CLASSES)
    if norm:
        hull_area_df.to_csv(data_path + "hull_area_norm.csv")
    else:
        hull_area_df.to_csv(data_path + "hull_area.csv")

    num_lcc_points_df = pd.DataFrame(num_lcc_points, columns=TB_CLASSES)
    if norm:
        num_lcc_points_df.to_csv(data_path + "num_lcc_points_norm.csv")
    else:
        num_lcc_points_df.to_csv(data_path + "num_lcc_points.csv")

    num_acc_points_df = pd.DataFrame(num_acc_points, columns=TB_CLASSES)
    if norm:
        num_acc_points_df.to_csv(data_path + "num_acc_points_norm.csv")
    else:
        num_acc_points_df.to_csv(data_path + "num_acc_points.csv")

    density_df = pd.DataFrame(density, columns=TB_CLASSES)
    if norm:
        density_df.to_csv(data_path + "density_norm.csv")
    else:
        density_df.to_csv(data_path + "density.csv")

    mean_density_df = density_df.mean(axis=1)
    if norm:
        mean_density_df.to_csv(data_path + "mean_density_norm.csv")
    else:
        mean_density_df.to_csv(data_path + "mean_density.csv")
    pd.options.display.float_format = '{:.2f}'.format

    print("density:\n", tabulated(density_df))
    print("mean_density:\n", mean_density_df)


def compute_likelihood_dict(p, norm, outlier_removal):
    """Calculate the likelihood dict from the datapoints"""
    assert outlier_removal in ["none", "lcc", "acc"]
    likelihood_dict, data_path = {}, p + "/data/"
    all_points_fname = data_path + "datapoints.pkl" if not norm else data_path + "datapoints_norm.pkl"
    all_points_fp = open(all_points_fname, "rb")
    all_points_dict = pickle.load(all_points_fp)
    if outlier_removal == "none":
        core_points_fname = data_path + "datapoints.pkl" if not norm else data_path + "datapoints_norm.pkl"
    elif outlier_removal == "lcc":
        core_points_fname = data_path + "lcc_points.pkl" if not norm else data_path + "lcc_points_norm.pkl"
    else:
        core_points_fname = data_path + "acc_points.pkl" if not norm else data_path + "acc_points_norm.pkl"
    core_points_fp = open(core_points_fname, "rb")
    core_points_dict = pickle.load(core_points_fp)

    xx, yy, coords = compute_grid_points(all_points=all_points_dict)
    ll_data_path = p + "ll/raw/" if outlier_removal == "none" else p + f"ll/{outlier_removal}/"
    for dset in DATASETS:
        for tb_cl in TB_CLASSES:
            likelihood_dict = get_all_kde(dataset=dset, cl=tb_cl,
                                          all_points=core_points_dict, ll_dict=likelihood_dict,
                                          grid_xx=xx, grid_yy=yy, grid_points=coords, img_path=ll_data_path)

    if norm:
        ll_fname = ll_data_path + "ll_norm.pkl"
    else:
        ll_fname = ll_data_path + "ll.pkl"
    ll_fp = open(ll_fname, "wb")
    pickle.dump(likelihood_dict, ll_fp)
    ll_fp.close()


def main(p, norm=True, outlier_removal="none"):
    """Main method"""

    assert outlier_removal in ["none", "lcc", "acc"]
    remove_outliers(p=p, norm=norm)
    compute_likelihood_dict(p=p, norm=norm, outlier_removal=outlier_removal)


def tabulated(df):
    """Tabulate a pandas dataframe"""
    from tabulate import tabulate
    return tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f")


def make_ll_tbl(save_path, dic):
    """Beautify table"""
    import statistics
    assert len(list(dic.keys())) == 2304
    
    arr = np.zeros(shape=(48, 48), dtype=float)
    for key in dic.keys():
        src_d, src_cl, trgt_d, trgt_cl = key
        row, col = DSET_CUTOFFS[src_d] + TB_CLASSES.index(src_cl), DSET_CUTOFFS[trgt_d] + TB_CLASSES.index(trgt_cl)
        arr[row][col] = statistics.fmean(dic[key])
    np.savetxt(save_path+"ll.csv", arr, delimiter=',')

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
    
    arr = np.zeros(shape=(4, 12), dtype=float)
    for i, cl in enumerate(TB_CLASSES):
        lls = dic[("tb_train", cl, "in12_train", cl)]
        arr[0][i] = statistics.fmean(lls)

        lls = dic[("tb_train", cl, "in12_test", cl)]
        arr[1][i] = statistics.fmean(lls)

        lls = dic[("in12_train", cl, "tb_train", cl)]
        arr[2][i] = statistics.fmean(lls)

        lls = dic[("in12_test", cl, "tb_train", cl)]
        arr[2][i] = statistics.fmean(lls)
    
    arr_df = pd.DataFrame(arr, columns=TB_CLASSES)
    arr_df.to_csv(save_path + "ll_domain.csv")
    pd.options.display.float_format = '{:.2f}'.format
    # print(tabulated(arr_df))
    
    arr_df_mean = arr_df.mean(axis=1)
    # print("LL_domain:\n", arr_df_mean)
    arr_df_mean = np.array([arr_df_mean.mean(axis=0)])
    # arr_df_mean.to_csv(save_path + "ll_domain_mean.csv")
    np.savetxt(save_path + "ll_domain_mean.csv", arr_df_mean, delimiter=',')
    # print("ll_domain mean:\n", arr_df_mean)
    # np.savetxt(save_path + "ll_domain.csv", arr, delimiter=',')


def print_umap_overlap(arr, model_name, key):
    print("-".join([model_name, key]).center(86))
    print(tabulate(arr, headers=['Class', 'Self LL', 'Avg Cross LL', 'Diff-1', 'Max Cross LL', 'Diff-2'],
                   tablefmt="psql", floatfmt=".1f"))


def print_overlap_table(data_dict, key):
    avg_diff, max_diff = [], []
    for model_name, data_arr in data_dict.items():
        # print_umap_overlap(arr=tb_arr, model_name=model_name, key='Toybox')

        avg_arr, max_arr = [model_name], [model_name]
        avg_sum, max_sum = 0.0, 0.0
        for i, cl in enumerate(TB_CLASSES):
            avg_arr.append(data_arr[i][3])
            avg_sum += data_arr[i][3]
            max_arr.append(data_arr[i][5])
            max_sum += data_arr[i][5]

        avg_arr.append(round(avg_sum / 12, 3))
        max_arr.append(round(max_sum / 12, 3))

        avg_diff.append(avg_arr)
        max_diff.append(max_arr)

    print("-".join([key, 'Avg Diff']).center(150))
    print(tabulate(avg_diff, headers=['Model'] + TB_CLASSES + ["Average"], tablefmt='psql', floatfmt='.2f'))
    print("-".join([key, 'Max Diff']).center(150))
    print(tabulate(max_diff, headers=['Model'] + TB_CLASSES + ["Average"], tablefmt='psql', floatfmt='.2f'))


def create_correlation_table(pre_accs, pre_avg, pre_max, post_accs, post_avg, post_max, key):
    corr_table = []
    corr, p = pearsonr(pre_accs, pre_avg)
    corr_table.append(['Avg Diff Pre', 'N', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    corr, p = pearsonr(post_accs, post_avg)
    corr_table.append(['Avg Diff', 'N', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    corr, p = pearsonr(pre_accs, pre_max)
    corr_table.append(['Max Diff Pre', 'N', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    corr, p = pearsonr(post_accs, post_max)
    corr_table.append(['Max Diff', 'N', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    corr, p = pearsonr(pre_accs, list(map(lambda x: math.log(x, 10), [0.01 if i <= 0 else i for i in pre_avg])))
    corr_table.append(['Avg Diff Pre', 'Y', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    corr, p = pearsonr(post_accs, list(map(lambda x: math.log(x, 10), [0.01 if i <= 0 else i for i in post_avg])))
    corr_table.append(['Avg Diff', 'Y', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    corr, p = pearsonr(pre_accs, list(map(lambda x: math.log(x, 10), [0.01 if i <= 0 else i for i in pre_max])))
    corr_table.append(['Max Diff Pre', 'Y', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    corr, p = pearsonr(post_accs, list(map(lambda x: math.log(x, 10), [0.01 if i <= 0 else i for i in post_max])))
    corr_table.append(['Max Diff', 'Y', round(corr, 3), round(p, 3) if round(p, 3) > 0.001 else "<0.001"])

    print(key.center(40))
    print(tabulate(corr_table, headers=['Metric', 'Log', 'r', 'p'], tablefmt='psql'))


def calc_overlap_corr_by_model(accs, data_dict, key):
    avg_arr, max_arr = {}, {}
    for model_name, data_arr in data_dict.items():
        avg_sum, max_sum = 0.0, 0.0
        for i, cl in enumerate(TB_CLASSES):
            avg_sum += data_arr[i][3]
            max_sum += data_arr[i][5]

        avg_arr[model_name] = round(avg_sum / 12, 2)
        max_arr[model_name] = round(max_sum / 12, 2)

    pre_accs, pre_avg, pre_max, post_accs, post_avg, post_max = [], [], [], [], [], []
    for model_name in data_dict.keys():
        if not model_name.endswith("_pre"):
            pre_accs.append(accs[model_name])
            post_accs.append(accs[model_name])
            pre_avg.append(avg_arr[model_name+"_pre"])
            pre_max.append(max_arr[model_name+"_pre"])
            post_avg.append(avg_arr[model_name])
            post_max.append(max_arr[model_name])

    create_correlation_table(pre_accs=pre_accs, pre_avg=pre_avg, pre_max=pre_max, post_accs=post_accs,
                             post_avg=post_avg, post_max=post_max, key=key)


def calc_overlap_corr_by_model_by_class(accs, data_dict, key):
    avg_arr, max_arr = defaultdict(list), defaultdict(list)
    for model_name, data_arr in data_dict.items():
        for i, cl in enumerate(TB_CLASSES):
            avg_arr[model_name].append(round(data_arr[i][3], 2))
            max_arr[model_name].append(round(data_arr[i][5], 2))

    pre_accs, pre_avg, pre_max, post_accs, post_avg, post_max = [], [], [], [], [], []
    for model_name in data_dict.keys():
        if not model_name.endswith("_pre"):
            pre_accs += accs[model_name]
            post_accs += accs[model_name]
            pre_avg += avg_arr[model_name+"_pre"]
            pre_max += max_arr[model_name+"_pre"]
            post_avg += avg_arr[model_name]
            post_max += max_arr[model_name]
    create_correlation_table(pre_accs=pre_accs, pre_avg=pre_avg, pre_max=pre_max, post_accs=post_accs,
                             post_avg=post_avg, post_max=post_max, key=key)


def get_scatter_plots_by_model(accs, data_dict, title, match_points=True):
    COLORS = {}
    cmap_name = 'tab10'
    idx = 0
    for model_name in accs.keys():
        if not model_name.endswith("_pre"):
            COLORS[model_name] = mpl.colormaps[cmap_name].colors[idx]
            COLORS[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
            idx += 1

    avg_avg = defaultdict(float)
    max_avg = defaultdict(float)

    for model_name, data_arr in data_dict.items():
        avg_sum, max_sum = 0.0, 0.0
        for i, cl in enumerate(TB_CLASSES):
            avg_sum += data_arr[i][3]
            max_sum += data_arr[i][5]

        avg_avg[model_name] = round(avg_sum / 12, 2)
        max_avg[model_name] = round(max_sum / 12, 2)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex='col', sharey=True)

    for model_name, acc in accs.items():
        if model_name.endswith("_pre"):
            ax[0][0].scatter(y=[acc], x=[avg_avg[model_name]], label=model_name[:-4], c=[COLORS[model_name]])
    ax[0][0].set_xlabel("Avg Diff Pre")
    ax[0][0].set_ylabel("Accuracy")
    ax[0][0].set_xscale('log')
    ax[0][0].set_ylim(0, 105)

    for model_name, acc in accs.items():
        if model_name.endswith("_pre"):
            ax[0][1].scatter(y=[acc], x=[max_avg[model_name]], label=model_name[:-4], c=[COLORS[model_name]])
    ax[0][1].set_xlabel("Max Diff Pre")
    ax[0][1].set_ylabel("Accuracy")
    ax[0][1].set_xscale('log')

    for model_name, acc in accs.items():
        if not model_name.endswith("_pre"):
            ax[1][0].scatter(y=[acc], x=[avg_avg[model_name]], label=model_name, c=[COLORS[model_name]])
    ax[1][0].set_xlabel("Avg Diff")
    ax[1][0].set_ylabel("Accuracy")
    ax[1][0].set_xscale('log')

    for model_name, acc in accs.items():
        if not model_name.endswith("_pre"):
            ax[1][1].scatter(y=[acc], x=[max_avg[model_name]], label=model_name, c=[COLORS[model_name]])
    ax[1][1].set_xlabel("Max Diff")
    ax[1][1].set_ylabel("Accuracy")
    ax[1][1].set_xscale('log')

    handles, labels = ax[1][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    if match_points:
        for model_name in data_dict.keys():
            if not model_name.endswith("_pre"):
                con = ConnectionPatch(xyA=(avg_avg[model_name + "_pre"], accs[model_name + "_pre"]),
                                      xyB=(avg_avg[model_name], accs[model_name]), coordsA="data", coordsB="data",
                                      axesA=ax[0][0], axesB=ax[1][0], arrowstyle='<->', color=COLORS[model_name], alpha=0.2)

                fig.add_artist(con)

                con = ConnectionPatch(xyA=(max_avg[model_name + "_pre"], accs[model_name + "_pre"]),
                                      xyB=(max_avg[model_name], accs[model_name]), coordsA="data", coordsB="data",
                                      axesA=ax[0][1], axesB=ax[1][1], arrowstyle='<->', color=COLORS[model_name], alpha=0.2)

                fig.add_artist(con)

    plt.suptitle(title)

    plt.show()


def get_scatter_plots_by_model_by_class(accs, data_dict, title, match_points=True):
    COLORS = {}
    cmap_name = 'tab10'
    idx = 0
    for model_name in accs.keys():
        if not model_name.endswith("_pre"):
            COLORS[model_name] = mpl.colormaps[cmap_name].colors[idx]
            COLORS[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
            idx += 1

    avg_avg = defaultdict(list)
    max_avg = defaultdict(list)

    for model_name, data_arr in data_dict.items():
        for i, cl in enumerate(TB_CLASSES):
            avg_avg[model_name].append(data_arr[i][3])
            max_avg[model_name].append(data_arr[i][5] if data_arr[i][5] > 0 else 1.0)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex='col', sharey=True)
    for model_name, acc in accs.items():
        if model_name.endswith("_pre"):
            ax[0][0].scatter(y=acc, x=avg_avg[model_name], label=model_name[:-4], c=[COLORS[model_name]])
    ax[0][0].set_xlabel("Avg Diff Pre")
    ax[0][0].set_ylabel("Accuracy")
    ax[0][0].set_xscale('log')
    ax[0][0].set_ylim(0, 105)

    for model_name, acc in accs.items():
        if model_name.endswith("_pre"):
            ax[0][1].scatter(y=acc, x=max_avg[model_name], label=model_name[:-4], c=[COLORS[model_name]])
    ax[0][1].set_xlabel("Max Diff Pre")
    ax[0][1].set_ylabel("Accuracy")
    ax[0][1].set_xscale('log')

    for model_name, acc in accs.items():
        if not model_name.endswith("_pre"):
            ax[1][0].scatter(y=acc, x=avg_avg[model_name], label=model_name, c=[COLORS[model_name]])
    ax[1][0].set_xlabel("Avg Diff")
    ax[1][0].set_ylabel("Accuracy")
    ax[1][0].set_xscale('log')

    for model_name, acc in accs.items():
        if not model_name.endswith("_pre"):
            ax[1][1].scatter(y=acc, x=max_avg[model_name], label=model_name, c=[COLORS[model_name]])
    ax[1][1].set_xlabel("Max Diff")
    ax[1][1].set_ylabel("Accuracy")
    ax[1][1].set_xscale('log')

    handles, labels = ax[1][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    if match_points:
        for model_name in data_dict.keys():
            if not model_name.endswith("_pre"):
                for i, cl in enumerate(TB_CLASSES):
                    con = ConnectionPatch(xyA=(avg_avg[model_name + "_pre"][i], accs[model_name + "_pre"][i]),
                                          xyB=(avg_avg[model_name][i], accs[model_name][i]),
                                          coordsA="data", coordsB="data",
                                          axesA=ax[0][0], axesB=ax[1][0], arrowstyle='<->', color=COLORS[model_name],
                                          alpha=0.2)

                    fig.add_artist(con)

                    con = ConnectionPatch(xyA=(max_avg[model_name + "_pre"][i], accs[model_name + "_pre"][i]),
                                          xyB=(max_avg[model_name][i], accs[model_name][i]),
                                          coordsA="data", coordsB="data",
                                          axesA=ax[0][1], axesB=ax[1][1], arrowstyle='<->', color=COLORS[model_name],
                                          alpha=0.2)
                    fig.add_artist(con)

    plt.suptitle(title)

    plt.show()


def get_scatter_plots_acc(accs, tb_dict, in12_dict, title, use_size_scale=True):
    min_acc, max_acc = 0, 100

    min_size, max_size = 0, 6
    if use_size_scale:
        size_func = lambda x: 4 ** (min_size + ((max_size - min_size) * (x - min_acc)) / (max_acc - min_acc))
        alpha = 0.2
    else:
        size_func = lambda x: 30
        alpha = 0.8
    COLORS = {}
    cmap_name = 'tab10'
    idx = 0
    for model_name in tb_dict.keys():
        if not model_name.endswith("_pre"):
            COLORS[model_name] = mpl.colormaps[cmap_name].colors[idx]
            COLORS[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
            idx += 1

    in12_avg_avg = defaultdict(list)
    in12_max_avg = defaultdict(list)
    tb_avg_avg = defaultdict(list)
    tb_max_avg = defaultdict(list)

    for model_name, data_arr in in12_dict.items():
        for i, cl in enumerate(TB_CLASSES):
            in12_avg_avg[model_name].append(in12_dict[model_name][i][3])
            in12_max_avg[model_name].append(in12_dict[model_name][i][5] if data_arr[i][5] > 0.1 else 1.0)
            tb_avg_avg[model_name].append(tb_dict[model_name][i][3])
            tb_max_avg[model_name].append(tb_dict[model_name][i][5] if data_arr[i][5] > 0.1 else 1.0)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(17, 16), sharex='col', sharey='col')
    for model_name in in12_dict.keys():
        if model_name.endswith("_pre"):
            ax[0][0].scatter(y=tb_avg_avg[model_name], x=in12_avg_avg[model_name], label=model_name[:-4],
                             c=[COLORS[model_name]], alpha=alpha, s=list(map(size_func, accs[model_name])))
    ax[0][0].set_xlabel("IN-12 Avg Diff Pre")
    ax[0][0].set_ylabel("Toybox Avg Diff Pre")
    ax[0][0].set_xscale('log')
    ax[0][0].set_yscale('log')
    xmin, xmax = ax[0][0].get_xlim()
    ymin, ymax = ax[0][0].get_ylim()
    ax[0][0].set_xlim(min(xmin, ymin), max(xmax, ymax))

    for model_name in in12_dict.keys():
        if model_name.endswith("_pre"):
            ax[0][1].scatter(y=tb_max_avg[model_name], x=in12_max_avg[model_name], label=model_name[:-4],
                             c=[COLORS[model_name]], alpha=alpha, s=list(map(size_func, accs[model_name])))
    ax[0][1].set_xlabel("IN-12 Max Diff Pre")
    ax[0][1].set_ylabel("Toybox Max Diff Pre")
    ax[0][1].set_xscale('log')
    ax[0][1].set_yscale('log')
    xmin, xmax = ax[0][1].get_xlim()
    ymin, ymax = ax[0][1].get_ylim()
    ax[0][1].set_xlim(min(xmin, ymin), max(xmax, ymax))

    for model_name in in12_dict.keys():
        if not model_name.endswith("_pre"):
            ax[1][0].scatter(y=tb_avg_avg[model_name], x=in12_avg_avg[model_name], label=model_name,
                             c=[COLORS[model_name]], alpha=alpha, s=list(map(size_func, accs[model_name])))
    ax[1][0].set_xlabel("IN-12 Avg Diff")
    ax[1][0].set_ylabel("Toybox Avg Diff")
    ax[1][0].set_xscale('log')
    ax[1][0].set_yscale('log')
    xmin, xmax = ax[1][0].get_xlim()
    ymin, ymax = ax[1][0].get_ylim()
    ax[1][0].set_xlim(min(xmin, ymin), max(xmax, ymax))

    for model_name in in12_dict.keys():
        if not model_name.endswith("_pre"):
            ax[1][1].scatter(y=tb_max_avg[model_name], x=in12_max_avg[model_name], label=model_name,
                             c=[COLORS[model_name]], alpha=alpha, s=list(map(size_func, accs[model_name])))
    ax[1][1].set_xlabel("IN-12 Max Diff")
    ax[1][1].set_ylabel("Toybox Max Diff")
    ax[1][1].set_xscale('log')
    ax[1][1].set_yscale('log')
    xmin, xmax = ax[1][1].get_xlim()
    ymin, ymax = ax[1][1].get_ylim()
    ax[1][1].set_xlim(min(xmin, ymin), max(xmax, ymax))

    handles, labels = ax[1][1].get_legend_handles_labels()

    legend = fig.legend(handles, labels, loc='center right', markerscale=0.5)
    for handle in legend.legendHandles:
        handle._sizes = [150]

    plt.suptitle(title)

    plt.show()


if __name__ == "__main__":
    file_path = "../ICLR_OUT/TB_SUP_IN12_SCRAMBLED/exp_Sep_29_2023_01_09/umap_all_data/umap_200_0.1_euclidean/"
    main(p=file_path)
    
    dic_fname = file_path + "data/ll.pkl"
    dpath = file_path + "data/"
    dic_fp = open(dic_fname, "rb")
    ll_dic = pickle.load(dic_fp)
    make_ll_tbl(save_path=dpath, dic=ll_dic)
    dic_fp.close()
    print(file_path)
    
    print("----------------------------------------------------------------------------------------------------")
