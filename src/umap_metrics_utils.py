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
from tabulate import tabulate, SEPARATING_LINE
from collections import defaultdict
from scipy.stats import pearsonr
import ot
from sklearn.mixture import GaussianMixture
import copy
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

import umap_preprocess_for_metrics
import covariance_util as cov_util

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


def build_and_compute_mst(path, dataset, target_cl, points, plot=False, cc_rem_threshold=5):
    scatter_out_path = path + "images/scatter/{}/".format(dataset)
    graph = nx.Graph()
    node_dic = {}
    min_x, max_x, min_y, max_y = float("inf"), float("-inf"), float("inf"), float("-inf")
    for i in range(len(points)):
        node_dic[i] = (points[i][0], points[i][1])
        min_x, max_x = min(min_x, points[i][0]), max(max_x, points[i][0])
        min_y, max_y = min(min_y, points[i][1]), max(max_y, points[i][1])

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
            graph.add_edge(i, j, weight=dist)
            graph.add_edge(j, i, weight=dist)

    min_x -= 0.05
    max_x += 0.05
    min_y -= 0.05
    max_y += 0.05

    mst = nx.minimum_spanning_tree(graph)

    if plot:
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
    edge_dist_threshold = quantiles[1] + 1.5 * (quantiles[1] - quantiles[0])

    removed_edges = []
    for edge in mst.edges():
        u, v = edge
        dist = math.sqrt((points[u][0] - points[v][0]) ** 2 + (points[u][1] - points[v][1]) ** 2)
        if dist > edge_dist_threshold:
            removed_edges.append(edge)
    # print(len(removed_edges))
    for e in removed_edges:
        mst.remove_edge(e[0], e[1])

    if plot:
        nx.draw_networkx(mst, pos=node_dic, with_labels=False, node_size=10, ax=ax[1])
        ax[1].set_title("MST of all datapoints (edges above threshold removed)")

    largest_cc = max(nx.connected_components(mst), key=len)
    largest_cc_g = mst.subgraph(largest_cc).copy()

    if plot:
        nx.draw_networkx(largest_cc_g, pos=node_dic, with_labels=False, node_size=10, ax=ax[2])
        ax[2].set_title("Largest connected component")

    new_mst = mst.copy()

    removed_cc = []
    all_cc = nx.connected_components(mst)
    all_cc_idxs = []
    num_valid_components = 0
    valid_component_means = []
    valid_component_covs = []
    for cc in all_cc:
        if len(cc) < cc_rem_threshold:
            removed_cc.append(cc)
        else:
            all_cc_idxs.append(copy.deepcopy(cc))
            num_valid_components += 1
            gmm_points = list(cc)
            cc_points = np.array([node_dic[gmm_points[i]] for i in range(len(cc))])
            cc_mean = np.mean(cc_points, axis=0, keepdims=True)
            cc_cov = np.cov(cc_points, rowvar=False)
            valid_component_means.append(cc_mean)
            valid_component_covs.append(cc_cov)
    # print(removed_cc)
    for cc in removed_cc:
        for node in cc:
            new_mst.remove_node(node)

    if plot:
        nx.draw_networkx(new_mst, pos=node_dic, with_labels=False, node_size=10, ax=ax[3])
        ax[3].set_title("All connected components with >= 5 nodes")
        plt.savefig(scatter_out_path + "{}_scatter.png".format(target_cl), bbox_inches='tight')
        plt.close()

    # print(dataset, target_cl, mst.number_of_nodes(), largest_cc_g.number_of_nodes(), new_mst.number_of_nodes())
    largest_cc_points = set(list(largest_cc_g.nodes))
    all_cc_points = set(list(new_mst.nodes))
    del node_dic, mst, new_mst, graph
    return largest_cc_points, all_cc_points, valid_component_means, valid_component_covs, all_cc_idxs


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


def get_likelihoods(model_type, dataset, cl, all_points, ll_dict, grid_xx, grid_yy, grid_points, img_path, **kwargs):
    """Get all kde"""
    assert model_type in ['kde', 'gmm']
    train_points = all_points[(dataset, cl)]
    train_data = np.array(train_points)
    if model_type == "kde":
        bandwidths = 10 ** np.linspace(-2, 1, 20)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=5,
                            n_jobs=1,
                            verbose=1, refit=True)
        grid.fit(train_data)
        print(grid.best_params_)
        model = grid.best_estimator_
        model.fit(train_data)
    else:
        means, covs = kwargs["means"][(dataset, cl)], kwargs["covs"][(dataset, cl)]
        precision_matrix = np.linalg.inv(covs)
        model = GaussianMixture(n_components=len(means), covariance_type='full', means_init=means,
                                precisions_init=precision_matrix)
        model.fit(train_data)

    for eval_dset in DATASETS:
        for eval_cl in TB_CLASSES:
            eval_points = all_points[(eval_dset, eval_cl)]
            eval_data = np.array(eval_points)
            eval_likelihood = model.score_samples(eval_data)
            # total_likelihood = model.score(eval_data)
            # ave_likelihood = total_likelihood / len(eval_points)
            ll_dict[(dataset, cl, eval_dset, eval_cl)] = eval_likelihood
            del eval_points, eval_data

    zz = model.score_samples(grid_points)
    # print(grid_points.shape, zz.shape)
    zz = zz.reshape(grid_xx.shape[0], -1)
    fig, ax = plt.subplots(figsize=(16, 9))
    cntr_levels = [
        -1000, -900, -800, -700, -600, -500, -400, -300, -200, -100,
        -90, -80, -70, -60, -50, -40, -30, -20, -10, -7.5, -5, -2.5,
        0, 5, 10, 20]
    a = np.linspace(0, 1, len(cntr_levels))
    colors = mpl.colormaps['viridis'](a)

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
    del zz, grid_xx, grid_yy, grid_points, colors, train_points, model


def get_min_max(all_points):
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for dset, cl in all_points.keys():
        points = all_points[(dset, cl)]
        xmin = min(xmin, points[:, 0].min())
        xmax = max(xmax, points[:, 0].max())
        ymin = min(ymin, points[:, 1].min())
        ymax = max(ymax, points[:, 1].max())

    return xmin, xmax, ymin, ymax


def compute_grid_points(all_points):
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    for dset, cl in all_points.keys():
        points = all_points[(dset, cl)]
        xmin = min(xmin, points[:, 0].min())
        xmax = max(xmax, points[:, 0].max())
        ymin = min(ymin, points[:, 1].min())
        ymax = max(ymax, points[:, 1].max())

    num_cells = 500
    x = np.linspace(xmin - 1, xmax + 1, num_cells + 1)
    y = np.linspace(ymin - 1, ymax + 1, num_cells + 1)
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
    component_means_dict = {}
    component_covs_dict = {}
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
            largest_cc_idxs, all_cc_idxs, comp_means, comp_covs, _ = \
                build_and_compute_mst(path=p, dataset=dset, target_cl=tb_cl, points=datapoints, plot=True)

            lcc_points = [datapoints[i] for i in range(len(datapoints)) if i in largest_cc_idxs]
            largest_cc_points_dict[(dset, tb_cl)] = np.array(lcc_points)

            acc_points = [datapoints[i] for i in range(len(datapoints)) if i in all_cc_idxs]
            all_cc_points_dict[(dset, tb_cl)] = np.array(acc_points)

            component_means_dict[(dset, tb_cl)] = np.concatenate(comp_means, axis=0)
            component_covs_dict[(dset, tb_cl)] = np.array(comp_covs)

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
    cc_means_fname = data_path + "cc_means.pkl" if not norm else data_path + "cc_means_norm.pkl"
    cc_covs_fname = data_path + "cc_covs.pkl" if not norm else data_path + "cc_covs_norm.pkl"
    hullpoints_fname = data_path + "hullpoints.pkl" if not norm else data_path + "hullpoints_norm.pkl"

    datapoints_fp = open(datapoints_fname, "wb")
    lcc_points_fp = open(lcc_points_fname, "wb")
    acc_points_fp = open(acc_points_fname, "wb")
    cc_means_fp = open(cc_means_fname, "wb")
    cc_covs_fp = open(cc_covs_fname, "wb")
    hullpoints_fp = open(hullpoints_fname, "wb")

    pickle.dump(datapoints_dict, datapoints_fp)
    pickle.dump(largest_cc_points_dict, lcc_points_fp)
    pickle.dump(all_cc_points_dict, acc_points_fp)
    pickle.dump(component_means_dict, cc_means_fp)
    pickle.dump(component_covs_dict, cc_covs_fp)
    pickle.dump(hull_points_dict, hullpoints_fp)

    lcc_points_fp.close()
    acc_points_fp.close()
    cc_means_fp.close()
    cc_covs_fp.close()
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
    del hull_area_df, num_lcc_points_df, num_acc_points, density_df, mean_density_df
    del datapoints_dict, largest_cc_points_dict, all_cc_points_dict, component_means_dict, \
        component_covs_dict, hull_points_dict


def get_emds(dataset, cl, all_points, ll_dict, out_path):
    """Get all kde"""
    train_points = all_points[(dataset, cl)]
    train_data = np.array(train_points)
    n_train = len(train_data)
    w_train = np.ones((n_train,)) / n_train
    xmin, xmax, ymin, ymax = get_min_max(all_points=all_points)

    for eval_dset in DATASETS:
        for eval_cl in TB_CLASSES:
            eval_points = all_points[(eval_dset, eval_cl)]
            eval_data = np.array(eval_points)
            n_eval = len(eval_data)
            w_eval = np.ones((n_eval,)) / n_eval
            dist_matrix = ot.dist(train_data, eval_data)
            ot_map = ot.emd(w_train, w_eval, dist_matrix, numItermax=1e6)
            dist = np.sum(np.multiply(ot_map, dist_matrix))
            ll_dict[(dataset, cl, eval_dset, eval_cl)] = [dist]
            # print(f"{dataset} ({cl}) - {eval_dset} ({eval_cl}): {dist}")
            mx = ot_map.max()

            # plt.imshow(dist_matrix, interpolation='nearest')
            # plt.title(f"{dataset} ({cl}) - {eval_dset} ({eval_cl})")
            # plt.show()
            if eval_dset == dataset:
                cl_out_path = out_path + f"images/discrete_emd/{dataset}/{cl}/"
                # print(cl_out_path)
                os.makedirs(cl_out_path, exist_ok=True)
                fig, ax = plt.subplots(figsize=(16, 9))
                ax.set_title(f"Discrete EMD ({dataset} {cl}-{eval_cl})")

                eval_points = all_points[(dataset, eval_cl)]
                eval_data = np.array(eval_points)

                ax.scatter(train_data[:, 0], train_data[:, 1], marker='x', c='blue', alpha=0.4, label=cl)
                ax.scatter(eval_data[:, 0], eval_data[:, 1], marker='o', c='red', alpha=0.4, label=eval_cl)
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                for i in range(train_points.shape[0]):
                    for j in range(eval_points.shape[0]):
                        if ot_map[i, j] / mx > 1e-5:
                            ax.plot([train_data[i, 0], eval_data[j, 0]], [train_data[i, 1], eval_data[j, 1]],
                                    alpha=0.1, c=[0.5, 0.5, 0.5], ls='--')
                plt.savefig(cl_out_path + f"{eval_cl}.png", bbox_inches='tight')
                plt.close()


def get_gmm_parameters(all_points, all_means, all_covs):
    gmm_means, gmm_covs, gmm_weights = {}, {}, {}
    for dataset in DATASETS:
        for cl in TB_CLASSES:
            train_points = all_points[(dataset, cl)]
            train_data = np.array(train_points)
            means, covs = all_means[(dataset, cl)], all_covs[(dataset, cl)]
            precision_matrix = np.linalg.pinv(covs)

            model = GaussianMixture(n_components=len(means), covariance_type='full', means_init=means,
                                    precisions_init=precision_matrix)
            model.fit(train_data)
            gmm_means[(dataset, cl)] = model.means_
            gmm_covs[(dataset, cl)] = model.covariances_
            gmm_weights[(dataset, cl)] = model.weights_
            # print(f"{dataset} {cl}, {model.means_.shape}, {model.covariances_.shape}, {model.weights_.shape},"
            #       f"{type(model.means_)}, {type(model.covariances_)}, {type(model.weights_)}")
    return gmm_means, gmm_covs, gmm_weights


def GW2(weights1, mu1, sigma1, weights2, mu2, sigma2):
    # print(weights1.shape, mu1.shape, sigma1.shape, weights2.shape, mu2.shape, sigma2.shape)
    num_comp_1, num_comp_2 = mu1.shape[0], mu2.shape[0]
    # dim  = mu1.shape[1]
    # sigma1 = sigma1.reshape(num_comp_1, dim, dim)
    # sigma2 = sigma2.reshape(num_comp_2, dim, dim)
    cost_matrix = np.zeros((num_comp_1, num_comp_2))

    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(num_comp_1):
        for l in range(num_comp_2):
            # M[k,l]  = GaussianW2(mu0[k,:], mu1[l:], S0[k,:,:], S1[l,:,:])
            cost_matrix[k, l] = cov_util.calculate_frechet_distance(mu1[k, :], sigma1[k, :, :],
                                                                    mu2[l, :], sigma2[l, :, :])

    # Then we compute the OT distance or OT map thanks to the OT library
    ot_map = ot.emd(weights1, weights2, cost_matrix)  # discrete transport plan
    dist = np.sum(ot_map * cost_matrix)
    return cost_matrix, ot_map, dist


def get_gmm_emds(dataset, cl, means, covs, weights, ll_dict):
    train_mean, train_cov, train_weight = means[(dataset, cl)], covs[(dataset, cl)], weights[(dataset, cl)]
    # costs, ot_map, dist = GW2(train_weight, train_mean, train_cov, train_weight, train_mean, train_cov)
    # print(dataset, cl, costs, ot_map)
    # ll_dict[(dataset, cl, dataset, cl)] = dist
    for eval_dset in DATASETS:
        for eval_cl in TB_CLASSES:
            eval_mean, eval_cov, eval_weight = \
                means[(eval_dset, eval_cl)], covs[(eval_dset, eval_cl)], weights[(eval_dset, eval_cl)]
            costs, ot_map, dist = GW2(train_weight, train_mean, train_cov, eval_weight, eval_mean, eval_cov)
            ll_dict[(dataset, cl, eval_dset, eval_cl)] = [dist]
    del costs, ot_map
    return ll_dict


def compute_emd_likelihood_dict(p, norm):
    likelihood_dict, data_path = {}, p + "/data/"
    core_points_fname = data_path + "acc_points.pkl" if not norm else data_path + "acc_points_norm.pkl"
    core_points_fp = open(core_points_fname, "rb")
    core_points_dict = pickle.load(core_points_fp)
    ll_data_path = p + "ll/acc/"
    for dset in DATASETS:
        for tb_cl in TB_CLASSES:
            get_emds(dataset=dset, cl=tb_cl, all_points=core_points_dict, ll_dict=likelihood_dict,
                     out_path=ll_data_path)

    if norm:
        ll_fname = ll_data_path + "discrete_emd_ll_norm.pkl"
    else:
        ll_fname = ll_data_path + "discrete_emd_ll.pkl"
    ll_fp = open(ll_fname, "wb")
    pickle.dump(likelihood_dict, ll_fp)
    ll_fp.close()
    core_points_fp.close()


def save_pkl(path, fname, norm, data_dict):
    os.makedirs(path, exist_ok=True)
    if norm:
        fpath = f"{path}/{fname}_norm.pkl"
    else:
        fpath = f"{path}/{fname}.pkl"

    fp = open(fpath, "wb")
    pickle.dump(data_dict, fp)
    fp.close()


def compute_gmm_emd_likelihood_dict(p, norm):
    likelihood_dict, data_path = {}, p + "/data/"
    # all_points_fname = data_path + "datapoints.pkl" if not norm else data_path + "datapoints_norm.pkl"
    # all_points_fp = open(all_points_fname, "rb")
    # all_points_dict = pickle.load(all_points_fp)
    core_points_fname = data_path + "acc_points.pkl" if not norm else data_path + "acc_points_norm.pkl"
    core_points_fp = open(core_points_fname, "rb")
    core_points_dict = pickle.load(core_points_fp)
    cc_means_fname = data_path + "cc_means.pkl" if not norm else data_path + "cc_means_norm.pkl"
    cc_means_fp = open(cc_means_fname, "rb")
    cc_means_dict = pickle.load(cc_means_fp)
    cc_covs_fname = data_path + "cc_covs.pkl" if not norm else data_path + "cc_covs_norm.pkl"
    cc_covs_fp = open(cc_covs_fname, "rb")
    cc_covs_dict = pickle.load(cc_covs_fp)

    gmm_means_dict, gmm_covs_dict, gmm_weights_dict = get_gmm_parameters(all_points=core_points_dict,
                                                                         all_means=cc_means_dict,
                                                                         all_covs=cc_covs_dict)

    ll_data_path = p + "ll/acc/"
    for dset in DATASETS:
        for cl in TB_CLASSES:
            get_gmm_emds(dataset=dset, cl=cl, means=gmm_means_dict, covs=gmm_covs_dict,
                         weights=gmm_weights_dict, ll_dict=likelihood_dict)

    save_pkl(path=ll_data_path, fname="gmm_emd_ll", norm=norm, data_dict=likelihood_dict)
    save_pkl(path=f"{ll_data_path}/gmm_params/", fname="gmm_means", norm=norm, data_dict=gmm_means_dict)
    save_pkl(path=f"{ll_data_path}/gmm_params/", fname="gmm_covs", norm=norm, data_dict=gmm_covs_dict)
    save_pkl(path=f"{ll_data_path}/gmm_params/", fname="gmm_weights", norm=norm, data_dict=gmm_weights_dict)
    core_points_fp.close()
    cc_means_fp.close()
    cc_covs_fp.close()
    del likelihood_dict


def compute_gmm_likelihood_dict(p, norm):
    likelihood_dict, data_path = {}, p + "/data/"
    all_points_fname = data_path + "datapoints.pkl" if not norm else data_path + "datapoints_norm.pkl"
    all_points_fp = open(all_points_fname, "rb")
    all_points_dict = pickle.load(all_points_fp)
    core_points_fname = data_path + "acc_points.pkl" if not norm else data_path + "acc_points_norm.pkl"
    core_points_fp = open(core_points_fname, "rb")
    core_points_dict = pickle.load(core_points_fp)
    cc_means_fname = data_path + "cc_means.pkl" if not norm else data_path + "cc_means_norm.pkl"
    cc_means_fp = open(cc_means_fname, "rb")
    cc_means_dict = pickle.load(cc_means_fp)
    cc_covs_fname = data_path + "cc_covs.pkl" if not norm else data_path + "cc_covs_norm.pkl"
    cc_covs_fp = open(cc_covs_fname, "rb")
    cc_covs_dict = pickle.load(cc_covs_fp)

    xx, yy, coords = compute_grid_points(all_points=all_points_dict)
    ll_data_path = p + "ll/acc/"
    for dset in DATASETS:
        for tb_cl in TB_CLASSES:
            get_likelihoods(model_type="gmm", dataset=dset, cl=tb_cl, all_points=core_points_dict,
                            ll_dict=likelihood_dict, grid_xx=xx, grid_yy=yy, grid_points=coords, img_path=ll_data_path,
                            means=cc_means_dict, covs=cc_covs_dict)

    if norm:
        ll_fname = ll_data_path + "gmm_ll_norm.pkl"
    else:
        ll_fname = ll_data_path + "gmm_ll.pkl"
    ll_fp = open(ll_fname, "wb")
    pickle.dump(likelihood_dict, ll_fp)
    ll_fp.close()
    del likelihood_dict, xx, yy, coords
    all_points_fp.close()
    core_points_fp.close()
    cc_means_fp.close()
    cc_covs_fp.close()


def compute_kde_likelihood_dict(p, norm, outlier_removal):
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
            get_likelihoods(model_type="kde", dataset=dset, cl=tb_cl,
                            all_points=core_points_dict, ll_dict=likelihood_dict,
                            grid_xx=xx, grid_yy=yy, grid_points=coords, img_path=ll_data_path)

    if norm:
        ll_fname = ll_data_path + "kde_ll_norm.pkl"
    else:
        ll_fname = ll_data_path + "kde_ll.pkl"
    ll_fp = open(ll_fname, "wb")
    pickle.dump(likelihood_dict, ll_fp)

    all_points_fp.close()
    core_points_fp.close()
    ll_fp.close()
    del likelihood_dict, xx, yy, coords


def main(p, norm=True, outlier_removal="none"):
    """Main method"""

    assert outlier_removal in ["none", "lcc", "acc"]
    # remove_outliers(p=p, norm=norm)
    # compute_kde_likelihood_dict(p=p, norm=norm, outlier_removal=outlier_removal)
    # compute_emd_likelihood_dict(p=p, norm=norm)
    compute_gmm_emd_likelihood_dict(p=p, norm=norm)


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
    np.savetxt(save_path + "ll.csv", arr, delimiter=',')

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


def remove_negatives(accs, metrics):
    accs_pos, metrics_pos = [], []
    for i in range(len(metrics)):
        if metrics[i] > 0:
            metrics_pos.append(metrics[i])
            accs_pos.append(accs[i])
    return accs_pos, metrics_pos


def make_positive(accs, metrics):
    accs_pos, metrics_pos = [], []
    min_val = min(metrics)
    if min_val > 0:
        perturbation = 0.0
    else:
        perturbation = abs(min_val) + 1e-6
    for i in range(len(metrics)):
        metrics_pos.append(metrics[i] + perturbation)
        accs_pos.append(accs[i])
    # print(len(accs_pos), len(metrics_pos))
    return accs_pos, metrics_pos


def threshold(accs, metrics):
    thres = 1e-6
    accs_pos, metrics_pos = [], []
    for i in range(len(metrics)):
        if metrics[i] <= 0:
            metrics_pos.append(thres)
        else:
            metrics_pos.append(metrics[i])
        accs_pos.append(accs[i])
    # print(len(accs_pos), len(metrics_pos))
    return accs_pos, metrics_pos


def create_correlation_table(pre_accs, pre_avg, pre_max, post_accs, post_avg, post_max, key):
    corr_table = []

    row = ['Avg Diff Pre']
    corr, p = pearsonr(pre_accs, pre_avg)
    row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
               f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")

    for f in [remove_negatives, make_positive, threshold]:
        pre_accs_pos, pre_avg_pos = f(accs=pre_accs, metrics=pre_avg)
        corr, p = pearsonr(pre_accs_pos, list(map(lambda x: math.log(x, 10), [i for i in pre_avg_pos])))
        row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
                   f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")
    corr_table.append(row)

    row = ['Max Diff Pre']
    corr, p = pearsonr(pre_accs, pre_max)
    row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
               f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")

    for f in [remove_negatives, make_positive, threshold]:
        pre_accs_pos, pre_max_pos = f(accs=pre_accs, metrics=pre_max)
        corr, p = pearsonr(pre_accs_pos, list(map(lambda x: math.log(x, 10), [i for i in pre_max_pos])))
        row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
                   f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")
    corr_table.append(row)
    corr_table.append(SEPARATING_LINE)

    row = ['Avg Diff']
    corr, p = pearsonr(post_accs, post_avg)
    row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
               f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")

    for f in [remove_negatives, make_positive, threshold]:
        post_accs_pos, post_avg_pos = f(accs=post_accs, metrics=post_avg)
        corr, p = pearsonr(post_accs_pos, list(map(lambda x: math.log(x, 10), [i for i in post_avg_pos])))
        row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
                   f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")
    corr_table.append(row)

    row = ['Max Diff']
    corr, p = pearsonr(post_accs, post_max)
    row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
               f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")

    for f in [remove_negatives, make_positive, threshold]:
        post_accs_pos, post_max_pos = f(accs=post_accs, metrics=post_max)
        corr, p = pearsonr(post_accs_pos, list(map(lambda x: math.log(x, 10), [i for i in post_max_pos])))
        row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
                   f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")
    corr_table.append(row)

    print(key.center(40))
    print(tabulate(corr_table,
                   headers=['Metric', 'Linear', 'Log (no -ve)', 'Log (translated)', 'Log (threshold)'],
                   tablefmt='psql'))


def calc_overlap_corr_by_model(accs, data_dict, key):
    avg_arr, max_arr = {}, {}
    for model_name, data_arr in data_dict.items():
        avg_sum, max_sum = 0.0, 0.0
        for i, cl in enumerate(TB_CLASSES):
            avg_sum += data_arr[i][3]
            max_sum += data_arr[i][5]

        avg_arr[model_name] = round(avg_sum / 12, 5)
        max_arr[model_name] = round(max_sum / 12, 5)

    pre_accs, pre_avg, pre_max, post_accs, post_avg, post_max = [], [], [], [], [], []
    for model_name in data_dict.keys():
        if not model_name.endswith("_pre"):
            pre_accs.append(accs[model_name])
            post_accs.append(accs[model_name])
            pre_avg.append(avg_arr[model_name + "_pre"])
            pre_max.append(max_arr[model_name + "_pre"])
            post_avg.append(avg_arr[model_name])
            post_max.append(max_arr[model_name])

    create_correlation_table(pre_accs=pre_accs, pre_avg=pre_avg, pre_max=pre_max, post_accs=post_accs,
                             post_avg=post_avg, post_max=post_max, key=key)


def create_comparative_correlation_table(pre_accs, pre_avg, pre_max, post_accs, post_avg, post_max, key, labels, log):
    corr_table = []

    def make_corr_table_row(row_title, pre, acc_dict, metric_dict, use_log):
        row = [row_title, 'Yes' if pre else 'No']
        for metric_name in labels:
            if use_log:
                accs_positive, metric_positive = make_positive(accs=acc_dict[metric_name], metrics=metric_dict[
                    metric_name])
                corr, p = pearsonr(accs_positive, list(map(lambda x: math.log(x, 10), [i for i in metric_positive])))
                row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
                           f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")
            else:
                corr, p = pearsonr(acc_dict[metric_name], metric_dict[metric_name])
                row.append(f"{round(corr, 3)} (p {'=' if round(p, 3) >= 0.001 else '<'} "
                           f"{str(round(p, 3)) if round(p, 3) >= 0.001 else '0.001'})")

        return row

    corr_table.append(make_corr_table_row(row_title="Liberal", acc_dict=pre_accs, metric_dict=pre_avg,
                                          use_log=log, pre=True))
    corr_table.append(make_corr_table_row(row_title="Strict", acc_dict=pre_accs, metric_dict=pre_max,
                                          use_log=log, pre=True))
    corr_table.append(make_corr_table_row(row_title="Liberal", acc_dict=post_accs, metric_dict=post_avg,
                                         use_log=log, pre=False))
    corr_table.append(make_corr_table_row(row_title="Strict", acc_dict=post_accs, metric_dict=post_max,
                                          use_log=log, pre=False))

    table_header = ["Separability", "Pretrained"]
    for label in labels:
        table_header += [label]

    print("".join(["Linear " if not log else "Log ", key]).center(135))
    print(tabulate(corr_table, headers=table_header, tablefmt='psql'))


def calc_metric_correlation_comparative(accs, data_dicts, labels, key, log):
    assert len(data_dicts) == len(labels)

    avg_arr_dict, max_arr_dict = {}, {}
    for metric_idx, metric_name in enumerate(labels):
        data_dict = data_dicts[metric_idx]
        avg_arr, max_arr = defaultdict(list), defaultdict(list)
        for model_name, data_arr in data_dict.items():
            for i, cl in enumerate(TB_CLASSES):
                avg_arr[model_name].append(round(data_arr[i][3], 2))
                max_arr[model_name].append(round(data_arr[i][5], 2))
        avg_arr_dict[metric_name] = avg_arr
        max_arr_dict[metric_name] = max_arr
        del avg_arr, max_arr

    pre_accs, pre_avg, pre_max, post_accs, post_avg, post_max = defaultdict(list), defaultdict(list), \
        defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),

    for metric_idx, metric_name in enumerate(labels):
        avg_arr, max_arr = avg_arr_dict[metric_name], max_arr_dict[metric_name]
        for model_name in avg_arr.keys():
            if not model_name.endswith("_pre"):
                pre_accs[metric_name] += accs[model_name]
                post_accs[metric_name] += accs[model_name]
                pre_avg[metric_name] += avg_arr[model_name + "_pre"]
                pre_max[metric_name] += max_arr[model_name + "_pre"]
                post_avg[metric_name] += avg_arr[model_name]
                post_max[metric_name] += max_arr[model_name]
    create_comparative_correlation_table(pre_accs=pre_accs,
                                         pre_avg=pre_avg, pre_max=pre_max, post_accs=post_accs,
                                         post_avg=post_avg, post_max=post_max, key=key, labels=labels, log=log)


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
            pre_avg += avg_arr[model_name + "_pre"]
            pre_max += max_arr[model_name + "_pre"]
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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex=True, sharey=True)

    for model_name, acc in accs.items():
        if model_name.endswith("_pre"):
            ax[0][0].scatter(y=[acc], x=[avg_avg[model_name]], label=model_name[:-4], c=[COLORS[model_name]])
    ax[0][0].set_xlabel("Avg Diff Pre")
    ax[0][0].set_ylabel("Accuracy")
    ax[0][0].set_xscale('symlog')
    ax[0][0].set_ylim(0, 105)

    for model_name, acc in accs.items():
        if model_name.endswith("_pre"):
            ax[0][1].scatter(y=[acc], x=[max_avg[model_name]], label=model_name[:-4], c=[COLORS[model_name]])
    ax[0][1].set_xlabel("Max Diff Pre")
    ax[0][1].set_ylabel("Accuracy")
    ax[0][1].set_xscale('symlog')

    for model_name, acc in accs.items():
        if not model_name.endswith("_pre"):
            ax[1][0].scatter(y=[acc], x=[avg_avg[model_name]], label=model_name, c=[COLORS[model_name]])
    ax[1][0].set_xlabel("Avg Diff")
    ax[1][0].set_ylabel("Accuracy")
    ax[1][0].set_xscale('symlog')

    for model_name, acc in accs.items():
        if not model_name.endswith("_pre"):
            ax[1][1].scatter(y=[acc], x=[max_avg[model_name]], label=model_name, c=[COLORS[model_name]])
    ax[1][1].set_xlabel("Max Diff")
    ax[1][1].set_ylabel("Accuracy")
    ax[1][1].set_xscale('symlog')

    handles, labels = ax[1][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    if match_points:
        for model_name in data_dict.keys():
            if not model_name.endswith("_pre"):
                con = ConnectionPatch(xyA=(avg_avg[model_name + "_pre"], accs[model_name + "_pre"]),
                                      xyB=(avg_avg[model_name], accs[model_name]), coordsA="data", coordsB="data",
                                      axesA=ax[0][0], axesB=ax[1][0], arrowstyle='<->', color=COLORS[model_name],
                                      alpha=0.2)

                fig.add_artist(con)

                con = ConnectionPatch(xyA=(max_avg[model_name + "_pre"], accs[model_name + "_pre"]),
                                      xyB=(max_avg[model_name], accs[model_name]), coordsA="data", coordsB="data",
                                      axesA=ax[0][1], axesB=ax[1][1], arrowstyle='<->', color=COLORS[model_name],
                                      alpha=0.2)

                fig.add_artist(con)

    plt.suptitle(title)
    plt.show()


def gen_scatter_plot_acc_metric(accuracies, metrics_dict, xlabel, ylabel, axis, pre, colors,
                                rem_non_pos=False, translate=False, threshold_negatives=False, title=""):
    for modelname, acc in accuracies.items():
        if (pre and modelname.endswith("_pre")) or (not pre and not modelname.endswith("_pre")):
            if rem_non_pos:
                accs_pos, metrics = remove_negatives(acc, metrics_dict[modelname])
                axis.scatter(y=accs_pos, x=metrics, label=modelname[:-4], c=[colors[modelname]])
            elif translate:
                accs_pos, metrics = make_positive(acc, metrics_dict[modelname])
                axis.scatter(y=accs_pos, x=metrics, label=modelname[:-4], c=[colors[modelname]])
            elif threshold_negatives:
                accs_pos, metrics = threshold(acc, metrics_dict[modelname])
                axis.scatter(y=accs_pos, x=metrics, label=modelname[:-4], c=[colors[modelname]])
            else:
                axis.scatter(y=acc, x=metrics_dict[modelname], label=modelname[:-4], c=[colors[modelname]])
    axis.set_xlabel(xlabel, fontsize=24)
    axis.set_ylabel(ylabel, fontsize=24)
    axis.set_xscale('symlog')
    axis.set_ylim(0, 105)

    axis.tick_params(axis='both', which='both', labelsize=10, labelbottom=True)
    # for tick in axis.get_xticklabels():
    #     tick.set_visible(True)
    if title != "":
        axis.set_title(title, fontsize=24)


def compare_scatter_plot_negative_val_removal(accs, data_dict, title):
    colors = {}
    cmap_name = 'tab10'
    idx = 0
    for model_name in accs.keys():
        if not model_name.endswith("_pre"):
            colors[model_name] = mpl.colormaps[cmap_name].colors[idx]
            colors[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
            idx += 1

    avg_avg = defaultdict(list)
    max_avg = defaultdict(list)

    for model_name, data_arr in data_dict.items():
        for i, cl in enumerate(TB_CLASSES):
            avg_avg[model_name].append(data_arr[i][3])
            max_avg[model_name].append(data_arr[i][5])

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(24, 24), sharex=True, sharey=True)

    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff Pre", ylabel="Accuracy",
                                axis=ax[0][0], rem_non_pos=False, translate=False, threshold_negatives=False,
                                pre=True, colors=colors, title="Original")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff Pre", ylabel="Accuracy",
                                axis=ax[0][1], rem_non_pos=True, translate=False, threshold_negatives=False,
                                pre=True, colors=colors, title="Non-positives removed")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff Pre", ylabel="Accuracy",
                                axis=ax[0][2], rem_non_pos=False, translate=True, threshold_negatives=False,
                                pre=True, colors=colors, title="Translated")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff Pre", ylabel="Accuracy",
                                axis=ax[0][3], rem_non_pos=False, translate=False, threshold_negatives=True,
                                pre=True, colors=colors, title="Non-positives thresholded")

    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff Pre", ylabel="Accuracy",
                                axis=ax[1][0], rem_non_pos=False, translate=False, threshold_negatives=False,
                                pre=True, colors=colors, title="Original")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff Pre", ylabel="Accuracy",
                                axis=ax[1][1], rem_non_pos=True, translate=False, threshold_negatives=False,
                                pre=True, colors=colors, title="Non-positives removed")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff Pre", ylabel="Accuracy",
                                axis=ax[1][2], rem_non_pos=False, translate=True, threshold_negatives=False,
                                pre=True, colors=colors, title="Translated")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff Pre", ylabel="Accuracy",
                                axis=ax[1][3], rem_non_pos=False, translate=False, threshold_negatives=True,
                                pre=True, colors=colors, title="Non-positives thresholded")

    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff", ylabel="Accuracy",
                                axis=ax[2][0], rem_non_pos=False, translate=False, threshold_negatives=False,
                                pre=False, colors=colors, title="Original")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff", ylabel="Accuracy",
                                axis=ax[2][1], rem_non_pos=True, translate=False, threshold_negatives=False,
                                pre=False, colors=colors, title="Non-positives removed")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff", ylabel="Accuracy",
                                axis=ax[2][2], rem_non_pos=False, translate=True, threshold_negatives=False,
                                pre=False, colors=colors, title="Translated")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff", ylabel="Accuracy",
                                axis=ax[2][3], rem_non_pos=False, translate=False, threshold_negatives=True,
                                pre=False, colors=colors, title="Non-positives thresholded")

    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff", ylabel="Accuracy",
                                axis=ax[3][0], rem_non_pos=False, translate=False, threshold_negatives=False,
                                pre=False, colors=colors, title="Original")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff", ylabel="Accuracy",
                                axis=ax[3][1], rem_non_pos=True, translate=False, threshold_negatives=False,
                                pre=False, colors=colors, title="Non-positives removed")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff", ylabel="Accuracy",
                                axis=ax[3][2], rem_non_pos=False, translate=True, threshold_negatives=False,
                                pre=False, colors=colors, title="Translated")
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff", ylabel="Accuracy",
                                axis=ax[3][3], rem_non_pos=False, translate=False, threshold_negatives=True,
                                pre=False, colors=colors, title="Non-positives thresholded")

    handles, labels = ax[1][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels))

    fig.suptitle(title, fontsize='x-large')
    fig.tight_layout(pad=4.0, h_pad=1.5, )
    plt.show()


def get_scatter_plots_by_model_by_class(accs, data_dict, title, match_points=True, rem_non_positive=False,
                                        translate_negative=False):
    assert rem_non_positive is False or translate_negative is False, (f"Both rem_non_positive and translate_negative "
                                                                      f"cannot be True")
    colors = {}
    cmap_name = 'tab10'
    idx = 0
    for model_name in accs.keys():
        if not model_name.endswith("_pre"):
            colors[model_name] = mpl.colormaps[cmap_name].colors[idx]
            colors[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
            idx += 1

    avg_avg = defaultdict(list)
    max_avg = defaultdict(list)

    for model_name, data_arr in data_dict.items():
        for i, cl in enumerate(TB_CLASSES):
            avg_avg[model_name].append(data_arr[i][3])
            max_avg[model_name].append(data_arr[i][5])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), sharex=True, sharey=True)

    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff Pre", ylabel="Accuracy",
                                axis=ax[0][0], rem_non_pos=rem_non_positive, translate=translate_negative, pre=True,
                                colors=colors)
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff Pre", ylabel="Accuracy",
                                axis=ax[0][1], rem_non_pos=rem_non_positive, translate=translate_negative, pre=True,
                                colors=colors)
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff", ylabel="Accuracy",
                                axis=ax[1][0], rem_non_pos=rem_non_positive, translate=translate_negative, pre=False,
                                colors=colors)
    gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff", ylabel="Accuracy",
                                axis=ax[1][1], rem_non_pos=rem_non_positive, translate=translate_negative, pre=False,
                                colors=colors)

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    if match_points:
        for model_name in data_dict.keys():
            if not model_name.endswith("_pre"):
                for i, cl in enumerate(TB_CLASSES):
                    con = ConnectionPatch(xyA=(avg_avg[model_name + "_pre"][i], accs[model_name + "_pre"][i]),
                                          xyB=(avg_avg[model_name][i], accs[model_name][i]),
                                          coordsA="data", coordsB="data",
                                          axesA=ax[0][0], axesB=ax[1][0], arrowstyle='<->', color=colors[model_name],
                                          alpha=0.2)

                    fig.add_artist(con)

                    con = ConnectionPatch(xyA=(max_avg[model_name + "_pre"][i], accs[model_name + "_pre"][i]),
                                          xyB=(max_avg[model_name][i], accs[model_name][i]),
                                          coordsA="data", coordsB="data",
                                          axesA=ax[0][1], axesB=ax[1][1], arrowstyle='<->', color=colors[model_name],
                                          alpha=0.2)
                    fig.add_artist(con)

    if rem_non_positive:
        title += " (non-positives removed)"
    plt.suptitle(title)

    plt.show()


def gen_per_cl_metric_scatter_plots(accs, data_dicts, labels, title, rem_non_positive=False,
                                    translate_negative=True):
    assert rem_non_positive is False or translate_negative is False, (f"Both rem_non_positive and translate_negative "
                                                                      f"cannot be True")
    assert len(data_dicts) == len(labels)
    num_metrics = len(data_dicts)
    colors = {}
    cmap_name = 'tab10'
    idx = 0
    for model_name in accs.keys():
        if not model_name.endswith("_pre"):
            colors[model_name] = mpl.colormaps[cmap_name].colors[idx]
            colors[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
            idx += 1

    fig, ax = plt.subplots(nrows=4, ncols=num_metrics, figsize=(8 * len(labels), 18), sharex=True, sharey=True,
                           layout="constrained")
    for metric_idx in range(num_metrics):
        avg_avg = defaultdict(list)
        max_avg = defaultdict(list)
        for model_name, data_arr in data_dicts[metric_idx].items():
            for i, cl in enumerate(TB_CLASSES):
                avg_avg[model_name].append(data_arr[i][3])
                max_avg[model_name].append(data_arr[i][5])

        gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff Pre", ylabel="Accuracy",
                                    axis=ax[0][metric_idx], rem_non_pos=rem_non_positive, translate=translate_negative,
                                    pre=True, colors=colors, title=labels[metric_idx])
        gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff Pre", ylabel="Accuracy",
                                    axis=ax[1][metric_idx], rem_non_pos=rem_non_positive, translate=translate_negative,
                                    pre=True, colors=colors, title=labels[metric_idx])
        gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=avg_avg, xlabel="Avg Diff", ylabel="Accuracy",
                                    axis=ax[2][metric_idx], rem_non_pos=rem_non_positive, translate=translate_negative,
                                    pre=False, colors=colors, title=labels[metric_idx])
        gen_scatter_plot_acc_metric(accuracies=accs, metrics_dict=max_avg, xlabel="Max Diff", ylabel="Accuracy",
                                    axis=ax[3][metric_idx], rem_non_pos=rem_non_positive, translate=translate_negative,
                                    pre=False, colors=colors, title=labels[metric_idx])

    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside center right')

    if rem_non_positive:
        title += " (non-positives removed)"
    elif translate_negative:
        title += " (translated)"
    plt.suptitle(title, fontsize=24)
    # fig.tight_layout()
    plt.show()


def plot_scatter_metric_metric(axis, accs, dict_1, dict_2, xlabel, ylabel, pre, use_size_scale, colors,
                               title=""):
    min_acc, max_acc = 0, 100

    min_size, max_size = 0, 6
    if use_size_scale:
        size_func = lambda x: 4 ** (min_size + ((max_size - min_size) * (x - min_acc)) / (max_acc - min_acc))
        alpha = 0.2
    else:
        size_func = lambda x: 30
        alpha = 0.8

    all_x, all_y = [], []
    for mod_name in dict_1.keys():
        if pre:
            if mod_name.endswith("_pre"):
                axis.scatter(y=dict_1[mod_name], x=dict_2[mod_name], label=mod_name[:-4],
                             c=[colors[mod_name]], alpha=alpha, s=list(map(size_func, accs[mod_name])))
                all_x += dict_2[mod_name]
                all_y += dict_1[mod_name]
        else:
            if not mod_name.endswith("_pre"):
                axis.scatter(y=dict_1[mod_name], x=dict_2[mod_name], label=mod_name,
                             c=[colors[mod_name]], alpha=alpha, s=list(map(size_func, accs[mod_name])))
                all_x += dict_2[mod_name]
                all_y += dict_1[mod_name]
    axis.set_xlabel(xlabel, fontsize=24)
    axis.set_ylabel(ylabel, fontsize=24)
    axis.set_xscale('symlog')
    axis.set_yscale('symlog')
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    axis.set_xlim(min(xmin, ymin), max(xmax, ymax))
    axis.set_ylim(min(xmin, ymin), max(xmax, ymax))

    # create new Axes on the right and on the top of the current Axes
    divider = make_axes_locatable(axis)
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 0.75, pad=0.05, sharex=axis)
    ax_histy = divider.append_axes("right", 0.75, pad=0.05, sharey=axis)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    # create histogram with custom limits
    xymin, xymax = min(xmin, ymin), max(xmax, ymax)
    bins = np.logspace(np.log10(max(xymin, 1e-6)), np.log10(xymax), 30, endpoint=True)
    if xymin <= 0:
        bins = np.insert(bins, [0], [xymin])

    ax_histx.hist(all_x, bins=bins, alpha=0.6)
    ax_histy.hist(all_y, bins=bins, orientation='horizontal', alpha=0.6)

    if title != "":
        ax_histx.set_title(title, fontsize=24)


def get_comparative_metric_metric_scatter_plots(accs, tb_dicts, in12_dicts, labels, title, sharex, sharey,
                                                use_size_scale=True):
    colors = {}
    cmap_name = 'tab10'
    idx = 0
    assert len(tb_dicts) == len(in12_dicts) and len(in12_dicts) == len(labels)
    for model_name in tb_dicts[0].keys():
        if not model_name.endswith("_pre"):
            colors[model_name] = mpl.colormaps[cmap_name].colors[idx]
            colors[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
            idx += 1

    num_metrics = len(labels)
    fig, ax = plt.subplots(nrows=4, ncols=num_metrics, figsize=(8 * num_metrics + 1, 33), sharex=sharex, sharey=sharey,
                           layout="constrained")

    for metric_idx, (tb_dict, in12_dict, label) in enumerate(zip(tb_dicts, in12_dicts, labels)):
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

        plot_scatter_metric_metric(axis=ax[0][metric_idx], accs=accs, dict_1=tb_avg_avg, dict_2=in12_avg_avg,
                                   xlabel="IN-12 Avg Diff Pre", ylabel="Toybox Avg Diff Pre", pre=True,
                                   use_size_scale=use_size_scale, colors=colors, title=label)
        plot_scatter_metric_metric(axis=ax[1][metric_idx], accs=accs, dict_1=tb_max_avg, dict_2=in12_max_avg,
                                   xlabel="IN-12 Max Diff Pre", ylabel="Toybox Max Diff Pre", pre=True,
                                   use_size_scale=use_size_scale, colors=colors, title=label)
        plot_scatter_metric_metric(axis=ax[2][metric_idx], accs=accs, dict_1=tb_avg_avg, dict_2=in12_avg_avg,
                                   xlabel="IN-12 Avg Diff", ylabel="Toybox Avg Diff", pre=False,
                                   use_size_scale=use_size_scale, colors=colors, title=label)
        plot_scatter_metric_metric(axis=ax[3][metric_idx], accs=accs, dict_1=tb_max_avg, dict_2=in12_max_avg,
                                   xlabel="IN-12 Max Diff", ylabel="Toybox Max Diff", pre=False,
                                   use_size_scale=use_size_scale, colors=colors, title=label)

    handles, labels = ax[1][1].get_legend_handles_labels()

    legend = fig.legend(handles, labels, loc='outside center right', markerscale=0.5, fontsize='xx-large')
    for handle in legend.legendHandles:
        handle._sizes = [150]
    plt.suptitle(title, fontsize=24)
    # fig.tight_layout()

    fig.get_layout_engine().set(w_pad=0.1, h_pad=0.2)
    plt.show()


def get_metric_metric_scatter_plots(accs, tb_dict, in12_dict, title, use_size_scale=True):
    colors = {}
    cmap_name = 'tab10'
    idx = 0
    for model_name in tb_dict.keys():
        if not model_name.endswith("_pre"):
            colors[model_name] = mpl.colormaps[cmap_name].colors[idx]
            colors[model_name + "_pre"] = mpl.colormaps[cmap_name].colors[idx]
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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18.6, 17.6), sharex='col', sharey='col')

    plot_scatter_metric_metric(axis=ax[0][0], accs=accs, dict_1=tb_avg_avg, dict_2=in12_avg_avg,
                               xlabel="IN-12 Avg Diff Pre", ylabel="Toybox Avg Diff Pre", pre=True,
                               use_size_scale=use_size_scale, colors=colors)
    plot_scatter_metric_metric(axis=ax[0][1], accs=accs, dict_1=tb_max_avg, dict_2=in12_max_avg,
                               xlabel="IN-12 Max Diff Pre", ylabel="Toybox Max Diff Pre", pre=True,
                               use_size_scale=use_size_scale, colors=colors)
    plot_scatter_metric_metric(axis=ax[1][0], accs=accs, dict_1=tb_avg_avg, dict_2=in12_avg_avg,
                               xlabel="IN-12 Avg Diff", ylabel="Toybox Avg Diff", pre=False,
                               use_size_scale=use_size_scale, colors=colors)
    plot_scatter_metric_metric(axis=ax[1][1], accs=accs, dict_1=tb_max_avg, dict_2=in12_max_avg,
                               xlabel="IN-12 Max Diff", ylabel="Toybox Max Diff", pre=False,
                               use_size_scale=use_size_scale, colors=colors)

    handles, labels = ax[1][1].get_legend_handles_labels()

    legend = fig.legend(handles, labels, loc='center right', markerscale=0.5)
    for handle in legend.legendHandles:
        handle._sizes = [150]
    plt.suptitle(title)
    # fig.tight_layout()
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
