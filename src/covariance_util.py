"""Code"""
import os
import csv
import collections
from scipy import linalg
import sklearn.decomposition as decom
import numpy as np
import bisect

TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'horse', 'helicopter', 'mug', 'spoon',
              'truck']
ACT_FNAMES = ['in12_train', 'in12_test', 'toybox_train', 'toybox_test']
SRC_FILES = {
    'toybox_train': "../data/data_12/Toybox/toybox_data_interpolated_cropped_dev.csv",
    'toybox_test': "../data/data_12/Toybox/toybox_data_interpolated_cropped_test.csv",
    'in12_train': "../data/data_12/IN-12/dev.csv",
    'in12_test': "../data/data_12/IN-12/test.csv"
}


def get_activations(path, dset, cl):
    assert cl in TB_CLASSES
    assert dset in ACT_FNAMES

    act_fpath = path + dset + "_activations.npy"
    idx_fpath = path + dset + "_indices.npy"
    assert os.path.isfile(act_fpath) and os.path.isfile(idx_fpath)

    all_activations = np.load(act_fpath)
    all_idxs = np.load(idx_fpath)
    src_data_fpath = SRC_FILES[dset]

    src_data_fp = open(src_data_fpath, "r")
    src_data = list(csv.DictReader(src_data_fp))
    #     print(all_activations.shape, all_idxs.shape, len(src_data), all_activations.dtype)
    sz = all_activations.shape[0] // len(TB_CLASSES)

    ret_arr = np.zeros(shape=(sz, all_activations.shape[1]), dtype=all_activations.dtype)
    cntr = 0
    idx_key = 'ID' if "toybox" in dset else "Index"
    for i in range(all_activations.shape[0]):
        idx = int(all_idxs[i])
        idx2, src_cl = int(src_data[idx][idx_key]), src_data[idx]['Class']
        assert idx == idx2
        if src_cl == cl:
            ret_arr[cntr] = all_activations[i]
            cntr += 1
    assert cntr == sz
    return ret_arr


def get_activations_dicts(model_path, epochs):
    activation_dicts, mean_dicts, cov_dicts = collections.defaultdict(dict), collections.defaultdict(
        dict), collections.defaultdict(dict)

    for epoch in epochs:
        activation_dicts[epoch] = {}
        mean_dicts[epoch] = {}
        cov_dicts[epoch] = {}

        act_path = model_path + f"activations_epoch_{epoch}/"
        assert os.path.isdir(act_path)
        for dset in ['toybox_train', 'in12_train']:
            for cl in TB_CLASSES:
                activations = get_activations(path=act_path, dset=dset, cl=cl)
                act_mean = np.mean(activations, axis=0, keepdims=True)
                act_cov = np.cov(activations, rowvar=False)
                assert act_mean.shape == (1, 512) and act_cov.shape == (512, 512)
                activation_dicts[epoch][(dset, cl)] = activations
                mean_dicts[epoch][(dset, cl)] = act_mean
                cov_dicts[epoch][(dset, cl)] = act_cov

    return activation_dicts, mean_dicts, cov_dicts


def get_arr_from_dict(src_dict):
    """Convert the dictionary src_dict to a 2d ndarray for plotting"""
    ret_arr = np.ones(shape=(24, 24), dtype=float)
    for key, value in src_dict.items():
        d1, cl1, d2, cl2 = key
        row = TB_CLASSES.index(cl1)
        row += 12 if 'in12' in d1 else 0
        col = TB_CLASSES.index(cl2)
        col += 12 if 'in12' in d2 else 0
        ret_arr[row, col] = value
    return ret_arr


def normalize_2d_arr(arr: np.ndarray) -> np.ndarray:
    ret_arr = np.ones(shape=arr.shape, dtype=arr.dtype)

    min_val, max_val = np.min(arr), np.max(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ret_arr[i, j] = (arr[i][j] - min_val) / (max_val - min_val)
    return ret_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    product = np.array(sigma1.dot(sigma2))
    if product.ndim < 2:
        product = product.reshape(-1, 1)
    covmean = linalg.sqrtm(product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return round(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean), 2)


def get_frechet_arrs(mean_dicts: dict, cov_dicts: dict, epoch: int) -> (np.ndarray, np.ndarray):
    """Method to calculate the between all pairs of clusters"""
    frechet_dict = collections.defaultdict(float)
    for src_dset in ['toybox_train', 'in12_train']:
        for trgt_dset in ['toybox_train', 'in12_train']:
            for src_cl in TB_CLASSES:
                mu1, sigma1 = mean_dicts[epoch][(src_dset, src_cl)], cov_dicts[epoch][(src_dset, src_cl)]
                for trgt_cl in TB_CLASSES:
                    mu2, sigma2 = mean_dicts[epoch][(trgt_dset, trgt_cl)], cov_dicts[epoch][(trgt_dset, trgt_cl)]
                    frechet = calculate_frechet_distance(mu1=mu1, sigma1=sigma1, mu2=mu2, sigma2=sigma2)
                    frechet_dict[(src_dset, src_cl, trgt_dset, trgt_cl)] = frechet

    frechet_arr = get_arr_from_dict(src_dict=frechet_dict)
    normalized_frechet_arr = normalize_2d_arr(arr=frechet_arr)
    return frechet_arr, normalized_frechet_arr


class PCAUtil:
    """Class definition for the principal components analysis utility model."""
    def __init__(self, src_activations, n_components=None):
        self.src_activations = src_activations
        self.n_components = n_components
        self.pca = decom.PCA(n_components=n_components, svd_solver='full')
        self.pca.fit(self.src_activations)
        self.thresholds = dict()
        self.cum_exp_ratio = [0.0]
        for i in range(self.pca.n_components_):
            self.cum_exp_ratio.append(self.cum_exp_ratio[-1] + self.pca.explained_variance_ratio_[i])

    def _get_explained_variance_ratio(self, trgt_activations, k):
        cov_original = np.cov(trgt_activations, rowvar=False)
        tot_var_original = np.trace(cov_original)

        transformed_activations = self.pca.transform(trgt_activations)
        cov_transformed = np.cov(transformed_activations, rowvar=False)
        tot_var_transformed = np.trace(cov_transformed[:k][:k])
        return tot_var_transformed / tot_var_original

    def transform_by_threshold(self, trgt_activations, threshold=0.95):
        """Transform the given activations using precomputed PCA and
        return the ratio of explained covariance using the top components
        that cumulatively exceed the threshold"""
        try:
            assert 0.0 < threshold <= 1.0
        except AssertionError:
            raise ValueError(f"Threshold must be > 0.0 and <= 1.0, but got {threshold}.")
        if threshold not in self.thresholds:
            self.thresholds[threshold] = bisect.bisect_left(self.cum_exp_ratio, threshold)
        num_components_for_threshold = self.thresholds[threshold]

        return self._get_explained_variance_ratio(trgt_activations, k=num_components_for_threshold)

    def transform_by_components(self, trgt_activations, k):
        """Transform the given activations using precomputed PCA and return the fraction of
        total variation explained by the top k components"""
        try:
            assert k <= self.pca.n_components_
        except AssertionError:
            raise ValueError(f"Number of components k must be less than PCA components {self.pca.n_components_},"
                             f"but got {k}.")

        return self._get_explained_variance_ratio(trgt_activations, k)

    def get_explained_variance_ratio_list(self, trgt_activations):
        """Compute the list containing the explained variance ratio as the number of components increases"""
        cov_original = np.cov(trgt_activations, rowvar=False)
        variation_original = np.trace(cov_original)

        transformed_activations = self.pca.transform(trgt_activations)
        cov_transformed = np.cov(transformed_activations, rowvar=False)
        n_components = cov_transformed.shape[0]
        ve_arr = [0.0]
        curr_ve = 0.0
        for i in range(n_components):
            curr_ve += cov_transformed[i][i]
            ve_arr.append(curr_ve / variation_original)

        return ve_arr


def get_explained_var_arr(activation_dicts, threshold=None, n_components=None):
    """Method to calculate the explained variance from the source cluster PCA for all target clusters"""
    try:
        assert threshold is not None or n_components is not None
    except AssertionError:
        raise ValueError(f'One of threshold or n_components must be set, but got {threshold} and {n_components}')
    ve_dict = collections.defaultdict(float)
    for src_dset in ['toybox_train', 'in12_train']:
        for trgt_dset in ['toybox_train', 'in12_train']:
            for src_cl in TB_CLASSES:
                src_activations = activation_dicts[100][(src_dset, src_cl)]
                src_pca = PCAUtil(src_activations)
                for trgt_cl in TB_CLASSES:
                    trgt_activations = activation_dicts[100][(trgt_dset, trgt_cl)]
                    if threshold is not None:
                        ve_dict[(src_dset, src_cl, trgt_dset, trgt_cl)] = \
                            src_pca.transform_by_threshold(trgt_activations, threshold=threshold)
                    else:
                        ve_dict[(src_dset, src_cl, trgt_dset, trgt_cl)] = \
                            src_pca.transform_by_components(trgt_activations, k=n_components)

    covariances = get_arr_from_dict(src_dict=ve_dict)
    return covariances


def get_aucs_arr(activation_dicts, n_components=None):
    """Method to calculate the explained variance AUC from the source cluster PCA for all target clusters"""
    auc_dict = collections.defaultdict(float)
    for src_dset in ['toybox_train', 'in12_train']:
        for trgt_dset in ['toybox_train', 'in12_train']:
            for src_cl in TB_CLASSES:
                src_activations = activation_dicts[100][(src_dset, src_cl)]
                src_pca = PCAUtil(src_activations=src_activations, n_components=n_components)
                for trgt_cl in TB_CLASSES:
                    trgt_activations = activation_dicts[100][(trgt_dset, trgt_cl)]
                    ve_arr = src_pca.get_explained_variance_ratio_list(trgt_activations=trgt_activations)
                    auc = np.trapz(ve_arr)
                    n_components = src_pca.pca.n_components_
                    auc_normalized = (auc - 0.5) / (n_components - 0.5)
                    auc_dict[(src_dset, src_cl, trgt_dset, trgt_cl)] = auc_normalized

    auc_arr = get_arr_from_dict(src_dict=auc_dict)
    return auc_arr
