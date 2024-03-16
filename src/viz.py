"""Mod"""
import cv2
import os

UMAP_FILENAMES = ["tb_train.csv", "tb_test.csv", "in12_train.csv", "in12_test.csv"]
NORM_UMAP_FILENAMES = ["tb_train_norm.csv", "tb_test_norm.csv", "in12_train_norm.csv", "in12_test_norm.csv"]
TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck',
              'giraffe', 'horse', 'helicopter', 'mug', 'spoon', 'truck']


if __name__ == "__main__":
    p = "/home/VANDERBILT/sanyald/Documents/AIVAS/Meeting Slides/Summer 2023/Aug-04-2023/umaps/dual_sup/" \
        "umap_all_data/umap_100_0.1_euclidean/"
    path = p + "/images/all/tb_train/"
    os.makedirs(path, exist_ok=True)
    path = p + "/images/all/in12_train/"
    os.makedirs(path, exist_ok=True)
    path = p + "/images/all/"
    for dset in ['tb_train', "in12_train"]:  # , 'tb_test', 'in12_train', 'in12_test']:
        for tb_cl in TB_CLASSES:
            umap_path = p + "/images/scatter/{}/{}.png".format(dset, tb_cl)
            assert os.path.isfile(umap_path)
            mst_path = p + "/images/scatter/{}/{}_mst_nx.png".format(dset, tb_cl)
            assert os.path.isfile(mst_path)
            thres_path = p + "/images/scatter/{}/{}_mst_thres.png".format(dset, tb_cl)
            assert os.path.isfile(thres_path)
            cc_path = p + "/images/scatter/{}/{}_largest_cc.png".format(dset, tb_cl)
            assert os.path.isfile(cc_path)
            hull_path = p + "/images/scatter/{}/{}_hull.png".format(dset, tb_cl)
            assert os.path.isfile(hull_path)
            kde_path = p + "/images/scatter/{}/{}_kde.png".format(dset, tb_cl)
            assert os.path.isfile(kde_path)
            
            im1 = cv2.imread(umap_path)
            im2 = cv2.imread(mst_path)
            im3 = cv2.imread(thres_path)
            im4 = cv2.imread(cc_path)
            im5 = cv2.imread(hull_path)
            im6 = cv2.imread(kde_path)
            im_c_1 = cv2.hconcat([im1, im2, im3])
            im_c_2 = cv2.hconcat([im4, im5, im6])
            im_c = cv2.vconcat([im_c_1, im_c_2])
            print(path+"/{}/{}_all.png".format(dset, tb_cl))
            cv2.imwrite(path+"/{}/{}_all.png".format(dset, tb_cl), im_c)
            