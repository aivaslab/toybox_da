"""Module with code to analyze the saved csvs"""
import collections
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import plot_util

INDEX_KEY = 'Index'
LABEL_KEY = 'Label'
PREDICTION_KEY = 'Prediction'
ACCURACY_KEY = 'Accuracy'
LOSS_KEY = 'Loss'

TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck', 'giraffe', 'helicopter', 'horse', 'mug', 'spoon',
              'truck']


class CSVUtil:
    """Class to analyze the saved csvs"""
    def __init__(self, fpath):
        """Initialize the object"""
        self.fpath = fpath
        assert os.path.isfile(self.fpath)
        self._fptr = open(self.fpath, 'r')
        self._reader = csv.DictReader(self._fptr)
        self.reader_fields = self._reader.fieldnames
        self.reader_data = list(self._reader)

    def get_accuracy(self, indices=None):
        """Calculate overall accuracy"""
        n_correct = 0
        n_total = 0
        for row in self.reader_data:
            index = int(row[INDEX_KEY])
            if indices is None or index in indices:
                n_correct += 1 if int(row[ACCURACY_KEY]) == 1 else 0
                n_total += 1
        acc = 100. * n_correct / n_total

        return round(acc, 2)

    def get_confusion_matrix(self, num_classes):
        """Calculate confusion matrix"""
        conf_matrix = np.zeros((num_classes, num_classes), dtype=float)
        cntr = collections.Counter()
        for row in self.reader_data:
            label, prediction = int(row[LABEL_KEY]), int(row[PREDICTION_KEY])
            cntr[label] += 1
            conf_matrix[label][prediction] += 1

        for cl in range(num_classes):
            for cl2 in range(num_classes):
                conf_matrix[cl][cl2] /= cntr[cl]

        return cntr, conf_matrix

    def get_correct_incorrect_indices(self):
        """Get indices for which predictions are correct and incorrect"""
        correct = []
        incorrect = []
        for row in self.reader_data:
            if int(row[ACCURACY_KEY]) == 1:
                correct.append(int(row[INDEX_KEY]))
            else:
                incorrect.append(int(row[INDEX_KEY]))

        return set(correct), set(incorrect)

    def get_loss(self, indices=None):
        """Calculate overall loss"""
        total_loss = 0.0
        n_total = 0.0
        for row in self.reader_data:
            index = int(row[INDEX_KEY])
            if indices is None or index in indices:
                n_total += 1
                total_loss += float(row[LOSS_KEY])

        avg_loss = total_loss / n_total
        return round(avg_loss, 2)


if __name__ == '__main__':
    csv_path = "../out/IN12_SUP/in12_supervised_pretrained_finetune_1/output/final_model/in12_test.csv"
    csv_util = CSVUtil(fpath=csv_path)
    print(f"Accuracy: {csv_util.get_accuracy()}")
    print(f"Loss: {csv_util.get_loss()}")
    correct_idxs, incorrect_idxs = csv_util.get_correct_incorrect_indices()
    print(f"Correct Accuracy: {csv_util.get_accuracy(correct_idxs)}")
    print(f"Correct Loss: {csv_util.get_loss(correct_idxs)}")
    print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect_idxs)}")
    print(f"Incorrect Loss: {csv_util.get_loss(incorrect_idxs)}")
    data_cntr, conf_mat = csv_util.get_confusion_matrix(num_classes=12)
    print(data_cntr)
    plot_util.plot_heatmap_2(arrs=conf_mat, title="Confusion matrix", xlabels=TB_CLASSES, ylabels=TB_CLASSES)
    plt.show()


    # csv_path = "../out/JAN/TB_IN12/exp_Mar_29_2024_03_25/output/final_model/tb_test.csv"
    # csv_util = CSVUtil(fpath=csv_path)
    # print(f"Accuracy: {csv_util.get_accuracy()}")
    # print(f"Loss: {csv_util.get_loss()}")
    # correct, incorrect = csv_util.get_correct_incorrect_indices()
    # print(f"Correct Accuracy: {csv_util.get_accuracy(correct)}")
    # print(f"Correct Loss: {csv_util.get_loss(correct)}")
    # print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect)}")
    # print(f"Incorrect Loss: {csv_util.get_loss(incorrect)}")
    #
    # csv_path = "../out/JAN/TB_IN12/exp_Mar_29_2024_03_25/output/final_model/in12_train.csv"
    # csv_util = CSVUtil(fpath=csv_path)
    # print(f"Accuracy: {csv_util.get_accuracy()}")
    # print(f"Loss: {csv_util.get_loss()}")
    # correct, incorrect = csv_util.get_correct_incorrect_indices()
    # print(f"Correct Accuracy: {csv_util.get_accuracy(correct)}")
    # print(f"Correct Loss: {csv_util.get_loss(correct)}")
    # print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect)}")
    # print(f"Incorrect Loss: {csv_util.get_loss(incorrect)}")
    #
    # csv_path = "../out/JAN/TB_IN12/exp_Mar_29_2024_03_25/output/final_model/in12_test.csv"
    # csv_util = CSVUtil(fpath=csv_path)
    # print(f"Accuracy: {csv_util.get_accuracy()}")
    # print(f"Loss: {csv_util.get_loss()}")
    # correct, incorrect = csv_util.get_correct_incorrect_indices()
    # print(f"Correct Accuracy: {csv_util.get_accuracy(correct)}")
    # print(f"Correct Loss: {csv_util.get_loss(correct)}")
    # print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect)}")
    # print(f"Incorrect Loss: {csv_util.get_loss(incorrect)}")
