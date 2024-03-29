"""Module with code to analyze the saved csvs"""

import os
import csv

INDEX_KEY = 'Index'
LABEL_KEY = 'Label'
PREDICTION_KEY = 'Prediction'
ACCURACY_KEY = 'Accuracy'
LOSS_KEY = 'Loss'


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
    csv_path = "../out/JAN/TB_IN12/exp_Mar_29_2024_03_25/output/final_model/tb_train.csv"
    csv_util = CSVUtil(fpath=csv_path)
    print(f"Accuracy: {csv_util.get_accuracy()}")
    print(f"Loss: {csv_util.get_loss()}")
    correct, incorrect = csv_util.get_correct_incorrect_indices()
    print(f"Correct Accuracy: {csv_util.get_accuracy(correct)}")
    print(f"Correct Loss: {csv_util.get_loss(correct)}")
    print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect)}")
    print(f"Incorrect Loss: {csv_util.get_loss(incorrect)}")

    csv_path = "../out/JAN/TB_IN12/exp_Mar_29_2024_03_25/output/final_model/tb_test.csv"
    csv_util = CSVUtil(fpath=csv_path)
    print(f"Accuracy: {csv_util.get_accuracy()}")
    print(f"Loss: {csv_util.get_loss()}")
    correct, incorrect = csv_util.get_correct_incorrect_indices()
    print(f"Correct Accuracy: {csv_util.get_accuracy(correct)}")
    print(f"Correct Loss: {csv_util.get_loss(correct)}")
    print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect)}")
    print(f"Incorrect Loss: {csv_util.get_loss(incorrect)}")

    csv_path = "../out/JAN/TB_IN12/exp_Mar_29_2024_03_25/output/final_model/in12_train.csv"
    csv_util = CSVUtil(fpath=csv_path)
    print(f"Accuracy: {csv_util.get_accuracy()}")
    print(f"Loss: {csv_util.get_loss()}")
    correct, incorrect = csv_util.get_correct_incorrect_indices()
    print(f"Correct Accuracy: {csv_util.get_accuracy(correct)}")
    print(f"Correct Loss: {csv_util.get_loss(correct)}")
    print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect)}")
    print(f"Incorrect Loss: {csv_util.get_loss(incorrect)}")

    csv_path = "../out/JAN/TB_IN12/exp_Mar_29_2024_03_25/output/final_model/in12_test.csv"
    csv_util = CSVUtil(fpath=csv_path)
    print(f"Accuracy: {csv_util.get_accuracy()}")
    print(f"Loss: {csv_util.get_loss()}")
    correct, incorrect = csv_util.get_correct_incorrect_indices()
    print(f"Correct Accuracy: {csv_util.get_accuracy(correct)}")
    print(f"Correct Loss: {csv_util.get_loss(correct)}")
    print(f"Incorrect Accuracy: {csv_util.get_accuracy(incorrect)}")
    print(f"Incorrect Loss: {csv_util.get_loss(incorrect)}")
