"""Module implementing the datasets for the MTL experiments"""
import csv
import cv2
import pickle
import numpy as np
import os

import torch.utils.data as torchdata

IN12_MEAN = (0.4980, 0.4845, 0.4541)
IN12_STD = (0.2756, 0.2738, 0.2928)
IN12_DATA_PATH = "../data/data_12/IN-12/"

TOYBOX_MEAN = (0.5199, 0.4374, 0.3499)
TOYBOX_STD = (0.1775, 0.1894, 0.1623)
TOYBOX_DATA_PATH = "../data/data_12/Toybox/"

TOYBOX_UMAP_DATA_PATH = "../data/umap_data/"

TOYBOX_CLASSES = ["airplane", "ball", "car", "cat", "cup", "duck", "giraffe", "helicopter", "horse", "mug", "spoon",
                  "truck"]

TOYBOX_VIDEOS = ("rxplus", "rxminus", "ryplus", "ryminus", "rzplus", "rzminus")


class DatasetIN12All(torchdata.Dataset):
    """
    This class implements the IN12 dataset
    """
    
    def __init__(self, train=True, transform=None, hypertune=True):
        self.train = train
        self.transform = transform
        self.root = IN12_DATA_PATH
        self.hypertune = hypertune
        
        if self.train:
            if self.hypertune:
                self.images_file = self.root + "dev.pickle"
                self.labels_file = self.root + "dev.csv"
            else:
                self.images_file = self.root + "train.pickle"
                self.labels_file = self.root + "train.csv"
        else:
            if self.hypertune:
                self.images_file = self.root + "val.pickle"
                self.labels_file = self.root + "val.csv"
            else:
                self.images_file = self.root + "test.pickle"
                self.labels_file = self.root + "test.csv"
        
        self.images = pickle.load(open(self.images_file, "rb"))
        self.labels = list(csv.DictReader(open(self.labels_file, "r")))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = np.array(cv2.imdecode(self.images[index], 3))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = int(self.labels[index]["Class ID"])
        if self.transform is not None:
            im = self.transform(im)
        return (index, index), im, label
    
    def __str__(self):
        return "IN12 Supervised All"


class DatasetIN12(torchdata.Dataset):
    """
    This class implements the IN12 dataset
    """
    
    def __init__(self, train=True, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        self.train = train
        self.transform = transform
        self.root = IN12_DATA_PATH
        self.fraction = fraction
        self.hypertune = hypertune
        self.equal_div = equal_div
        
        if self.train:
            if self.hypertune:
                self.images_file = self.root + "dev.pickle"
                self.labels_file = self.root + "dev.csv"
            else:
                self.images_file = self.root + "train.pickle"
                self.labels_file = self.root + "train.csv"
        else:
            if self.hypertune:
                self.images_file = self.root + "val.pickle"
                self.labels_file = self.root + "val.csv"
            else:
                self.images_file = self.root + "test.pickle"
                self.labels_file = self.root + "test.csv"
        
        self.images = pickle.load(open(self.images_file, "rb"))
        self.labels = list(csv.DictReader(open(self.labels_file, "r")))
        if self.train:
            if self.fraction < 1.0:
                len_all_images = len(self.images)
                rng = np.random.default_rng(0)
                if self.equal_div:
                    len_images_class = len_all_images // 12
                    len_train_images_class = int(self.fraction * len_images_class)
                    self.selected_indices = []
                    for i in range(12):
                        all_indices = np.arange(i * len_images_class, (i + 1) * len_images_class)
                        sel_indices = rng.choice(all_indices, len_train_images_class, replace=False)
                        self.selected_indices = self.selected_indices + list(sel_indices)
                else:
                    len_train_images = int(len_all_images * self.fraction)
                    self.selected_indices = rng.choice(len_all_images, len_train_images, replace=False)
            else:
                self.selected_indices = np.arange(len(self.images))
    
    def __len__(self):
        if self.train:
            return len(self.selected_indices)
        else:
            return len(self.images)
    
    def __getitem__(self, index):
        if self.train:
            item = self.selected_indices[index]
        else:
            item = index
        im = np.array(cv2.imdecode(self.images[item], 3))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = int(self.labels[item]["Class ID"])
        if self.transform is not None:
            im = self.transform(im)
        return (index, item), im, label
    
    def __str__(self):
        return "IN12 Supervised"


class DatasetIN12Offset(DatasetIN12):
    """
    This class loads IN-12 data with labels offset by 12, so labels are 12-23
    """
    
    def __init__(self, train=True, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        super().__init__(train=train, transform=transform, fraction=fraction, hypertune=hypertune, equal_div=equal_div)
    
    def __getitem__(self, index):
        if self.train:
            item = self.selected_indices[index]
        else:
            item = index
        im = np.array(cv2.imdecode(self.images[item], 3))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = int(self.labels[item]["Class ID"])
        if self.transform is not None:
            im = self.transform(im)
        return (index, item), im, label + 12
    
    def __str__(self):
        return "IN12 Offset"


class DatasetIN12SSL(DatasetIN12):
    """
    This class can be used to load IN-12 data in PyTorch for SimCLR/BYOL-like methods
    """
    
    def __init__(self, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        super().__init__(train=True, transform=transform, fraction=fraction, hypertune=hypertune, equal_div=equal_div)
    
    def __getitem__(self, index):
        actual_index = self.selected_indices[index]
        img = np.array(cv2.imdecode(self.images[actual_index], 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            imgs = [self.transform(img) for _ in range(2)]
        else:
            imgs = [img, img]
    
        return (index, actual_index), imgs
    
    def __str__(self):
        return "IN12 SSL"


class IN12SSLWithLabels(DatasetIN12):
    """
    This class can be used to load IN-12 data in PyTorch for SimCLR/BYOL-like methods
    """
    
    def __init__(self, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        super().__init__(train=True, transform=transform, fraction=fraction, hypertune=hypertune,
                         equal_div=equal_div)
    
    def __getitem__(self, index):
        actual_index = self.selected_indices[index]
        img = np.array(cv2.imdecode(self.images[actual_index], 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(self.labels[actual_index]["Class ID"])
        if self.transform is not None:
            imgs = [self.transform(img) for _ in range(2)]
        else:
            imgs = [img, img]
        
        return (index, actual_index), imgs, label
    
    def __str__(self):
        return "IN12 SSL With Labels"


class ToyboxDatasetUMAP(torchdata.Dataset):
    """
    Class for loading Toybox data for UMAP/t-SNE. Provides access to all images for specified split
    Use case is for getting activations for all images
    """
    
    def __init__(self, train=True, transform=None, hypertune=True):
        self.data_path = TOYBOX_UMAP_DATA_PATH
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.label_key = 'Class ID'
        
        if self.train:
            self.images_file = self.data_path + "toybox_data_umap_cropped_train.pickle"
            self.labels_file = self.data_path + "toybox_data_umap_cropped_train.csv"
        else:
            self.images_file = self.data_path + "toybox_data_umap_cropped_test.pickle"
            self.labels_file = self.data_path + "toybox_data_umap_cropped_test.csv"
        
        super().__init__()
        
        with open(self.images_file, "rb") as pickleFile:
            self.data = pickle.load(pickleFile)
        with open(self.labels_file, "r") as csvFile:
            self.csvFile = list(csv.DictReader(csvFile))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = cv2.imdecode(self.data[index], 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        label = int(self.csvFile[index][self.label_key])
        
        if self.transform is not None:
            imgs = self.transform(img)
        else:
            imgs = img
        return (index, index), imgs, label
    
    def __str__(self):
        return "ToyboxAll"


class ToyboxDatasetAll(torchdata.Dataset):
    """
    Class for loading Toybox data for classification. Provides access to all images for specified split
    Use case is for getting activations for all images
    """
    
    def __init__(self, train=True, transform=None, hypertune=True):
        self.data_path = TOYBOX_DATA_PATH
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.label_key = 'Class ID'
        
        if self.train:
            if self.hypertune:
                self.images_file = self.data_path + "toybox_data_interpolated_cropped_dev.pickle"
                self.labels_file = self.data_path + "toybox_data_interpolated_cropped_dev.csv"
            else:
                self.images_file = self.data_path + "toybox_data_interpolated_cropped_train.pickle"
                self.labels_file = self.data_path + "toybox_data_interpolated_cropped_train.csv"
                
        else:
            if self.hypertune:
                self.images_file = self.data_path + "toybox_data_interpolated_cropped_val.pickle"
                self.labels_file = self.data_path + "toybox_data_interpolated_cropped_val.csv"
            else:
                self.images_file = self.data_path + "toybox_data_interpolated_cropped_test.pickle"
                self.labels_file = self.data_path + "toybox_data_interpolated_cropped_test.csv"
                
        super().__init__()

        with open(self.images_file, "rb") as pickleFile:
            self.data = pickle.load(pickleFile)
        with open(self.labels_file, "r") as csvFile:
            self.csvFile = list(csv.DictReader(csvFile))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imdecode(self.data[index], 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        label = int(self.csvFile[index][self.label_key])
    
        if self.transform is not None:
            imgs = self.transform(img)
        else:
            imgs = img
        return (index, index), imgs, label

    def __str__(self):
        return "ToyboxAll"


class ToyboxDataset(torchdata.Dataset):
    """
    Class for loading Toybox data for classification. The user can specify the number of instances per class
    and the number of images per class. If number of images per class is -1, all images are selected.
    """
    
    def __init__(self, rng, train=True, transform=None, size=224, hypertune=True, num_instances=-1,
                 num_images_per_class=-1, views=TOYBOX_VIDEOS):
        
        self.data_path = TOYBOX_DATA_PATH
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.size = size
        self.rng = rng
        self.num_instances = num_instances
        self.num_images_per_class = num_images_per_class
        self.views = []
        for view in views:
            assert view in TOYBOX_VIDEOS
            self.views.append(view)
        try:
            assert os.path.isdir(self.data_path)
        except AssertionError:
            raise AssertionError("Data directory not found:", self.data_path)
        self.label_key = 'Class ID'
        if self.hypertune:
            self.trainImagesFile = self.data_path + "toybox_data_interpolated_cropped_dev.pickle"
            self.trainLabelsFile = self.data_path + "toybox_data_interpolated_cropped_dev.csv"
            self.testImagesFile = self.data_path + "toybox_data_interpolated_cropped_val.pickle"
            self.testLabelsFile = self.data_path + "toybox_data_interpolated_cropped_val.csv"
        else:
            self.trainImagesFile = self.data_path + "toybox_data_interpolated_cropped_train.pickle"
            self.trainLabelsFile = self.data_path + "toybox_data_interpolated_cropped_train.csv"
            self.testImagesFile = self.data_path + "toybox_data_interpolated_cropped_test.pickle"
            self.testLabelsFile = self.data_path + "toybox_data_interpolated_cropped_test.csv"
        
        super().__init__()
        
        if self.train:
            self.indicesSelected = []
            with open(self.trainImagesFile, "rb") as pickleFile:
                self.train_data = pickle.load(pickleFile)
            with open(self.trainLabelsFile, "r") as csvFile:
                self.train_csvFile = list(csv.DictReader(csvFile))
            self.set_train_indices()
            self.verify_train_indices()
        else:
            with open(self.testImagesFile, "rb") as pickleFile:
                self.test_data = pickle.load(pickleFile)
            with open(self.testLabelsFile, "r") as csvFile:
                self.test_csvFile = list(csv.DictReader(csvFile))
    
    def __len__(self):
        if self.train:
            return len(self.indicesSelected)
        else:
            return len(self.test_data)
    
    def __getitem__(self, index):
        if self.train:
            actual_index = self.indicesSelected[index]
            img = cv2.imdecode(self.train_data[actual_index], 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            label = int(self.train_csvFile[actual_index][self.label_key])
        else:
            actual_index = index
            img = cv2.imdecode(self.test_data[index], 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            label = int(self.test_csvFile[index][self.label_key])
        
        if self.transform is not None:
            imgs = self.transform(img)
        else:
            imgs = img
        return (index, actual_index), imgs, label
    
    def __str__(self):
        return "Toybox"
    
    def verify_train_indices(self):
        """
        This method verifies that the indices chosen for training has the same number of instances
        per class as specified in self.num_instances.
        """
        unique_objs = {}
        for idx_selected in self.indicesSelected:
            cl = self.train_csvFile[idx_selected]['Class']
            if cl not in unique_objs.keys():
                unique_objs[cl] = []
            obj = int(self.train_csvFile[idx_selected]['Object'])
            if obj not in unique_objs[cl]:
                unique_objs[cl].append(obj)
            view = self.train_csvFile[idx_selected]['Transformation']
            assert view in self.views
        for cl in TOYBOX_CLASSES:
            assert len(unique_objs[cl]) == self.num_instances
    
    def set_train_indices(self):
        """
        This method sets the training indices based on the settings provided in init().
        """
        obj_dict = {}
        obj_id_dict = {}
        for row in self.train_csvFile:
            cl = row['Class']
            if cl not in obj_dict.keys():
                obj_dict[cl] = []
            obj = int(row['Object'])
            if obj not in obj_dict[cl]:
                obj_dict[cl].append(obj)
                obj_start_id = int(row['Obj Start'])
                obj_end_id = int(row['Obj End'])
                obj_id_dict[(cl, obj)] = (obj_start_id, obj_end_id)
        
        if self.num_instances < 0:
            self.num_instances = len(obj_dict['airplane'])
        
        assert self.num_instances <= len(obj_dict['airplane']), "Number of instances must be less than number " \
                                                                "of objects in CSV: {}".format(len(obj_dict['ball']))
        
        if self.num_images_per_class < 0:
            num_images_per_instance = [-1 for _ in range(self.num_instances)]
        else:
            num_images_per_instance = [int(self.num_images_per_class / self.num_instances) for _ in
                                       range(self.num_instances)]
            remaining = max(0, self.num_images_per_class - num_images_per_instance[0] * self.num_instances)
            idx_instance = 0
            while remaining > 0:
                num_images_per_instance[idx_instance] += 1
                idx_instance = (idx_instance + 1) % self.num_instances
                remaining -= 1
        
        for cl in obj_dict.keys():
            obj_list = obj_dict[cl]
            selected_objs = self.rng.choice(obj_list, self.num_instances, replace=False)
            assert len(selected_objs) == len(set(selected_objs))
            for idx_obj, obj in enumerate(selected_objs):
                start_row = obj_id_dict[(cl, obj)][0]
                end_row = obj_id_dict[(cl, obj)][1]
                all_possible_rows = [obj_row for obj_row in range(start_row, end_row + 1)]
                
                rows_with_specified_views = []
                for obj_row in all_possible_rows:
                    view_row = self.train_csvFile[obj_row]['Transformation']
                    if view_row in self.views:
                        rows_with_specified_views.append(obj_row)
                num_images_obj = len(rows_with_specified_views)
                
                num_required_images = num_images_per_instance[idx_obj]
                if num_required_images < 0:
                    num_required_images = num_images_obj
                
                selected_indices_obj = []
                while num_required_images >= num_images_obj:
                    for idx_row in rows_with_specified_views:
                        selected_indices_obj.append(idx_row)
                    num_required_images -= num_images_obj
                additional_rows = self.rng.choice(rows_with_specified_views, num_required_images,
                                                  replace=False)
                assert len(additional_rows) == len(set(additional_rows))
                
                for idx_row in additional_rows:
                    selected_indices_obj.append(idx_row)
                for idx_row in selected_indices_obj:
                    assert start_row <= idx_row <= end_row
                    row_video = self.train_csvFile[idx_row]['Transformation']
                    assert row_video in self.views
                    self.indicesSelected.append(idx_row)
    