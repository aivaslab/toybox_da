"""Main methods for the JAN algorithm"""
import argparse
import os
import datetime
import time

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb

import datasets
import models_ensemble
import networks_da
import utils

TEMP_DIR = "../temp/JAN/"
OUT_DIR = "../out/JAN/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_train_test_acc(model, loaders, logger):
    """Get train and test accuracy"""
    start_time = time.time()
    accs = {}
    log_str = ""
    for loader_name, loader in loaders.items():
        acc = model.eval(loader=loader)
        accs[loader_name] = acc
        log_str += f"{loader_name} acc: {acc}   "

    log_str += f"T: {time.time() - start_time:.2f}s"
    logger.info(log_str)
    return accs


def get_parser():
    """Return parser for JAN experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-b", "--bsize", default=128, type=int, help="Batch size for training")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("-f", "--final", action="store_true", help="Use this flag to run on all training data")
    parser.add_argument("--load-paths", type=str, nargs="+", required=True,
                        help="Use this option to specify the directory from which model weights should be loaded")
    return vars(parser.parse_args())


def main():
    """Main method"""
    exp_args = get_parser()
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    hypertune = not exp_args['final']
    load_paths = exp_args['load_paths']
    print(load_paths)

    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "TB_IN12/" + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/"
    logger = utils.create_logger(log_level_str='info', log_file_name=tb_path + "log.txt", no_save=True)

    prob = 0.2
    color_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(hue=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(saturation=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=prob),
                        transforms.RandomEqualize(p=prob),
                        transforms.RandomPosterize(bits=4, p=prob),
                        transforms.RandomAutocontrast(p=prob)
                        ]

    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((256, 256)),
                                          transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0),
                                                                       interpolation=
                                                                       transforms.InterpolationMode.BICUBIC),
                                          transforms.RandomOrder(color_transforms),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD),
                                          transforms.RandomErasing(p=0.5)
                                          ])
    data_train = datasets.DatasetIN12(train=True, transform=transform_train, fraction=1.0,
                                      hypertune=hypertune)
    loader_train = torchdata.DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                        drop_last=True)

    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    data_test = datasets.DatasetIN12(train=False, transform=transform_test, fraction=1.0, hypertune=hypertune)
    loader_test = torchdata.DataLoader(data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)

    networks = []
    for load_path in load_paths:
        load_file_path = load_path + "final_model.pt"
        load_file = torch.load(load_file_path)
        assert "dropout" in load_file.keys()
        logger.info(f"Loading model weights from {load_file_path} ({load_file['type']})")
        bb_wts = load_file['backbone']
        btlnk_wts = load_file['bottleneck'] if 'bottleneck' in load_file.keys() else None
        dropout = load_file['dropout']
        cl_wts = load_file['classifier'] if (btlnk_wts is not None and 'classifier' in load_file.keys()) else None
        net = networks_da.ResNet18JAN(num_classes=12, backbone_weights=bb_wts, bottleneck_weights=btlnk_wts,
                                      classifier_weights=cl_wts, dropout=dropout, p=0.5)
        networks.append(net)

    model = models_ensemble.JANEnsembleModel(networks=networks)

    all_loaders = {
        'in12_train': loader_train,
        'in12_test': loader_test
    }

    accs = get_train_test_acc(model=model, loaders=all_loaders, logger=logger)


if __name__ == "__main__":
    main()
