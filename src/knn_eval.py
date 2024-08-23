"""Module implementing the knn-based evaluation of a trained network"""
import argparse
import numpy as np
import os
import datetime
import csv

import torch
import torch.utils.data as torchdata
import torch.utils.tensorboard as tb

import datasets
import models_eval
import networks
import tb_in12_transforms
import utils

TEMP_DIR = "../temp/knn_eval/"
OUT_DIR = "../out/knn_eval/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_train_test_acc(model, src_train_loader, src_test_loader, writer: tb.SummaryWriter, no_save, logger,
                       num_neighbors, use_cosine):
    """Get train and test accuracy"""
    src_tr_acc = model.eval(loader=src_train_loader, num_neighbors=num_neighbors, use_cosine=use_cosine)
    src_te_acc = model.eval(loader=src_test_loader, num_neighbors=num_neighbors, use_cosine=use_cosine)
    tr_accs, te_accs = {}, {}
    for idx, n_neighbor in enumerate(num_neighbors):
        logger.info("Neighbors: {} Train acc: {:.2f}   Test acc: {:.2f}".format(n_neighbor, src_tr_acc[idx],
                                                                                src_te_acc[idx]))
        tr_accs[n_neighbor] = src_tr_acc[idx]
        te_accs[n_neighbor] = src_te_acc[idx]
    if not no_save:
        writer.add_scalars(main_tag="Accuracies",
                           tag_scalar_dict={
                               'train_acc': src_tr_acc,
                               'test_acc': src_te_acc,
                           },
                           global_step=0)
    return tr_accs, te_accs


def save_model_prediction_val(model, all_loaders, out_path: str, logger):
    os.makedirs(out_path, exist_ok=True)
    all_labels_dict = {}
    all_preds_dict = {}

    for loader_name, loader in all_loaders.items():

        # Get losses, labels and predictions for current dataloader
        all_labels_dict[loader_name], all_preds_dict[loader_name] = model.get_eval_dicts(loader=loader,
                                                                                         loader_name=loader_name)

        # Assert keys are the same in different dicts
        for key in all_labels_dict[loader_name]:
            try:
                assert key in all_preds_dict[loader_name]
            except AssertionError:
                print(f'Warning:key {key} not found in {loader_name}')
        assert len(all_labels_dict[loader_name].keys()) == len(all_preds_dict[loader_name].keys())

        # Create and write output to specified CSV file
        csv_fpath = out_path + loader_name + ".csv"
        logger.info(f"Saving prediction and loss data to {csv_fpath}")
        csv_fp = open(csv_fpath, "w")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(["Index", 'Label', 'Prediction', 'Accuracy'])
        keys = sorted(list(all_labels_dict[loader_name].keys()))
        labels_dict, preds_dict = all_labels_dict[loader_name], all_preds_dict[loader_name],
        for key in keys:
            label, pred = labels_dict[key], preds_dict[key]
            csv_writer.writerow([key, label, pred, 1 if label == pred else 0])
        csv_fp.close()


def get_parser():
    """Return parser for source pretrain experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-b", "--bsize", default=128, type=int, help="Batch size for training")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of workers for loading data")
    parser.add_argument("-f", "--final", default=False, action='store_true',
                        help="Use this flag to run experiment on train+val set")
    parser.add_argument("--dataset", choices=['toybox', 'in12'], type=str, help="Choose dataset for linear eval.")
    parser.add_argument("--instances", default=-1, type=int, help="Set number of toybox instances to train on")
    parser.add_argument("--images", default=5000, type=int, help="Set number of images per class to train on")
    parser.add_argument("--seed", default=-1, type=int, help="Seed for running experiments")
    parser.add_argument("--log", choices=["debug", "info", "warning", "error", "critical"],
                        default="info", type=str)
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--model-name", default="final_model.pt", type=str,
                        help="Use this option to specify the name of the model from which weights should be loaded")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use this flag to start from network pretrained on ILSVRC")
    parser.add_argument("--no-save", default=False, action='store_true', help="Use this flag to not save anything.")
    parser.add_argument("--save-dir", default="", type=str, help="Directory to save the model")
    parser.add_argument("--neighbors", type=int, nargs='+', help="Set number of neighbors to use for KNN")
    parser.add_argument("--use-cosine", default=False, action='store_true', help="Use this flag to use cosine "
                                                                                 "similarity as the distance metric")
    return vars(parser.parse_args())


def run_knn_eval(exp_args, ):
    """Main method"""
    exp_args['seed'] = 0 if exp_args['seed'] == -1 else exp_args['seed']
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    hypertune = not exp_args['final']
    num_instances = exp_args['instances']
    num_images_per_class = exp_args['images']
    no_save = exp_args['no_save']
    save_dir = exp_args['save_dir']
    n_neighbors = exp_args['neighbors']
    use_cosine = exp_args['use_cosine']

    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
        OUT_DIR + save_dir + "/"
    assert not os.path.isdir(tb_path), f"Directory {tb_path} already exists..."
    tb_writer = tb.SummaryWriter(log_dir=tb_path) if not no_save else None
    logger = utils.create_logger(log_level_str=exp_args['log'], log_file_name=tb_path + "log.txt", no_save=no_save)

    transform = tb_in12_transforms.get_eval_transform(dset=exp_args['dataset'])

    if exp_args['dataset'] == 'toybox':

        src_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(exp_args['seed']), train=True,
                                                transform=transform,
                                                hypertune=hypertune, num_instances=num_instances,
                                                num_images_per_class=num_images_per_class,
                                                )
        logger.debug(f"Dataset: {src_data_train}  Size: {len(src_data_train)}")
        src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=False, num_workers=n_workers)

        src_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=False, transform=transform,
                                               hypertune=hypertune)
        src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)

    else:

        src_data_train = datasets.DatasetIN12(train=True, transform=transform, hypertune=hypertune)
        logger.debug(f"Dataset: {src_data_train}  Size: {len(src_data_train)}")
        src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=False, num_workers=n_workers)

        src_data_test = datasets.DatasetIN12(train=False, transform=transform, hypertune=hypertune)
        src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)

    if not no_save:
        logger.info("Experimental details and results saved to {}".format(tb_path))

    if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
        load_file_path = exp_args['load_path'] + exp_args['model_name']
        assert os.path.isfile(load_file_path), f"Tried to load weights from {load_file_path}, but does not exist..."
        load_file = torch.load(load_file_path)
        logger.info(f"Loading model weights from {load_file_path} ({load_file['type']})")
        bb_wts = load_file['backbone']
        net = networks.ResNet18Backbone(weights=bb_wts)
    else:
        net = networks.ResNet18Backbone(pretrained=exp_args['pretrained'])

    for params in net.model.parameters():
        params.requires_grad = False

    pre_model = models_eval.ModelKNNEval(network=net, train_loader=src_loader_train, logger=logger, dim=512)
    pre_model.train()

    tr_accs, te_accs = get_train_test_acc(model=pre_model, src_train_loader=src_loader_train,
                                          src_test_loader=src_loader_test, writer=tb_writer, logger=logger,
                                          no_save=no_save, num_neighbors=n_neighbors, use_cosine=use_cosine)

    if not no_save:
        tb_writer.close()

        exp_args['train_acc'] = tr_accs
        exp_args['test_acc'] = te_accs
        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['train_transform'] = str(transform)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))
        # train_loader_name = "tb_train" if exp_args['dataset'] == "toybox" else "in12_train"
        # test_loader_name = "tb_test" if exp_args['dataset'] == "toybox" else "in12_test"
        # all_loaders = {
        #     train_loader_name: src_loader_train,
        #     test_loader_name: src_loader_test
        # }
        # save_model_prediction_val(model=pre_model, all_loaders=all_loaders, out_path=tb_path + "output/final_model/",
        #                           logger=logger)


if __name__ == "__main__":
    args = get_parser()
    run_knn_eval(exp_args=args)
