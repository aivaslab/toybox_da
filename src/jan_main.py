"""Main methods for the JAN algorithm"""
import argparse
import csv

import numpy as np
import os
import datetime
import time

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb

import datasets
import models_da
import networks_da
import utils

TEMP_DIR = "../temp/JAN/"
OUT_DIR = "../out/JAN/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_train_test_acc(model, loaders, writer: tb.SummaryWriter, step: int, logger, no_save):
    """Get train and test accuracy"""
    start_time = time.time()
    accs = {}
    log_str = ""
    for loader_name, loader in loaders.items():
        acc = model.eval(loader=loader)
        accs[loader_name] = acc
        log_str += f"{loader_name} acc: {acc}   "

    if not no_save:
        scalar_dict = {}
        for loader_name, acc in accs.items():
            scalar_dict[loader_name] = acc
        writer.add_scalars(main_tag="Accuracies", tag_scalar_dict=scalar_dict, global_step=step)
    log_str += f"T: {time.time() - start_time:.2f}s"
    logger.info(log_str)
    return accs


def get_parser():
    """Return parser for JAN experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="Number of epochs of training")
    parser.add_argument("-it", "--iters", default=500, type=int, help="Number of training steps per epoch")
    parser.add_argument("-b", "--bsize", default=128, type=int, help="Batch size for training")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("-f", "--final", default=False, action='store_true',
                        help="Use this flag to run experiments on train+val set")
    parser.add_argument("-c", "--combined-batch", default=False, action='store_true',
                        help="Use this flag to run experiments in the combined batch setting")
    parser.add_argument("-lr", "--lr", default=0.05, type=float, help="Learning rate for the experiment")
    parser.add_argument("-wd", "--wd", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--instances", default=-1, type=int, help="Set number of toybox instances to train on")
    parser.add_argument("--images", default=1000, type=int, help="Set number of images per class to train on")
    parser.add_argument("--seed", default=-1, type=int, help="Seed for running experiments")
    parser.add_argument("--log", choices=["debug", "info", "warning", "error", "critical"], default="info", type=str)
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use this flag to start from pretrained network")
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--dropout", "-drop", default=0, type=int, choices=[0, 1, 2, 3],
                        help="Use this flag to enable dropout")
    parser.add_argument("--amp", default=False, action='store_true', help="Use AMP training")
    parser.add_argument("--no-save", default=False, action='store_true', help="Use this flag to not save anything")
    parser.add_argument("--mode", default="train", choices=["train", "eval"],
                        help="Use this flag to specify the run mode")
    parser.add_argument("--p", "-p", default=0.5, type=float, help="Probability of Dropout")
    parser.add_argument("--save-freq", default=-1, type=int, help="Frequency of saving model")
    return vars(parser.parse_args())


def eval_model(exp_args):
    """Eval model"""
    hypertune = not exp_args['final']
    num_instances = exp_args['instances']
    num_images_per_class = exp_args['images']
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    logger = utils.create_logger(log_level_str='info', log_file_name="", no_save=True)

    # Load Toybox Train dataset and dataloader
    src_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])

    src_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True,
                                            transform=src_transform,
                                            hypertune=hypertune, num_instances=num_instances,
                                            num_images_per_class=num_images_per_class,
                                            )
    logger.info(f"Source Train dataset: {src_data_train}  Size: {len(src_data_train)}")
    src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=False, num_workers=n_workers,
                                            drop_last=False)

    # Load Toybox Test dataset and dataloader
    src_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=False, transform=src_transform,
                                           hypertune=hypertune)
    logger.info(f"Source test dataset: {src_data_test}  Size: {len(src_data_test)}")
    src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)

    # Load IN-12 train dataset and dataloader
    trgt_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])

    trgt_data_train = datasets.DatasetIN12(train=True, transform=trgt_transform, fraction=1.0, hypertune=hypertune)
    logger.info(f"Target Train dataset: {trgt_data_train}  Size: {len(trgt_data_train)}")
    trgt_loader_train = torchdata.DataLoader(trgt_data_train, batch_size=b_size, shuffle=False, num_workers=n_workers,
                                             drop_last=False)

    # Load IN-12 test dataset and dataloader
    trgt_data_test = datasets.DatasetIN12(train=False, transform=trgt_transform, fraction=1.0, hypertune=hypertune)
    logger.info(f"Target test dataset: {trgt_data_test}  Size: {len(trgt_data_test)}")
    trgt_loader_test = torchdata.DataLoader(trgt_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)

    # Assert specified file exists and is of type ResNet18JAN
    load_file_path = exp_args['load_path']
    assert os.path.isfile(load_file_path)
    load_file = torch.load(load_file_path)
    assert load_file['type'] == networks_da.ResNet18JAN.__name__
    logger.info(f"Loading model weights from {load_file_path} ({load_file['type']})")

    # Initialize JAN model with weights from specified file
    bb_wts, btlnk_wts, dropout, cl_wts = (load_file['backbone'], load_file['bottleneck'], load_file['dropout'],
                                          load_file['classifier'])
    net = networks_da.ResNet18JAN(num_classes=12, backbone_weights=bb_wts, bottleneck_weights=btlnk_wts,
                                  classifier_weights=cl_wts, dropout=dropout)

    model = models_da.JANModel(network=net, source_loader=src_loader_train, target_loader=trgt_loader_train,
                               logger=logger, combined_batch=True, use_amp=exp_args['amp'], no_save=True)

    all_loaders = {
        'tb_train': src_loader_train,
        'tb_test': src_loader_test,
        'in12_train': trgt_loader_train,
        'in12_test': trgt_loader_test
    }

    # Create directory for CSV output
    load_path_split = exp_args['load_path'].split("/")
    out_dir = "/".join(load_path_split[:-1]) + "/output/" + load_path_split[-1][:-3] + "/"
    os.makedirs(out_dir, exist_ok=True)
    all_losses_dict = {}
    all_labels_dict = {}
    all_preds_dict = {}

    for loader_name, loader in all_loaders.items():

        # Get losses, labels and predictions for current dataloader
        all_losses_dict[loader_name] = model.get_val_loss_dict(loader=loader, loader_name=loader_name)
        all_labels_dict[loader_name], all_preds_dict[loader_name] = model.get_eval_dicts(loader=loader,
                                                                                         loader_name=loader_name)

        # Assert keys are the same in different dicts
        for key in all_losses_dict[loader_name]:
            try:
                assert key in all_labels_dict[loader_name] and key in all_preds_dict[loader_name]
            except AssertionError:
                print(f'Warning:key {key} not found in {loader_name}')
        assert len(all_losses_dict[loader_name].keys()) == len(all_preds_dict[loader_name].keys())
        assert len(all_labels_dict[loader_name].keys()) == len(all_preds_dict[loader_name].keys())

        # Create and write output to specified CSV file
        csv_fpath = out_dir + loader_name + ".csv"
        logger.info(f"Saving prediction and loss data to {csv_fpath}")
        csv_fp = open(csv_fpath, "w")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(["Index", 'Label', 'Prediction', 'Accuracy', 'Loss'])
        keys = sorted(list(all_losses_dict[loader_name].keys()))
        labels_dict, preds_dict, losses_dict = all_labels_dict[loader_name], all_preds_dict[loader_name], \
            all_losses_dict[loader_name]
        for key in keys:
            label, pred, loss = labels_dict[key], preds_dict[key], losses_dict[key]
            csv_writer.writerow([key, label, pred, 1 if label == pred else 0, loss])
        csv_fp.close()


def main():
    """Main method"""
    exp_args = get_parser()
    exp_args['seed'] = None if exp_args['seed'] == -1 else exp_args['seed']
    num_epochs = exp_args['epochs']
    run_mode = exp_args['mode']
    if run_mode != 'train':
        eval_model(exp_args=exp_args)
        return
    steps = exp_args['iters']
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    hypertune = not exp_args['final']
    num_instances = exp_args['instances']
    num_images_per_class = exp_args['images']
    combined_batch = exp_args['combined_batch']
    dropout = exp_args['dropout']
    no_save = exp_args['no_save']
    amp = exp_args['amp']
    dropout_p = exp_args['p']
    save_freq = exp_args['save_freq']
    if amp:
        torch.set_float32_matmul_precision('high')
    
    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "TB_IN12/" + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/"
    tb_writer = tb.SummaryWriter(log_dir=tb_path) if not no_save else None
    logger = utils.create_logger(log_level_str=exp_args['log'], log_file_name=tb_path + "log.txt", no_save=no_save)
    
    prob = 0.2
    color_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(hue=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(saturation=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=prob),
                        transforms.RandomEqualize(p=prob),
                        transforms.RandomPosterize(bits=4, p=prob),
                        transforms.RandomAutocontrast(p=prob)
                        ]
    src_transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((256, 256)),
                                              transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0),
                                                                           interpolation=
                                                                           transforms.InterpolationMode.BICUBIC),
                                              transforms.RandomOrder(color_transforms),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD),
                                              transforms.RandomErasing(p=0.5)
                                              ])
    
    src_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(exp_args['seed']), train=True,
                                            transform=src_transform_train,
                                            hypertune=hypertune, num_instances=num_instances,
                                            num_images_per_class=num_images_per_class,
                                            )
    logger.debug(f"Source dataset: {src_data_train}  Size: {len(src_data_train)}")
    src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                            drop_last=True)
    
    src_transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])
    
    src_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=False, transform=src_transform_test,
                                           hypertune=hypertune)
    src_loader_test = torchdata.DataLoader(src_data_test, batch_size=2*b_size, shuffle=False, num_workers=n_workers)
    
    trgt_transform_train = transforms.Compose([transforms.ToPILImage(),
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
    trgt_data_train = datasets.DatasetIN12(train=True, transform=trgt_transform_train, fraction=1.0,
                                           hypertune=hypertune)
    trgt_loader_train = torchdata.DataLoader(trgt_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                             drop_last=True)
    
    trgt_transform_test = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    trgt_data_test = datasets.DatasetIN12(train=False, transform=trgt_transform_test, fraction=1.0, hypertune=hypertune)
    trgt_loader_test = torchdata.DataLoader(trgt_data_test, batch_size=2*b_size, shuffle=False, num_workers=n_workers)
    
    # logger.debug(utils.online_mean_and_sd(src_loader_train), utils.online_mean_and_sd(src_loader_test))
    # logger.debug(utils.online_mean_and_sd(trgt_loader_test))
    
    if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
        load_file_path = exp_args['load_path'] + "final_model.pt"
        load_file = torch.load(load_file_path)
        logger.info(f"Loading model weights from {load_file_path} ({load_file['type']})")
        bb_wts = load_file['backbone']
        btlnk_wts = load_file['bottleneck'] if 'bottleneck' in load_file.keys() else None
        dropout = load_file['dropout'] if 'dropout' in load_file.keys() else dropout
        cl_wts = load_file['classifier'] if (btlnk_wts is not None and 'classifier' in load_file.keys()) else None
        net = networks_da.ResNet18JAN(num_classes=12, backbone_weights=bb_wts, bottleneck_weights=btlnk_wts,
                                      classifier_weights=cl_wts, dropout=dropout, p=dropout_p)
    else:
        net = networks_da.ResNet18JAN(num_classes=12, pretrained=exp_args['pretrained'], dropout=dropout,
                                      p=dropout_p)

    model = models_da.JANModel(network=net, source_loader=src_loader_train, target_loader=trgt_loader_train,
                               logger=logger, combined_batch=combined_batch, use_amp=amp, no_save=no_save)

    bb_lr_wt = 0.1 if (exp_args['pretrained'] or
                       (exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']))) \
        else 1.0

    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=bb_lr_wt * exp_args['lr'], weight_decay=exp_args['wd'],
                                momentum=0.9, nesterov=True)
    optimizer.add_param_group({'params': net.bottleneck.parameters(), 'lr': exp_args['lr'], 'wd': exp_args['wd']})
    optimizer.add_param_group({'params': net.classifier_head.parameters(), 'lr': exp_args['lr'], 'wd': exp_args['wd']})

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1. + 10 * float(x / (num_epochs * steps))) ** (-0.75))

    all_loaders = {
        'tb_train': src_loader_train,
        'tb_test': src_loader_test,
        'in12_train': trgt_loader_train,
        'in12_test': trgt_loader_test
    }

    val_loaders = {
        'tb_test': src_loader_test,
        'in12_test': trgt_loader_test
    }

    best_tb_val_loss = float("inf")
    best_tb_val_acc = 0.0
    best_tb_val_loss_ep = -1
    best_tb_val_acc_ep = -1

    accs = get_train_test_acc(model=model, loaders=all_loaders, writer=tb_writer, step=0, logger=logger,
                              no_save=no_save)

    if accs['tb_test'] >= best_tb_val_acc:
        best_tb_val_acc = accs['tb_test']
        best_tb_val_acc_ep = 0
        if not no_save:
            logger.info("Experimental details and results saved to {}".format(tb_path))
            net.save_model(fpath=tb_path+"best_tb_val_acc_model.pt")

    val_losses = model.calc_val_loss(ep=0, steps=steps, writer=tb_writer, loaders=[src_loader_test, trgt_loader_test],
                                     loader_names=['tb_test', 'in12_test'])

    if val_losses[0] <= best_tb_val_loss:
        best_tb_val_loss = val_losses[0]
        best_tb_val_loss_ep = 0
        if not no_save:
            net.save_model(fpath=tb_path+"best_tb_val_loss_model.pt")

    if not no_save and save_freq > 0:
        net.save_model(fpath=tb_path+f"model_epoch_0.pt")

    for ep in range(1, num_epochs + 1):
        model.train(optimizer=optimizer, scheduler=lr_scheduler, steps=steps,
                    ep=ep, ep_total=num_epochs, writer=tb_writer)
        val_losses = model.calc_val_loss(ep=ep, steps=steps, writer=tb_writer,
                                         loaders=[src_loader_test, trgt_loader_test],
                                         loader_names=['tb_test', 'in12_test'])
        if val_losses[0] <= best_tb_val_loss:
            best_tb_val_loss = val_losses[0]
            best_tb_val_loss_ep = ep
            if not no_save:
                net.save_model(fpath=tb_path + "best_tb_val_loss_model.pt")

        if ep % 10 == 0:
            accs = get_train_test_acc(model=model, loaders=all_loaders, writer=tb_writer, step=ep * steps,
                                      logger=logger, no_save=no_save)
        else:
            accs = get_train_test_acc(model=model, loaders=val_loaders, writer=tb_writer, step=ep*steps,
                                      logger=logger, no_save=no_save)
        if accs['tb_test'] >= best_tb_val_acc:
            best_tb_val_acc = accs['tb_test']
            best_tb_val_acc_ep = ep
            if not no_save:
                net.save_model(fpath=tb_path + "best_tb_val_acc_model.pt")

        if not no_save and save_freq > 0 and ep % save_freq == 0:
            net.save_model(fpath=tb_path + f"model_epoch_{ep}.pt")

    accs = get_train_test_acc(model=model, loaders=all_loaders, writer=tb_writer, step=num_epochs * steps,
                              logger=logger, no_save=no_save)
    logger.info(f"Best val loss on tb_test at epoch {best_tb_val_loss_ep} with val "
                f"{best_tb_val_loss:.2f}")

    logger.info(f"Best val acc on tb_test at epoch {best_tb_val_acc_ep} with acc "
                f"{best_tb_val_acc}")
    if not no_save:
        tb_writer.close()
        net.save_model(fpath=tb_path + "final_model.pt")
        net.save_model(fpath=tb_path + f"model_epoch_{num_epochs}.pt")
    
        exp_args['tb_train'] = accs['tb_train']
        exp_args['tb_test'] = accs['tb_test']
        exp_args['in_train'] = accs['in12_train']
        exp_args['in12_test'] = accs['in12_test']
        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['train_transform'] = str(src_transform_train)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))

        logger.info("-------------------------------------------------------------------------")
        exp_args['load_path'] = tb_path + "best_tb_val_loss_model.pt"
        logger.info("Evaluating model with best val loss on Toybox")
        eval_model(exp_args)

        logger.info("-------------------------------------------------------------------------")
        exp_args['load_path'] = tb_path + "best_tb_val_acc_model.pt"
        logger.info("Evaluating model with best val acc on Toybox")
        eval_model(exp_args)

        logger.info("-------------------------------------------------------------------------")
        exp_args['load_path'] = tb_path + "final_model.pt"
        logger.info("Evaluating model at the end of training")
        eval_model(exp_args)

        logger.info("-------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
