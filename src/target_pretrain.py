"""Module implementing the supervised training for IN-12 dataset"""
import argparse
import numpy as np
import os
import datetime
import time
import csv

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb

import datasets
import models
import networks
import utils

TEMP_DIR = "../temp/IN12_SUP/"
OUT_DIR = "../out/IN12_SUP/"
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
    """Return parser for source pretrain experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="Number of epochs of training")
    parser.add_argument("-it", "--iters", default=500, type=int, help="Number of training steps per epoch")
    parser.add_argument("-b", "--bsize", default=128, type=int, help="Batch size for training")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("-f", "--final", default=False, action='store_true',
                        help="Use this flag to run experiment on train+val set")
    parser.add_argument("-lr", "--lr", default=0.05, type=float, help="Learning rate for the experiment")
    parser.add_argument("-wd", "--wd", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--fraction", "-frac", default=1.0, type=float,
                        help="Fraction of training data to use for training")
    parser.add_argument("--seed", default=-1, type=int, help="Seed for running experiments")
    parser.add_argument("--log", choices=["debug", "info", "warning", "error", "critical"],
                        default="info", type=str)
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use this flag to start from network pretrained on ILSVRC")
    parser.add_argument("--no-save", default=False, action='store_true', help="Set this flag to not save anything")
    parser.add_argument("--save-freq", default=-1, type=int, help="Frequence for saving model.")
    return vars(parser.parse_args())


def eval_model(exp_args):
    """Eval model"""
    hypertune = not exp_args['final']
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    logger = utils.create_logger(log_level_str='info', log_file_name="", no_save=True)

    # Load IN-12 Train dataset and dataloader
    src_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])

    src_data_train = datasets.DatasetIN12(train=True, hypertune=hypertune, fraction=1.0, transform=src_transform,
                                          equal_div=True)
    logger.info(f"Source Train dataset: {src_data_train}  Size: {len(src_data_train)}")
    src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers)

    # Load IN-12 Test dataset and dataloader
    src_data_test = datasets.DatasetIN12(train=False, transform=src_transform, fraction=1.0, hypertune=hypertune)
    logger.info(f"Source Test dataset: {src_data_test}  Size: {len(src_data_test)}")
    src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)

    # Load Toybox Train Dataset and dataloader
    trgt_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])

    trgt_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(exp_args['seed']), train=True,
                                             transform=trgt_transform,
                                             hypertune=hypertune, num_instances=-1,
                                             num_images_per_class=1000,
                                             )
    logger.info(f"Target Train dataset: {trgt_data_train}  Size: {len(trgt_data_train)}")
    trgt_loader_train = torchdata.DataLoader(trgt_data_train, batch_size=b_size, shuffle=False, num_workers=n_workers)

    # Load Toybox Test dataset and dataloader
    trgt_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=False, transform=trgt_transform,
                                            hypertune=hypertune)
    trgt_loader_test = torchdata.DataLoader(trgt_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)
    logger.info(f"Target Test dataset: {trgt_data_test}  Size: {len(trgt_data_test)}")

    load_file_path = exp_args['load_path']
    assert os.path.isfile(load_file_path)
    load_file = torch.load(load_file_path)
    assert load_file['type'] == networks.ResNet18Sup.__name__
    logger.info(f"Loading model weights from {load_file_path} ({load_file['type']})")
    bb_wts, cl_wts = load_file['backbone'], load_file['classifier']
    net = networks.ResNet18Sup(num_classes=12, backbone_weights=bb_wts, classifier_weights=cl_wts)

    model = models.SupModel(network=net, source_loader=src_loader_train, logger=logger, no_save=True)
    all_loaders = {
        'in12_train': src_loader_train,
        'in12_test': src_loader_test,
        'tb_train': trgt_loader_train,
        'tb_test': trgt_loader_test
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
    steps = exp_args['iters']
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    hypertune = not exp_args['final']
    frac = exp_args['fraction']
    no_save = exp_args['no_save']
    save_freq = exp_args['save_freq']
    
    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/"
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
                                              transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD),
                                              transforms.RandomErasing(p=0.5)
                                              ])
    src_data_train = datasets.DatasetIN12(train=True, hypertune=hypertune, fraction=frac, transform=src_transform_train,
                                          equal_div=True)
    logger.info(f"Dataset: {src_data_train}  Size: {len(src_data_train)}")
    src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers)
    
    src_transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    
    src_data_test = datasets.DatasetIN12(train=False, transform=src_transform_test, fraction=1.0, hypertune=hypertune)
    src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)
    
    trgt_transform_test = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])

    trgt_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(exp_args['seed']), train=True,
                                             transform=trgt_transform_test,
                                             hypertune=hypertune, num_instances=-1,
                                             num_images_per_class=1000,
                                             )
    logger.info(f"Dataset: {trgt_data_train}  Size: {len(trgt_data_train)}")
    trgt_loader_train = torchdata.DataLoader(trgt_data_train, batch_size=b_size, shuffle=False, num_workers=n_workers)
    trgt_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=False, transform=trgt_transform_test,
                                            hypertune=hypertune)
    trgt_loader_test = torchdata.DataLoader(trgt_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)
    
    # logger.debug(utils.online_mean_and_sd(src_loader_train), utils.online_mean_and_sd(src_loader_test))
    # logger.debug(utils.online_mean_and_sd(trgt_loader_test))
    
    if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
        load_file_path = exp_args['load_path'] + "final_model.pt"
        load_file = torch.load(load_file_path)
        logger.info(f"Loading model weights from {load_file_path} ({load_file['type']})")
        bb_wts = load_file['backbone']
        cl_wts = load_file['classifier'] if 'classifier' in load_file.keys() else None
        net = networks.ResNet18Sup(num_classes=12, backbone_weights=bb_wts, classifier_weights=cl_wts)
    else:
        net = networks.ResNet18Sup(num_classes=12, pretrained=exp_args['pretrained'])
    pre_model = models.SupModel(network=net, source_loader=src_loader_train, logger=logger, no_save=no_save)

    bb_lr_wt = 0.1 if exp_args['pretrained'] else 1.0
    optimizer = torch.optim.Adam(net.backbone.parameters(), lr=exp_args['lr']*bb_lr_wt, weight_decay=exp_args['wd'])
    optimizer.add_param_group({'params': net.classifier_head.parameters(), 'lr': exp_args['lr']})
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])

    all_loaders = {
        'in12_train': src_loader_train,
        'in12_test': src_loader_test,
        'tb_train': trgt_loader_train,
        'tb_test': trgt_loader_test
    }
    get_train_test_acc(model=pre_model, loaders=all_loaders, writer=tb_writer, step=0, logger=logger, no_save=no_save)
    pre_model.calc_val_loss(loader=src_loader_test, loader_name="in12_test", ep=0, steps=steps, writer=tb_writer,
                            no_save=no_save)
    
    if not no_save:
        logger.info("Experimental details and results saved to {}".format(tb_path))
        net.save_model(fpath=tb_path+"model_epoch_0.pt")
    
    for ep in range(1, num_epochs + 1):
        pre_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps, ep=ep, ep_total=num_epochs,
                        writer=tb_writer)
        if ep % 5 == 0 and ep != num_epochs:
            pre_model.calc_val_loss(loader=src_loader_test, loader_name="in12_test", ep=ep, steps=steps,
                                    writer=tb_writer, no_save=no_save)
        if ep % 10 == 0 and ep != num_epochs:
            get_train_test_acc(model=pre_model, loaders=all_loaders,
                               writer=tb_writer, step=ep * steps, logger=logger, no_save=no_save)

        if not no_save:
            if save_freq > 0 and ep % save_freq == 0:
                net.save_model(fpath=tb_path + f"model_epoch_{ep}.pt")

    pre_model.calc_val_loss(loader=src_loader_test, loader_name="in12_test", ep=num_epochs, steps=steps,
                            writer=tb_writer, no_save=no_save)
    
    src_tr_acc, src_te_acc, trgt_tr_acc, trgt_te_acc = get_train_test_acc(model=pre_model, loaders=all_loaders,
                                                                          writer=tb_writer, step=num_epochs * steps,
                                                                          logger=logger, no_save=no_save)
    
    if not no_save:
        tb_writer.close()
        net.save_model(fpath=tb_path + f"model_epoch_{num_epochs}.pt")
        net.save_model(fpath=tb_path + f"final_model.pt")
        
        exp_args['in12_train'] = src_tr_acc
        exp_args['in12_test'] = src_te_acc
        exp_args['tb_train'] = trgt_tr_acc
        exp_args['tb_test'] = trgt_te_acc
        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['train_transform'] = str(src_transform_train)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))

        logger.info("-------------------------------------------------------------------------")
        exp_args['load_path'] = tb_path + "final_model.pt"
        logger.info("Evaluating final model")
        eval_model(exp_args)

        logger.info("-------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
