"""Module implementing the MTL pretraining algorithm"""
import argparse
import datetime
import os
import numpy as np

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb

import datasets
import models
import networks
import utils

TB_OUT_DIR = "../out/TB_SSL/"
IN12_OUT_DIR = "../out/IN12_SSL/"
os.makedirs(TB_OUT_DIR, exist_ok=True)
os.makedirs(IN12_OUT_DIR, exist_ok=True)


def get_parser():
    """Return parser for the experiment"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--epochs", "-e", default=100, type=int, help="Set the number of epochs of training")
    parser.add_argument("--iters", "-it", default=250, type=int, help="Set the number of iters of training per epoch")
    parser.add_argument("--bsize", "-b", default=128, type=int, help="Set the batch size for experiments")
    parser.add_argument("--workers", "-w", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("--lr", "-lr", default=0.1, type=float, help="Set initial lr for experiments")
    parser.add_argument("-wd", "--wd", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--log", choices=['debug', 'info', 'warning', 'error', 'critical'], default='info',
                        help="Set the log level for the experiment", type=str)
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--no-save", action='store_true', default=False, help="Use this option to disable saving")
    parser.add_argument("--save-dir", default="", type=str, help="Directory to save")
    parser.add_argument("--decoupled", action='store_true', default=False, help="This flag allows us  to use "
                                                                                "decoupled contrastive loss")
    parser.add_argument("--dataset", required=True, choices=['in12', 'toybox', 'joint'], help="SSL dataset to train on")
    
    return vars(parser.parse_args())


def main():
    """Main method"""
    exp_args = get_parser()
    steps = exp_args['iters']
    num_epochs = exp_args['epochs']
    workers = exp_args['workers']
    no_save = exp_args['no_save']
    save_dir = exp_args['save_dir']
    decoupled_loss = exp_args['decoupled']
    dataset = exp_args['dataset'].lower()
    assert dataset in ['in12', 'toybox']

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.2, saturation=0.8)
    gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

    if dataset == 'in12':
        transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.RandomApply([gaussian_blur], p=0.2),
                                              transforms.Resize(256),
                                              transforms.RandomResizedCrop(size=224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
        data_train = datasets.DatasetIN12SSL(transform=transform_train, fraction=1.0, hypertune=True)
        loader_train = torchdata.DataLoader(data_train, batch_size=exp_args['bsize'], shuffle=True, num_workers=workers,
                                            pin_memory=True, persistent_workers=True, drop_last=True)
    else:
        transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.RandomApply([gaussian_blur], p=0.2),
                                              transforms.Resize(256),
                                              transforms.RandomResizedCrop(size=224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])
        data_train = datasets.ToyboxDatasetSSL(rng=np.random.default_rng(0), transform=transform_train, fraction=1.0,
                                               hypertune=True, distort='object')
        loader_train = torchdata.DataLoader(data_train, batch_size=exp_args['bsize'], shuffle=True, num_workers=workers,
                                            pin_memory=True, persistent_workers=True, drop_last=True)

    start_time = datetime.datetime.now()
    if dataset == "in12":
        tb_path = IN12_OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
            IN12_OUT_DIR + save_dir + "/"
    else:
        tb_path = TB_OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
            TB_OUT_DIR + save_dir + "/"
    assert not os.path.isdir(tb_path), f"{tb_path} already exists.."
    tb_writer = tb.SummaryWriter(log_dir=tb_path) if not no_save else None
    logger = utils.create_logger(log_level_str=exp_args['log'], log_file_name=tb_path+"log.txt", no_save=no_save)

    if not no_save:
        logger.info("Experimental details and results saved to {}".format(tb_path))

    # print(utils.online_mean_and_sd(trgt_loader_test))

    if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
        load_file_path = exp_args['load_path'] + "final_model.pt"
        load_file = torch.load(load_file_path)
        logger.info(f"Loading weights from {load_file_path} ({load_file['type']})")

        bb_wts = load_file['backbone']
        ssl_wts = load_file['ssl_head'] if 'ssl_head' in load_file.keys() else None
        net = networks.ResNet18SSL(backbone_weights=bb_wts, ssl_weights=ssl_wts)
    else:
        net = networks.ResNet18SSL()
    ssl_model = models.SSLModel(network=net, loader=loader_train, logger=logger, no_save=no_save,
                                decoupled=decoupled_loss)

    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=exp_args['lr'], weight_decay=exp_args['wd'])
    optimizer.add_param_group({'params': net.ssl_head.parameters(), 'lr': exp_args['lr']})

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])

    for ep in range(1, num_epochs + 1):
        ssl_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                        ep=ep, ep_total=num_epochs, writer=tb_writer)

    if not no_save:
        net.save_model(fpath=tb_path+"final_model.pt")
        tb_writer.close()

        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['train_transform'] = str(transform_train)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))


if __name__ == "__main__":
    main()
