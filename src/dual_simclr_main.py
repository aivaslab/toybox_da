"""Module containing main method for unsupervised algorithms with the combined data"""

import argparse
import numpy as np
import os
import datetime

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb

import datasets
import models_ssl
import networks_ssl
import utils

TEMP_DIR = "../temp/DUAL_SIMCLR/"
OUT_DIR = "../out/DUAL_SIMCLR/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_parser():
    """Return parser for SimCLR experiments on both IN-12 and Toybox"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="Number of epochs of training")
    parser.add_argument("-it", "--iters", default=500, type=int, help="Number of training steps per epoch")
    parser.add_argument("-b", "--bsize", default=128, type=int, help="Batch size for training")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("-f", "--final", default=False, action='store_true',
                        help="Use this flag to run experiments on train+val set")
    parser.add_argument("-lr", "--lr", default=0.05, type=float, help="Learning rate for the experiment")
    parser.add_argument("-wd", "--wd", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--source-frac", "-sf", default=0.2, type=float,
                        help="Set fraction of training images to be used for source dataset")
    parser.add_argument("--target-frac", "-tf", default=1.0, type=float,
                        help="Set fraction of training images to be used for target dataset")
    parser.add_argument("--seed", default=-1, type=int, help="Seed for running experiments")
    parser.add_argument("--log", choices=["debug", "info", "warning", "error", "critical"],
                        default="info", type=str)
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use this flag to start from pretrained network")
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--no-save", default=False, action='store_true', help="Set this flag to not save anything")
    return vars(parser.parse_args())


def main():
    """Main method"""
    
    exp_args = get_parser()
    exp_args['seed'] = None if exp_args['seed'] == -1 else exp_args['seed']
    num_epochs = exp_args['epochs']
    steps = exp_args['iters']
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    hypertune = not exp_args['final']
    target_frac = exp_args['target_frac']
    source_frac = exp_args['source_frac']
    no_save = exp_args['no_save']
    
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
    
    src_data_train = datasets.ToyboxDatasetSSL(rng=np.random.default_rng(exp_args['seed']),
                                               transform=src_transform_train,
                                               hypertune=hypertune, fraction=source_frac)
    logger.info(f"Source dataset: {src_data_train}  Size: {len(src_data_train)}")
    src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                            drop_last=True)
    
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
    trgt_data_train = datasets.DatasetIN12SSL(transform=trgt_transform_train, fraction=target_frac,
                                              hypertune=hypertune)
    trgt_loader_train = torchdata.DataLoader(trgt_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                             drop_last=True)
    logger.info(f"Target dataset: {trgt_data_train}  Size: {len(trgt_data_train)}")
    
    # logger.debug(utils.online_mean_and_sd(src_loader_train), utils.online_mean_and_sd(src_loader_test))
    # logger.debug(utils.online_mean_and_sd(trgt_loader_test))
    
    net = networks_ssl.SimCLRResNet18()
    
    model = models_ssl.SimCLRModel(network=net, source_loader=src_loader_train, target_loader=trgt_loader_train,
                                   logger=logger, temp=0.1, no_save=no_save)
    
    optimizer = torch.optim.Adam(net.student_backbone.parameters(), lr=exp_args['lr'], weight_decay=exp_args['wd'])
    optimizer.add_param_group({'params': net.student_projection.parameters(), 'lr': exp_args['lr']})
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])
    
    for ep in range(1, num_epochs + 1):
        model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                    ep=ep, ep_total=num_epochs, writer=tb_writer)
    
    if not no_save:
        tb_writer.close()
        save_dict = {
            'type': net.__class__.__name__,
            'backbone': net.student_backbone.model.state_dict(),
            'projection': net.student_projection.state_dict(),
            'teacher_backbone': net.teacher_backbone.model.state_dict(),
            'teacher_projection': net.teacher_projection.state_dict(),
        }
        torch.save(save_dict, tb_path + "final_model.pt")
        
        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['train_transform'] = str(src_transform_train)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))


if __name__ == "__main__":
    main()
