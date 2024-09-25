import numpy as np
import datetime
import random
import os

import torch.utils.data as torchdata
import torch
import torch.utils.tensorboard as tb

import datasets
import models_dual_ssl
import networks
import tb_in12_transforms
import parsers
import utils

OUT_DIR = "../out/DUAL_SSL_DOM_MMD_V1/"
os.makedirs(OUT_DIR, exist_ok=True)


def show_images(loader, mean, std, aug, n_cols):
    for _, (idxs, images, labels) in enumerate(loader):
        # print(type(idxs), len(idxs), idxs[0].shape, len(images), images[0].shape, images[1].shape, labels)

        concat_images = torch.concat(images)
        batch_images = utils.get_images(concat_images, mean=mean, std=std, aug=aug, cols=n_cols)
        batch_images.show()
        break


def get_dataloader(dset, transform, ssl_type, batch_size, shuffle, workers, hypertune):
    assert dset in ["toybox", "in12"]
    if dset == "toybox":
        data_train = datasets.ToyboxDatasetSSLPaired(rng=np.random.default_rng(0), transform=transform,
                                                     fraction=1.0, hypertune=hypertune, distort=ssl_type)
        loader_train = torchdata.DataLoader(data_train, batch_size=batch_size // 4, shuffle=shuffle,
                                            num_workers=workers, pin_memory=True, persistent_workers=True,
                                            drop_last=True)
    else:
        if ssl_type == 'class':
            data_train = datasets.IN12CategorySSLWithLabels(transform=transform, fraction=1.0,
                                                            hypertune=hypertune)
        else:
            data_train = datasets.IN12SSLWithLabels(transform=transform, hypertune=hypertune)

        loader_train = torchdata.DataLoader(data_train, batch_size=batch_size // 2, shuffle=shuffle,
                                            num_workers=workers, pin_memory=True, persistent_workers=True,
                                            drop_last=True)

    return loader_train


def run_training(exp_args):
    # Extract parameters for training runs
    steps = exp_args['iters']
    num_epochs = exp_args['epochs']
    hypertune = not exp_args['final']
    workers = exp_args['workers']
    tb_ssl_type = exp_args['tb_ssl_type']
    in12_ssl_type = exp_args['in12_ssl_type']
    b_size = exp_args['bsize']
    no_save = exp_args['no_save']
    save_dir = exp_args['save_dir']
    tb_alpha = exp_args['tb_alpha']
    in12_alpha = exp_args['in12_alpha']
    save_freq = exp_args['save_freq']
    div_alpha = exp_args['div_alpha']
    ignore_div_loss = exp_args['ignore_div_loss']
    tb_ssl_loss = exp_args['tb_ssl_loss']
    in12_ssl_loss = exp_args['in12_ssl_loss']
    asymmetric = exp_args['asymmetric']
    use_ot = exp_args['use_ot']
    div_metric = exp_args['div_metric']
    combined_forward_pass = not exp_args['separate_forward_pass']
    track_knn_acc = exp_args['track_knn_acc']
    queue_factor = exp_args['queue_factor']
    split_div_loss = exp_args['split_div_loss']
    num_split_images = exp_args['num_split_images']
    div_alpha_schedule = exp_args['div_alpha_type']
    div_alpha_start = exp_args['div_alpha_start']
    ind_mmd_loss = exp_args['ind_mmd_loss']
    use_bb_mmd = exp_args['use_bb_mmd']

    tb_transform_train = tb_in12_transforms.get_ssl_transform(dset="toybox")
    tb_loader_train = get_dataloader(dset="toybox", batch_size=b_size, ssl_type=tb_ssl_type,
                                     transform=tb_transform_train, hypertune=hypertune, workers=workers, shuffle=True)

    in12_transform_train = tb_in12_transforms.get_ssl_transform(dset="in12")
    in12_loader_train = get_dataloader(dset="in12", batch_size=b_size, ssl_type=in12_ssl_type,
                                       transform=in12_transform_train, hypertune=hypertune, workers=workers,
                                       shuffle=True)

    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
        OUT_DIR + save_dir + "/"
    assert not os.path.isdir(tb_path), f"{tb_path} already exists.."
    tb_writer = tb.SummaryWriter(log_dir=tb_path + "tensorboard/") if not no_save else None
    logger = utils.create_logger(log_level_str=exp_args['log'], log_file_name=tb_path + "log.txt", no_save=no_save)

    # logger.info(utils.online_mean_and_sd(tb_loader_train), utils.online_mean_and_sd(in12_loader_train))

    if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
        load_file_path = exp_args['load_path'] + "final_model.pt"
        load_file = torch.load(load_file_path)
        logger.info(f"Loading weights from {load_file_path} ({load_file['type']})")

        bb_wts = load_file['backbone']
        ssl_wts = load_file['ssl_head'] if 'ssl_head' in load_file.keys() else None
        net = networks.ResNet18SSL(backbone_weights=bb_wts, ssl_weights=ssl_wts)
    else:
        net = networks.ResNet18SSL()

    if ignore_div_loss:
        if split_div_loss:
            model_name = models_dual_ssl.DualSSLWithinDomainSplitDistMatchingModelBase
        else:
            model_name = models_dual_ssl.DualSSLWithinDomainDistMatchingModelBase
    else:
        if split_div_loss:
            model_name = models_dual_ssl.DualSSLWithinDomainSplitDistMatchingModel
        else:
            model_name = models_dual_ssl.DualSSLWithinDomainAllDistMatchingModel
    ssl_model = model_name(
        network=net, src_loader=tb_loader_train, trgt_loader=in12_loader_train, logger=logger, no_save=no_save,
        tb_ssl_loss=tb_ssl_loss, in12_ssl_loss=in12_ssl_loss, tb_alpha=tb_alpha, in12_alpha=in12_alpha,
        div_alpha=div_alpha, ignore_div_loss=ignore_div_loss, asymmetric=asymmetric, use_ot=use_ot,
        div_metric=div_metric, combined_fwd_pass=combined_forward_pass, div_alpha_schedule=div_alpha_schedule,
        div_alpha_start=div_alpha_start, track_knn_acc=track_knn_acc, queue_size=queue_factor * b_size,
        num_split_images=num_split_images, ind_mmd_loss=ind_mmd_loss, use_bb_mmd=use_bb_mmd)

    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=exp_args['lr'], weight_decay=exp_args['wd'],
                                momentum=0.9, nesterov=True)
    optimizer.add_param_group({'params': net.ssl_head.parameters(), 'lr': exp_args['lr']})

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])

    if not no_save:
        logger.info("Experimental details and results saved to {}".format(tb_path))
        net.save_model(fpath=tb_path + "initial_model.pt")
        if save_freq > 0:
            net.save_model(fpath=tb_path + "model_epoch_0.pt")

    for ep in range(1, num_epochs + 1):
        ssl_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                        ep=ep, ep_total=num_epochs, writer=tb_writer)
        if not no_save and save_freq > 0 and ep % save_freq == 0:
            net.save_model(fpath=tb_path + f"model_epoch_{ep}.pt")

    if not no_save:
        net.save_model(fpath=tb_path + "final_model.pt")
        if save_freq > 0:
            net.save_model(fpath=tb_path + f"model_epoch_{num_epochs}.pt")
        tb_writer.close()

        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['tb_transform'] = str(tb_transform_train)
        exp_args['in12_transform'] = str(in12_transform_train)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))


def main():
    parser = parsers.get_dual_ssl_class_mmd_v1_parser()
    exp_args = vars(parser.parse_args())
    run_training(exp_args=exp_args)


if __name__ == '__main__':
    main()
