"""Module to run IN-12 training with noisy labels from ensembles of JAN pretrained models"""
import collections
from tabulate import tabulate
import torchvision.transforms as transforms
import torch.utils.data as torchdata
import torch
import datetime
import os
import time
import torch.utils.tensorboard as tb

import analyze_csv as ac
import datasets
import networks
import models
import utils

TEMP_DIR = "../temp/IN12_SUP_NOISY/"
OUT_DIR = "../out/IN12_SUP_NOISY/"
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



def get_all_correct_indices(correct_idxs_dict, incorrect_idxs_dict, idxs):
    correct_idxs = []
    for idx in correct_idxs_dict[idxs[0]]:
        cnt = 0
        for i in idxs:
            assert idx in correct_idxs_dict[i] or idx in incorrect_idxs_dict[i]
            cnt += 1 if idx in correct_idxs_dict[i] else 0
        if cnt == len(idxs):
            correct_idxs.append(idx)

    return correct_idxs


def form_correctness_dict(correct_idxs_dict, incorrect_idxs_dict, idxs):
    correct_dict = collections.Counter()
    for idx in correct_idxs_dict[idxs[0]]:
        cnt = 0
        for i in idxs:
            assert idx in correct_idxs_dict[i] or idx in incorrect_idxs_dict[i]
            cnt += 1 if idx in correct_idxs_dict[i] else 0
        correct_dict[cnt] += 1
    for idx in incorrect_idxs_dict[idxs[0]]:
        cnt = 0
        for i in idxs:
            assert idx in correct_idxs_dict[i] or idx in incorrect_idxs_dict[i]
            cnt += 1 if idx in correct_idxs_dict[i] else 0
        correct_dict[cnt] += 1
    return correct_dict


def count_incorrect_idxs_all_same(incorrect_idxs, csvs_dict, idxs):
    incorrect_idxs_all_models = incorrect_idxs[idxs[0]]
    for i in idxs:
        incorrect_idxs_all_models = incorrect_idxs_all_models.intersection(incorrect_idxs[i])

    incorrect_predictions = {}
    for i in idxs:
        incorrect_predictions[i] = csvs_dict[i].get_predictions(indices=incorrect_idxs_all_models)

    count = 0
    for idx in incorrect_idxs_all_models:
        preds = []
        for i in idxs:
            preds.append(incorrect_predictions[i][idx])
        assert (len(preds) == len(idxs))
        if len(set(preds)) == 1:
            count += 1
    return count


def get_incorrect_idxs_all_same(incorrect_idxs, csvs_dict, idxs):
    incorrect_idxs_all_models = incorrect_idxs[idxs[0]]
    for i in idxs:
        incorrect_idxs_all_models = incorrect_idxs_all_models.intersection(incorrect_idxs[i])

    incorrect_predictions = {}
    for i in idxs:
        incorrect_predictions[i] = csvs_dict[i].get_predictions(indices=incorrect_idxs_all_models)

    incorrect_idxs = []
    incorrect_preds = []
    for idx in incorrect_idxs_all_models:
        preds = []
        for i in idxs:
            preds.append(incorrect_predictions[i][idx])
        assert (len(preds) == len(idxs))
        if len(set(preds)) == 1:
            incorrect_idxs.append(idx)
            incorrect_preds.append(preds[0])
    return incorrect_idxs, incorrect_preds


def get_indices():
    idxs = [1, 2, 4, 5, 7, 8]
    model_paths = {}
    for i in idxs:
        model_paths[i] = f"../out/JAN/TB_IN12/jan_tb_sup_final_{i}/output/final_model/"
    tb_tr_csvs, tb_te_csvs, in12_tr_csvs, in12_te_csvs = {}, {}, {}, {}
    for i in idxs:
        tb_tr_csvs[i] = ac.CSVUtil(fpath=model_paths[i] + "tb_train.csv")
        tb_te_csvs[i] = ac.CSVUtil(fpath=model_paths[i] + "tb_test.csv")
        in12_tr_csvs[i] = ac.CSVUtil(fpath=model_paths[i] + "in12_train.csv")
        in12_te_csvs[i] = ac.CSVUtil(fpath=model_paths[i] + "in12_test.csv")

    tb_tr_correct_idxs, tb_tr_incorrect_idxs, tb_te_correct_idxs, tb_te_incorrect_idxs = {}, {}, {}, {}
    in12_tr_correct_idxs, in12_tr_incorrect_idxs, in12_te_correct_idxs, in12_te_incorrect_idxs = {}, {}, {}, {}
    for i in idxs:
        tb_tr_correct_idxs[i], tb_tr_incorrect_idxs[i] = tb_tr_csvs[i].get_correct_incorrect_indices()
        tb_te_correct_idxs[i], tb_te_incorrect_idxs[i] = tb_te_csvs[i].get_correct_incorrect_indices()
        in12_tr_correct_idxs[i], in12_tr_incorrect_idxs[i] = in12_tr_csvs[i].get_correct_incorrect_indices()
        in12_te_correct_idxs[i], in12_te_incorrect_idxs[i] = in12_te_csvs[i].get_correct_incorrect_indices()

    tb_tr_correct_dict = form_correctness_dict(tb_tr_correct_idxs, tb_tr_incorrect_idxs, idxs=idxs)
    tb_te_correct_dict = form_correctness_dict(tb_te_correct_idxs, tb_te_incorrect_idxs, idxs=idxs)
    in12_tr_correct_dict = form_correctness_dict(in12_tr_correct_idxs, in12_tr_incorrect_idxs, idxs=idxs)
    in12_te_correct_dict = form_correctness_dict(in12_te_correct_idxs, in12_te_incorrect_idxs, idxs=idxs)

    ret_arr = []
    for cnt in range(len(idxs) + 1):
        ret_arr.append([cnt, in12_tr_correct_dict[cnt], in12_te_correct_dict[cnt], tb_tr_correct_dict[cnt],
                        tb_te_correct_dict[cnt]])

    # print(tabulate(ret_arr, headers=["Num Correct", "IN-12 Train", "IN-12 Test", "Toybox Train", "Toybox Test"],
    #                tablefmt="psql"))

    tb_tr_cnt = count_incorrect_idxs_all_same(tb_tr_incorrect_idxs, tb_tr_csvs, idxs=idxs)
    tb_te_cnt = count_incorrect_idxs_all_same(tb_te_incorrect_idxs, tb_te_csvs, idxs=idxs)
    in12_tr_cnt = count_incorrect_idxs_all_same(in12_tr_incorrect_idxs, in12_tr_csvs, idxs=idxs)
    in12_te_cnt = count_incorrect_idxs_all_same(in12_te_incorrect_idxs, in12_te_csvs, idxs=idxs)
    # print(tb_tr_cnt, tb_te_cnt, in12_tr_cnt, in12_te_cnt)

    in12_tr_all_correct_indices = get_all_correct_indices(in12_tr_correct_idxs, in12_tr_incorrect_idxs, idxs=idxs)
    in12_tr_correct_preds_dict = in12_tr_csvs[idxs[0]].get_predictions(indices=in12_tr_all_correct_indices)
    in12_tr_correct_preds = [in12_tr_correct_preds_dict[idx] for idx in in12_tr_all_correct_indices]
    in12_tr_same_incorrect_indices, in12_tr_incorrect_preds = get_incorrect_idxs_all_same(in12_tr_incorrect_idxs,
                                                                                          in12_tr_csvs, idxs=idxs)
    # print(len(in12_tr_all_correct_indices), len(in12_tr_correct_preds),
    #       len(in12_tr_same_incorrect_indices), len(in12_tr_incorrect_preds))
    all_same_indices = in12_tr_all_correct_indices + in12_tr_same_incorrect_indices
    all_same_preds = in12_tr_correct_preds + in12_tr_incorrect_preds
    return all_same_indices, all_same_preds


def run_training(indices, predictions):
    num_epochs = 50
    b_size = 256
    n_workers = 8
    steps = 100
    lr = 0.025
    wd = 1e-5
    save_dir = ""
    no_save = True

    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
        OUT_DIR + save_dir + "/"
    assert not os.path.isdir(tb_path), f"{tb_path} already exists"
    tb_writer = tb.SummaryWriter(log_dir=tb_path) if not no_save else None
    logger = utils.create_logger(log_level_str="info", log_file_name=tb_path + "log.txt", no_save=no_save)

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
                                                                       interpolation=transforms.
                                                                       InterpolationMode.BICUBIC),
                                          transforms.RandomOrder(color_transforms),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD),
                                          transforms.RandomErasing(p=0.5)
                                          ])
    data_train = datasets.DatasetIN12FromIndices(hypertune=False, transform=transform_train,
                                                 train_indices=indices, labels=predictions)
    noisy_loader_train = torchdata.DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                              pin_memory=True)

    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])

    gt_data_train = datasets.DatasetIN12(train=True, transform=transform_test, fraction=1.0, hypertune=False)
    gt_loader_train = torchdata.DataLoader(gt_data_train, batch_size=b_size, shuffle=False, num_workers=n_workers)

    gt_data_test = datasets.DatasetIN12(train=False, transform=transform_test, fraction=1.0, hypertune=False)
    gt_loader_test = torchdata.DataLoader(gt_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)

    net = networks.ResNet18Sup(num_classes=12, pretrained=False)

    pre_model = models.SupModel(network=net, source_loader=noisy_loader_train, logger=logger, no_save=True,
                                linear_eval=False)

    bb_lr_wt = 1.0
    optimizer = torch.optim.Adam(net.classifier_head.parameters(), lr=lr, weight_decay=wd)
    optimizer.add_param_group({'params': net.backbone.parameters(), 'lr': lr * bb_lr_wt})

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])

    all_loaders = {
        "noisy_in12_train": noisy_loader_train,
        "gt_in12_train": gt_loader_train,
        "gt_in12_test": gt_loader_test
    }

    get_train_test_acc(model=pre_model, loaders=all_loaders, writer=tb_writer, step=0, logger=logger, no_save=no_save)
    pre_model.calc_val_loss(loader=gt_loader_train, loader_name="gt_in12_train", ep=0, steps=steps, writer=tb_writer,
                            no_save=no_save)
    pre_model.calc_val_loss(loader=gt_loader_test, loader_name="gt_in12_test", ep=0, steps=steps, writer=tb_writer,
                            no_save=no_save)

    for ep in range(1, num_epochs + 1):
        pre_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps, ep=ep, ep_total=num_epochs,
                        writer=tb_writer)
        if ep % 5 == 0 and ep != num_epochs:
            pre_model.calc_val_loss(loader=gt_loader_train, loader_name="gt_in12_train", ep=0, steps=steps,
                                    writer=tb_writer,
                                    no_save=no_save)
            pre_model.calc_val_loss(loader=gt_loader_test, loader_name="gt_in12_test", ep=0, steps=steps,
                                    writer=tb_writer,
                                    no_save=no_save)
        if ep % 10 == 0 and ep != num_epochs:
            get_train_test_acc(model=pre_model, loaders=all_loaders,
                               writer=tb_writer, step=ep * steps, logger=logger, no_save=no_save)

    pre_model.calc_val_loss(loader=gt_loader_train, loader_name="gt_in12_train", ep=0, steps=steps, writer=tb_writer,
                            no_save=no_save)
    pre_model.calc_val_loss(loader=gt_loader_test, loader_name="gt_in12_test", ep=0, steps=steps, writer=tb_writer,
                            no_save=no_save)

    accs = get_train_test_acc(model=pre_model, loaders=all_loaders, writer=tb_writer, step=num_epochs * steps,
                              logger=logger, no_save=no_save)

def main():
    """Main method"""
    indices, preds = get_indices()
    run_training(indices=indices, predictions=preds)


if __name__ == "__main__":
    main()
