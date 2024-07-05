"""Module implementing the MTL pretraining task for Toybox->IN-12"""
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
import torch.nn.functional as func
import torchvision.transforms as transforms

import utils
import ccmmd


class ModelLE:
    """model for linear eval of network"""

    def __init__(self, network, train_loader, test_loader, logger):
        self.network = network
        self.train_loader = utils.ForeverDataLoader(train_loader)
        self.test_loader = test_loader
        self.logger = logger
        self.network.cuda()

    def train(self, optimizer, scheduler, ep, ep_total, steps):
        """Train the network for 1 epoch"""
        self.network.set_linear_eval()
        num_batches = 0
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            idx, images, labels = self.train_loader.get_next_batch()
            images, labels = images.cuda(), labels.cuda()
            logits = self.network.forward(images)
            loss = src_criterion(logits, labels)
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ce_loss_total += loss.item()
            num_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  LR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'], ce_loss_total / num_batches,
                                                                                time.time() - start_time)
                )

        self.logger.info("Ep: {}/{}  Step: {}/{}  LR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'], ce_loss_total / num_batches,
                                                                         time.time() - start_time)
        )

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)

    def get_val_loss_dict(self, loader, loader_name):
        """Calculate loss on provided dataloader"""
        start_time = time.time()
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses_dict = {}
        num_items = 0
        total_loss = 0.0
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
                loss = criterion(logits, labels)
                for i in range(loss.shape[0]):
                    total_loss += loss[i].item()
                    num_items += 1
                    idx = idxs[1][i].item()
                    losses_dict[idx] = loss[i].item()
        avg_loss = total_loss / num_items

        self.logger.info("Validation Loss -- {:s}: {:.2f}     T: {:.2f}s".format(loader_name, avg_loss,
                                                                                 time.time() - start_time))
        return losses_dict

    def get_eval_dicts(self, loader, loader_name):
        """Evaluate the model on the provided dataloader"""
        start_time = time.time()
        n_total = 0
        n_correct = 0
        labels_dict = {}
        preds_dict = {}
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            for i in range(pred.shape[0]):
                idx = idxs[1][i].item()
                labels_dict[idx] = labels[i].item()
                preds_dict[idx] = pred[i].item()
                n_total += 1
                n_correct += 1 if labels[i].item() == pred[i].item() else 0
        acc = 100. * n_correct / n_total
        self.logger.info("Accuracy -- {:s}: {:.2f}     T: {:.2f}s".format(loader_name, acc,
                                                                          time.time() - start_time))
        return labels_dict, preds_dict


class MTLModel:
    """Module implementing the MTL method for pretraining the DA model"""

    def __init__(self, network, source_loader, target_loader, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.network.cuda()

    def train(self, optimizer, scheduler, steps, ep, ep_total, lmbda, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ce_loss_total = 0.0
        ssl_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        trgt_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            src_logits = self.network.forward_class(src_images)
            src_loss = src_criterion(src_logits, src_labels)

            trgt_idx, trgt_images = self.target_loader.get_next_batch()
            trgt_images = torch.cat(trgt_images, dim=0)
            trgt_images = trgt_images.cuda()
            trgt_feats = self.network.forward_ssl(trgt_images)
            logits, labels = utils.info_nce_loss(features=trgt_feats, temp=0.5)
            trgt_loss = trgt_criterion(logits, labels)

            total_loss = src_loss + lmbda * trgt_loss
            total_loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ce_loss_total += src_loss.item()
            ssl_loss_total += trgt_loss.item()
            num_batches += 1

            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ssl_loss_ep': ssl_loss_total / num_batches,
                    'ssl_loss_batch': trgt_loss.item(),
                    'ce_loss_ep': ce_loss_total / num_batches,
                    'ce_loss_batch': src_loss.item(),
                    'total_loss_batch': total_loss.item()
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'cl_head': optimizer.param_groups[1]['lr'],
                    'ssl_head': optimizer.param_groups[2]['lr'],
                },
                global_step=(ep - 1) * steps + num_batches,
            )

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{} Step:{}/{}  BLR: {:.3f}  CLR: {:.3f}  SLR: {:.3f}  CE: {:.3f}  SSL: {:.3f}"
                                 "T: {:.2f}s".
                                 format(ep, ep_total, step, steps,
                                        optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                        optimizer.param_groups[2]['lr'], ce_loss_total / num_batches,
                                        ssl_loss_total / num_batches, time.time() - start_time)
                                 )
        self.logger.info("Ep: {}/{} Step:{}/{}  BLR: {:.3f}  CLR: {:.3f}  SLR: {:.3f}  CE: {:.3f}  SSL: {:.3f}"
                         "T: {:.2f}s".
                         format(ep, ep_total, steps, steps,
                                optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                optimizer.param_groups[2]['lr'], ce_loss_total / num_batches,
                                ssl_loss_total / num_batches, time.time() - start_time)
                         )

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward_class(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)


class DualSSLModel:
    """Module implementing the SSL method for pretraining the DA model with both TB and IN-12 data"""

    def __init__(self, network, src_loader, trgt_loader, logger, no_save, decoupled, tb_alpha, in12_alpha,
                 combined):
        self.network = network
        self.src_loader = utils.ForeverDataLoader(src_loader)
        self.trgt_loader = utils.ForeverDataLoader(trgt_loader)
        self.logger = logger
        self.network.cuda()
        self.network.freeze_train()
        self.no_save = no_save
        self.decoupled = decoupled
        self.tb_alpha = tb_alpha
        self.in12_alpha = in12_alpha
        self.combined = combined

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_loss_total = 0.0
        trgt_loss_total = 0.0
        ssl_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images = self.src_loader.get_next_batch()
            trgt_idx, trgt_images, trgt_labels = self.trgt_loader.get_next_batch()

            if self.combined:
                anchors, positives = torch.cat([src_images[0], trgt_images[0]], dim=0), \
                    torch.cat([src_images[1], trgt_images[1]], dim=0)

                images = torch.cat([anchors, positives], dim=0)
                images = images.cuda()
                feats = self.network.forward(images)
                if self.decoupled:
                    loss = utils.decoupled_contrastive_loss(features=feats, temp=0.5)
                else:
                    logits, labels = utils.info_nce_loss(features=feats, temp=0.5)
                    loss = criterion(logits, labels)
                src_loss, trgt_loss = torch.tensor(0.0), torch.tensor(0.0)
            else:
                src_images, trgt_images = torch.cat(src_images, dim=0), torch.cat(trgt_images, dim=0)
                images = torch.cat([src_images, trgt_images], dim=0)
                images = images.cuda()
                feats = self.network.forward(images)
                src_size = src_images.shape[0]
                src_feats, trgt_feats = feats[:src_size], feats[src_size:]

                if self.decoupled:
                    src_loss = utils.decoupled_contrastive_loss(features=src_feats, temp=0.5)
                    trgt_loss = utils.decoupled_contrastive_loss(features=trgt_feats, temp=0.5)
                else:
                    src_logits, src_labels = utils.info_nce_loss(features=src_feats, temp=0.5)
                    src_loss = criterion(src_logits, src_labels)
                    trgt_logits, trgt_labels = utils.info_nce_loss(features=trgt_feats, temp=0.5)
                    trgt_loss = criterion(trgt_logits, trgt_labels)

                loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss
                src_loss_total += src_loss.item()
                trgt_loss_total += trgt_loss.item()
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ssl_loss_total += loss.item()
            num_batches += 1
            # if 0 <= step - halfway < 1:
            #     if self.combined:
            #         self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  "
            #                          "SSL: {:.3f}  T: {:.2f}s".format(
            #                              ep, ep_total, step, steps,
            #                              optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            #                              ssl_loss_total / num_batches, time.time() - start_time)
            #                          )
            #     else:
            #         self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
            #                          "SSL: {:.3f}  T: {:.2f}s".format(
            #                             ep, ep_total, step, steps,
            #                             optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            #                             src_loss_total / num_batches, trgt_loss_total / num_batches,
            #                             ssl_loss_total / num_batches, time.time() - start_time)
            #                          )

            if not self.no_save:
                if self.combined:
                    scalar_dict = {
                        'ssl_loss_ep': ssl_loss_total / num_batches,
                        'ssl_loss_batch': loss.item(),
                    }
                else:
                    scalar_dict = {
                        'src_ssl_loss_ep': src_loss_total / num_batches,
                        'src_ssl_loss_batch': src_loss.item(),
                        'trgt_ssl_loss_ep': trgt_loss_total / num_batches,
                        'trgt_ssl_loss_batch': trgt_loss.item(),
                        'ssl_loss_ep': ssl_loss_total / num_batches,
                        'ssl_loss_batch': loss.item(),
                    }
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict=scalar_dict,
                    global_step=(ep - 1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'ssl_head': optimizer.param_groups[1]['lr'],
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
        if self.combined:
            self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  "
                             "SSL: {:.3f}  T: {:.2f}s".format(
                                ep, ep_total, steps, steps,
                                optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                ssl_loss_total / num_batches, time.time() - start_time)
                             )
        else:
            self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
                             "SSL: {:.3f}  T: {:.2f}s".format(
                                ep, ep_total, steps, steps,
                                optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                src_loss_total / num_batches, trgt_loss_total / num_batches,
                                ssl_loss_total / num_batches, time.time() - start_time)
                             )


class SSLModel:
    """Module implementing the SSL method for pretraining the DA model"""

    def __init__(self, network, loader, logger, no_save, decoupled):
        self.network = network
        self.loader = utils.ForeverDataLoader(loader)
        self.logger = logger
        self.network.cuda()
        self.network.freeze_train()
        self.no_save = no_save
        self.decoupled = decoupled

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ssl_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            idx, images = self.loader.get_next_batch()
            images = torch.cat(images, dim=0)
            images = images.cuda()
            feats = self.network.forward(images)
            if self.decoupled:
                loss = utils.decoupled_contrastive_loss(features=feats, temp=0.1)
            else:
                logits, labels = utils.info_nce_loss(features=feats, temp=0.5)
                loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ssl_loss_total += loss.item()
            num_batches += 1
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    ssl_loss_total / num_batches, time.time() - start_time))

            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'ssl_loss_ep': ssl_loss_total / num_batches,
                        'ssl_loss_batch': loss.item(),
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'ssl_head': optimizer.param_groups[1]['lr'],
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            ssl_loss_total / num_batches, time.time() - start_time))


class SupContrModel:
    """Module implementing the Supervised Contrastive model"""

    def __init__(self, network, loader, logger):
        self.network = network
        self.loader = utils.ForeverDataLoader(loader)
        self.logger = logger
        self.criterion = utils.SupConLoss(temperature=0.1)
        self.network.cuda()

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ssl_loss_total = 0.0
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            idx, images, labels = self.loader.get_next_batch()
            images = torch.cat(images, dim=0)
            bsize = labels.shape[0]
            # labels = torch.cat([labels, labels], dim=0)
            images = images.cuda()
            labels = labels.cuda()
            # with torch.autograd.detect_anomaly():
            feats = self.network.forward(images)
            feats_norm = torch.nn.functional.normalize(feats)
            f1, f2 = torch.split(feats_norm, [bsize, bsize], dim=0)

            features_split = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # print(features_split.shape, labels.shape)
            loss = self.criterion(features=features_split, labels=labels)
            # print(loss, feats, labels)

            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ssl_loss_total += loss.item()
            num_batches += 1
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    ssl_loss_total / num_batches, time.time() - start_time))

            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ssl_loss_ep': ssl_loss_total / num_batches,
                    'ssl_loss_batch': loss.item(),
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'ssl_head': optimizer.param_groups[1]['lr'],
                },
                global_step=(ep - 1) * steps + num_batches,
            )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            ssl_loss_total / num_batches, time.time() - start_time))


class SupModelLM:
    """Module implementing the supervised pretraining on source"""

    def __init__(self, network, source_loader, logger, no_save=False):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.logger = logger
        self.no_save = no_save
        self.network.cuda()

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            src_logits = self.network.forward(src_images, src_labels)
            src_loss = src_criterion(src_logits, src_labels)

            src_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ce_loss_total += src_loss.item()
            num_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    ce_loss_total / num_batches, time.time() - start_time))
            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'ce_loss_ep': ce_loss_total / num_batches,
                        'ce_loss_batch': src_loss.item(),
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'fc': optimizer.param_groups[1]['lr'],
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            ce_loss_total / num_batches, time.time() - start_time))

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)

    def calc_val_loss(self, loader, loader_name, ep, steps, writer: tb.SummaryWriter, no_save):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        num_batches = 0
        total_loss = 0.0
        for _, (idxs, images, labels) in enumerate(loader):
            num_batches += 1
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                logits = self.network.forward(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
        self.logger.info("Validation Losses -- {:s}: {:.2f}".format(loader_name, total_loss / num_batches))
        if not no_save:
            writer.add_scalars(main_tag="val_loss",
                               tag_scalar_dict={
                                   loader_name: total_loss / num_batches,
                               },
                               global_step=ep * steps)


class SupModel:
    """Module implementing the supervised pretraining on source"""

    def __init__(self, network, source_loader, logger, no_save=False, linear_eval=False):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.logger = logger
        self.no_save = no_save
        self.linear_eval = linear_eval
        self.network.cuda()
        if self.linear_eval:
            self.network.freeze_linear_eval()
        else:
            self.network.freeze_train()

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter, mixup: bool = False):
        """Train model"""
        if self.linear_eval:
            self.network.set_linear_eval()
        else:
            self.network.set_train()

        num_batches = 0
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            if mixup:
                src_labels = func.one_hot(src_labels, num_classes=12)
                src_images, src_labels = utils.mixup(src_images, src_labels, np.random.beta(0.2, 0.2))
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            src_logits = self.network.forward(src_images)
            src_loss = src_criterion(src_logits, src_labels)

            src_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ce_loss_total += src_loss.item()
            num_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  CLR: {:.3f}  BLR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0,
                    ce_loss_total / num_batches, time.time() - start_time))
            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'ce_loss_ep': ce_loss_total / num_batches,
                        'ce_loss_batch': src_loss.item(),
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'fc': optimizer.param_groups[0]['lr'],
                        'bb': optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0,
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
        self.logger.info("Ep: {}/{}  Step: {}/{}  CLR: {:.3f}  BLR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'],
            optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0.0,
            ce_loss_total / num_batches, time.time() - start_time))

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)

    def calc_val_loss(self, loader, loader_name, ep, steps, writer: tb.SummaryWriter, no_save):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        num_batches = 0
        total_loss = 0.0
        for _, (idxs, images, labels) in enumerate(loader):
            num_batches += 1
            images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                logits = self.network.forward(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
        self.logger.info("Validation Losses -- {:s}: {:.2f}".format(loader_name, total_loss / num_batches))
        if not no_save:
            writer.add_scalars(main_tag="val_loss",
                               tag_scalar_dict={
                                   loader_name: total_loss / num_batches,
                               },
                               global_step=ep * steps)

    def get_val_loss_dict(self, loader, loader_name):
        """Calculate loss on provided dataloader"""
        start_time = time.time()
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses_dict = {}
        num_items = 0
        total_loss = 0.0
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
                loss = criterion(logits, labels)
                for i in range(loss.shape[0]):
                    total_loss += loss[i].item()
                    num_items += 1
                    idx = idxs[1][i].item()
                    losses_dict[idx] = loss[i].item()
        avg_loss = total_loss / num_items

        self.logger.info("Validation Loss -- {:s}: {:.2f}     T: {:.2f}s".format(loader_name, avg_loss,
                                                                                 time.time() - start_time))
        return losses_dict

    def get_eval_dicts(self, loader, loader_name):
        """Evaluate the model on the provided dataloader"""
        start_time = time.time()
        n_total = 0
        n_correct = 0
        labels_dict = {}
        preds_dict = {}
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            for i in range(pred.shape[0]):
                idx = idxs[1][i].item()
                labels_dict[idx] = labels[i].item()
                preds_dict[idx] = pred[i].item()
                n_total += 1
                n_correct += 1 if labels[i].item() == pred[i].item() else 0
        acc = 100. * n_correct / n_total
        self.logger.info("Accuracy -- {:s}: {:.2f}     T: {:.2f}s".format(loader_name, acc,
                                                                          time.time() - start_time))
        return labels_dict, preds_dict


class DualSupModel:
    """Module implementing the combined supervised pretraining on source and target"""

    def __init__(self, network, source_loader, target_loader, combined_batch, logger,
                 scramble_target_for_classification=False, scrambler_seed=None):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.combined_batch = combined_batch
        self.network.cuda()

        self.scramble_labels = scramble_target_for_classification
        self.scrambler_seed = scrambler_seed if scrambler_seed is not None else 42
        import scramble_labels
        self.scrambler = \
            scramble_labels.RandomToyboxScrambler(seed=self.scrambler_seed) if self.scramble_labels else None

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()

            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                logits = self.network.forward(comb_images)
                src_size = src_images.shape[0]
                src_logits = logits[:src_size]
                trgt_logits = logits[src_size:]
            else:
                src_logits = self.network.forward(src_images)
                trgt_logits = self.network.forward(trgt_images)

            if self.scrambler is not None:
                scrambled_trgt_labels = self.scrambler.scramble(trgt_labels)
            else:
                scrambled_trgt_labels = trgt_labels.clone()

            src_loss = criterion(src_logits, src_labels)
            trgt_loss = criterion(trgt_logits, scrambled_trgt_labels)
            total_loss = src_loss + trgt_loss

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            num_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  SCE: {:.3f}  TCE: {:.3f}  "
                                 "T: {:.2f}s"
                                 .format(ep, ep_total, step, steps,
                                         optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                         src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                         time.time() - start_time))

            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'src_ce_loss_ep': src_ce_loss_total / num_batches,
                    'src_ce_loss_batch': src_loss.item(),
                    'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                    'trgt_ce_loss_batch': trgt_loss.item(),
                    'total_loss_batch': total_loss.item(),
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'fc': optimizer.param_groups[1]['lr'],
                },
                global_step=(ep - 1) * steps + num_batches,
            )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  SCE: {:.3f}  TCE: {:.3f}  "
                         "T: {:.2f}s".format(ep, ep_total, steps, steps,
                                             optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                             src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                             time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for idx, loader in enumerate(loaders):
            num_batches = 0
            total_loss = 0.0
            for _, (idxs, images, labels) in enumerate(loader):
                num_batches += 1
                images, labels = images.cuda(), labels.cuda()
                if "in12" in loader_names[idx] and self.scrambler is not None and self.scramble_labels:
                    scrambled_labels = self.scrambler.scramble(labels)
                else:
                    scrambled_labels = labels.clone()
                with torch.no_grad():
                    logits = self.network.forward(images)
                loss = criterion(logits, scrambled_labels)
                total_loss += loss.item()
            losses.append(total_loss / num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
        writer.add_scalars(main_tag="val_loss",
                           tag_scalar_dict={
                               loader_names[0]: losses[0],
                               loader_names[1]: losses[1]
                           },
                           global_step=ep * steps)

    def eval(self, loader, scramble=False):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            if scramble and self.scrambler is not None:
                scrambled_labels = self.scrambler.scramble(labels)
            else:
                scrambled_labels = labels.clone()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=scrambled_labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)


class DualSupModelWith2Classifiers:
    """Module implementing the combined supervised pretraining on source and target datasets with
    two different classifiers"""

    def __init__(self, network, source_loader, target_loader, combined_batch, logger, no_save):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.combined_batch = combined_batch
        self.no_save = no_save
        self.network.cuda()

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()

            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                src_feats, trgt_feats, src_logits, trgt_logits = self.network.forward(
                    comb_images)
            else:
                src_feats, src_logits = self.network.forward_1(src_images)
                trgt_feats, trgt_logits = self.network.forward_2(trgt_images)

            src_loss = criterion(src_logits, src_labels)
            trgt_loss = criterion(trgt_logits, trgt_labels)
            total_loss = src_loss + trgt_loss

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            num_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  SCE: {:.3f}  TCE: {:.3f}  "
                                 "T: {:.2f}s"
                                 .format(ep, ep_total, step, steps,
                                         optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                         src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                         time.time() - start_time))

            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'src_ce_loss_ep': src_ce_loss_total / num_batches,
                        'src_ce_loss_batch': src_loss.item(),
                        'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                        'trgt_ce_loss_batch': trgt_loss.item(),
                        'total_loss_batch': total_loss.item(),
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'fc-1': optimizer.param_groups[1]['lr'],
                        'fc-2': optimizer.param_groups[2]['lr'],
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  SCE: {:.3f}  TCE: {:.3f}  "
                         "T: {:.2f}s".format(ep, ep_total, steps, steps,
                                             optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                             src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                             time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for idx, loader in enumerate(loaders):
            num_batches = 0
            total_loss = 0.0
            for _, (idxs, images, labels) in enumerate(loader):
                num_batches += 1
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    if "in12" in loader_names[idx]:
                        _, logits = self.network.forward_2(images)
                    else:
                        _, logits = self.network.forward_1(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
            losses.append(total_loss / num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
        if not self.no_save:
            writer.add_scalars(main_tag="val_loss",
                               tag_scalar_dict={
                                   loader_names[0]: losses[0],
                                   loader_names[1]: losses[1]
                               },
                               global_step=ep * steps)

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total_1 = 0
        n_correct_1 = 0
        n_total_2 = 0
        n_correct_2 = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                _, logits_1 = self.network.forward_1(images)
                _, logits_2 = self.network.forward_2(images)
            top_1, pred_1 = utils.calc_accuracy(output=logits_1, target=labels, topk=(1,))
            n_correct_1 += top_1[0].item() * pred_1.shape[0]
            n_total_1 += pred_1.shape[0]

            top_2, pred_2 = utils.calc_accuracy(output=logits_2, target=labels, topk=(1,))
            n_correct_2 += top_2[0].item() * pred_2.shape[0]
            n_total_2 += pred_2.shape[0]

        acc_1, acc_2 = n_correct_1 / n_total_1, n_correct_2 / n_total_2
        return round(acc_1, 2), round(acc_2, 2)


class DualSupModelWithDomain:
    """Module implementing the combined supervised pretraining on source and target and domain classification"""

    def __init__(self, network, source_loader, target_loader, logger, no_save=False):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.no_save = no_save
        self.network.cuda()

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        dom_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        dom_criterion = nn.BCEWithLogitsLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()
            dom_labels = torch.cat([torch.zeros(src_images.size(0)), torch.ones(trgt_images.size(0))])
            dom_labels = dom_labels.cuda()

            comb_images = torch.concat([src_images, trgt_images], dim=0)
            logits, dom_logits = self.network.forward(comb_images)
            src_size = src_images.shape[0]
            src_logits = logits[:src_size]
            trgt_logits = logits[src_size:]

            src_loss = criterion(src_logits, src_labels)
            trgt_loss = criterion(trgt_logits, trgt_labels)
            dom_loss = dom_criterion(dom_logits, dom_labels)
            total_loss = src_loss + trgt_loss + dom_loss

            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            dom_loss_total += dom_loss.item()
            num_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  "
                                 "SCE: {:.3f}  TCE: {:.3f}  D-BCE: {:.3f}  "
                                 "T: {:.2f}s"
                                 .format(ep, ep_total, step, steps,
                                         optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                         src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                         dom_loss_total / num_batches,
                                         time.time() - start_time))
            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'src_ce_loss_ep': src_ce_loss_total / num_batches,
                        'src_ce_loss_batch': src_loss.item(),
                        'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                        'trgt_ce_loss_batch': trgt_loss.item(),
                        'dom_loss_ep': dom_loss_total / num_batches,
                        'dom_loss_batch': dom_loss.item(),
                        'total_loss_batch': total_loss.item(),
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'fc': optimizer.param_groups[1]['lr'],
                        'dom_fc': optimizer.param_groups[2]['lr']
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  "
                         "SCE: {:.3f}  TCE: {:.3f}  D-BCE: {:.3f}  "
                         "T: {:.2f}s".format(ep, ep_total, steps, steps,
                                             optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                             src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                             dom_loss_total / num_batches,
                                             time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for idx, loader in enumerate(loaders):
            num_batches = 0
            total_loss = 0.0
            for _, (idxs, images, labels) in enumerate(loader):
                num_batches += 1
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    logits, _ = self.network.forward(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
            losses.append(total_loss / num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
        if not self.no_save:
            writer.add_scalars(main_tag="val_loss",
                               tag_scalar_dict={
                                   loader_names[0]: losses[0],
                                   loader_names[1]: losses[1]
                               },
                               global_step=ep * steps)

    def eval(self, loader, source):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        dom_n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits, dom_logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]

            dom_logits_sigmoid = torch.sigmoid(dom_logits)
            dom_preds = torch.where(dom_logits_sigmoid > 0.5, 1, 0)
            n_ones = torch.sum(dom_preds)
            n_zeros = pred.shape[0] - n_ones
            if source:
                dom_n_correct += n_zeros.cpu().numpy()
            else:
                dom_n_correct += n_ones.cpu().numpy()

        acc = n_correct / n_total
        dom_acc = 100. * dom_n_correct / n_total
        return round(acc, 2), round(dom_acc, 2)


class DualSupWithCCMMDModel:
    """Module implementing the Dual Supervised Model with CCMMD loss"""

    def __init__(self, network, source_loader, target_loader, combined_batch, logger, scramble_labels=False,
                 scrambler_seed=None, scramble_target_for_classification=False, lmbda=0.05, no_save=False):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.lmbda = lmbda
        self.ccmmd_loss = ccmmd.ClassConditionalMMDLoss(
            kernels=([ccmmd.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        ).cuda()
        self.ccmmd_loss.train()
        self.logger = logger
        self.combined_batch = combined_batch
        self.scramble_labels = scramble_labels
        self.scrambler_seed = scrambler_seed if scrambler_seed is not None else 42
        import scramble_labels
        self.scrambler = \
            scramble_labels.RandomToyboxScrambler(seed=self.scrambler_seed) if self.scramble_labels else None
        self.scramble_target_for_classification = scramble_target_for_classification
        self.network.cuda()
        self.no_save = no_save

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        total_batches = (ep - 1) * steps
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        ccmmd_loss_total = 0.0
        comb_loss_total = 0.0
        ce_criterion = nn.CrossEntropyLoss()

        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            p = total_batches / (steps * ep_total)
            alfa = 2 / (1 + math.exp(-10 * p)) - 1

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()

            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()

            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                feats, logits = self.network.forward(comb_images)
                src_size = src_images.shape[0]
                src_feats, src_logits = feats[:src_size], logits[:src_size]
                trgt_feats, trgt_logits = feats[src_size:], logits[src_size:]
            else:
                src_feats, src_logits = self.network.forward(src_images)
                trgt_feats, trgt_logits = self.network.forward(trgt_images)

            src_loss = ce_criterion(src_logits, src_labels)
            if self.scrambler is not None:
                scrambled_trgt_labels = self.scrambler.scramble(labels=trgt_labels)
                if self.scramble_target_for_classification:
                    scrambled_trgt_labels_cl = scrambled_trgt_labels.clone()
                else:
                    scrambled_trgt_labels_cl = trgt_labels.clone()
            else:
                scrambled_trgt_labels = trgt_labels.clone()
                scrambled_trgt_labels_cl = trgt_labels.clone()

            trgt_loss = ce_criterion(trgt_logits, scrambled_trgt_labels_cl)
            ccmmd_loss = self.ccmmd_loss(z_s=src_feats, z_t=trgt_feats, l_s=src_labels, l_t=scrambled_trgt_labels)
            total_loss = src_loss + trgt_loss + self.lmbda * alfa * ccmmd_loss

            total_loss.backward()
            optimizer.step()

            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            ccmmd_loss_total += ccmmd_loss.item()
            comb_loss_total += total_loss.item()
            num_batches += 1
            total_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SCE: {:.4f}  TCE: {:.4f}  "
                                 "CCMMD: {:.4f}  Tot: {:.4f}  T: {:.0f}s".format(
                    ep, ep_total, step, steps,
                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                    ccmmd_loss_total / num_batches,
                    comb_loss_total / num_batches, time.time() - start_time))
            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'src_ce_loss_ep': src_ce_loss_total / num_batches,
                        'src_ce_loss_batch': src_loss.item(),
                        'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                        'trgt_ce_loss_batch': trgt_loss.item(),
                        'ccmmd_loss_ep': ccmmd_loss_total / num_batches,
                        'ccmmd_loss_batch': ccmmd_loss.item(),
                        'comb_loss_ep': comb_loss_total / num_batches,
                        'comb_loss_batch': total_loss.item(),
                    },
                    global_step=total_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'fc': optimizer.param_groups[1]['lr'],
                    },
                    global_step=total_batches,
                )
                writer.add_scalars(
                    main_tag="training",
                    tag_scalar_dict={
                        'p': p,
                        'lambda': alfa,
                    },
                    global_step=total_batches,
                )
            if scheduler is not None:
                scheduler.step()

        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SCE: {:.4f}  TCE: {:.4f}  CCMMD: {:.4f}  "
                         "Tot: {:.4f}  T: {:.0f}s".format(
            ep, ep_total, steps, steps,
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
            ccmmd_loss_total / num_batches, comb_loss_total / num_batches,
            time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for idx, loader in enumerate(loaders):
            num_batches = 0
            total_loss = 0.0
            for _, (idxs, images, labels) in enumerate(loader):
                num_batches += 1
                images, labels = images.cuda(), labels.cuda()
                if "in12" in loader_names[idx] and self.scrambler is not None \
                        and self.scramble_target_for_classification:
                    scrambled_labels = self.scrambler.scramble(labels)
                else:
                    scrambled_labels = labels.clone()
                with torch.no_grad():
                    _, logits = self.network.forward(images)
                loss = criterion(logits, scrambled_labels)
                total_loss += loss.item()
            losses.append(total_loss / num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
        if not self.no_save:
            writer.add_scalars(main_tag="val_loss",
                               tag_scalar_dict={
                                   loader_names[0]: losses[0],
                                   loader_names[1]: losses[1]
                               },
                               global_step=ep * steps)

    def eval(self, loader, scramble=False):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            if scramble and self.scrambler is not None:
                scrambled_labels = self.scrambler.scramble(labels)
            else:
                scrambled_labels = labels.clone()
            with torch.no_grad():
                _, logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=scrambled_labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)


class TBSupWithCCMMDModel:
    """Module implementing the Supervised Model on Toybox with CCMMD loss"""

    def __init__(self, network, source_loader, target_loader, combined_batch, logger, lmbda=0.05):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.lmbda = lmbda
        self.ccmmd_loss = ccmmd.ClassConditionalMMDLoss(
            kernels=([ccmmd.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        ).cuda()
        self.ccmmd_loss.train()
        self.logger = logger
        self.combined_batch = combined_batch
        self.network.cuda()

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        total_batches = (ep - 1) * steps
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        ccmmd_loss_total = 0.0
        comb_loss_total = 0.0
        ce_criterion = nn.CrossEntropyLoss()

        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            p = total_batches / (steps * ep_total)
            alfa = 2 / (1 + math.exp(-10 * p)) - 1

            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()

            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()

            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                feats, logits = self.network.forward(comb_images)
                src_size = src_images.shape[0]
                src_feats, src_logits = feats[:src_size], logits[:src_size]
                trgt_feats, trgt_logits = feats[src_size:], logits[src_size:]
            else:
                src_feats, src_logits = self.network.forward(src_images)
                trgt_feats, trgt_logits = self.network.forward(trgt_images)

            src_loss = ce_criterion(src_logits, src_labels)
            trgt_loss = ce_criterion(trgt_logits, trgt_labels)
            ccmmd_loss = self.ccmmd_loss(z_s=src_feats, z_t=trgt_feats, l_s=src_labels, l_t=trgt_labels)
            total_loss = src_loss + self.lmbda * alfa * ccmmd_loss

            total_loss.backward()
            optimizer.step()

            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            ccmmd_loss_total += ccmmd_loss.item()
            comb_loss_total += total_loss.item()
            num_batches += 1
            total_batches += 1

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SCE: {:.4f}  TCE: {:.4f}  "
                                 "CCMMD: {:.4f}  Tot: {:.4f}  T: {:.0f}s".format(
                    ep, ep_total, step, steps,
                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                    ccmmd_loss_total / num_batches,
                    comb_loss_total / num_batches, time.time() - start_time))

            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'src_ce_loss_ep': src_ce_loss_total / num_batches,
                    'src_ce_loss_batch': src_loss.item(),
                    'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                    'trgt_ce_loss_batch': trgt_loss.item(),
                    'ccmmd_loss_ep': ccmmd_loss_total / num_batches,
                    'ccmmd_loss_batch': ccmmd_loss.item(),
                    'comb_loss_ep': comb_loss_total / num_batches,
                    'comb_loss_batch': total_loss.item(),
                },
                global_step=total_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'fc': optimizer.param_groups[1]['lr'],
                },
                global_step=total_batches,
            )
            writer.add_scalars(
                main_tag="training",
                tag_scalar_dict={
                    'p': p,
                    'lambda': alfa,
                },
                global_step=total_batches,
            )
            if scheduler is not None:
                scheduler.step()

        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SCE: {:.4f}  TCE: {:.4f}  CCMMD: {:.4f}  "
                         "Tot: {:.4f}  T: {:.0f}s".format(
            ep, ep_total, steps, steps,
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
            ccmmd_loss_total / num_batches, comb_loss_total / num_batches,
            time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for idx, loader in enumerate(loaders):
            num_batches = 0
            total_loss = 0.0
            for _, (idxs, images, labels) in enumerate(loader):
                num_batches += 1
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    _, logits = self.network.forward(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
            losses.append(total_loss / num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
        writer.add_scalars(main_tag="val_loss",
                           tag_scalar_dict={
                               loader_names[0]: losses[0],
                               loader_names[1]: losses[1]
                           },
                           global_step=ep * steps)

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                _, logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)
