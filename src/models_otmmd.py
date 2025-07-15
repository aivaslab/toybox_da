
import time
import torch.nn as nn
import torch.nn.functional as func
import torch
import math
import torch.utils.tensorboard as tb

import utils
import mmd_util


class OTMMDModel:
    """Module implementing the JAN architecture"""

    def __init__(self, network, loader, combined_batch, logger, asymmetric, no_save=False, mmd_w=0.1):
        self.network = network
        self.loader = utils.ForeverDataLoaderOTMMD(loader)
        self.jmmd_loss = mmd_util.JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([mmd_util.GaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                     [mmd_util.GaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                     [mmd_util.GaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                     ),
            linear=False,
        ).cuda()
        self.jmmd_loss.train()
        self.logger = logger
        self.combined_batch = combined_batch
        self.no_save = no_save
        self.asymmetric = asymmetric
        self.mmd_w = mmd_w
        self.network.cuda()

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        total_batches = (ep - 1) * steps
        ce_loss_total = 0.0
        jmmd_loss_total = 0.0
        comb_loss_total = 0.0
        d_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        alfa = self.mmd_w * (2 / (1 + math.exp(-10 * 0.0)) - 1)
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            p = total_batches / (steps * ep_total)
            alfa = self.mmd_w * (2 / (1 + math.exp(-10 * p)) - 1)

            idxs, (src_images, trgt_images), labels = self.loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), labels.cuda()
            trgt_images, trgt_labels = trgt_images.cuda(), labels.cuda()

            comb_images = torch.concat([src_images, trgt_images], dim=0)
            bb_feats, btl_feats, logits = self.network.forward(comb_images)
            src_size = src_images.shape[0]
            src_bb_feats, src_btl_feats, src_logits = bb_feats[:src_size], btl_feats[:src_size], logits[:src_size]
            trgt_bb_feats, trgt_btl_feats, trgt_logits = \
                bb_feats[src_size:], btl_feats[src_size:], logits[src_size:]

            src_loss = src_criterion(src_logits, src_labels)
            jmmd_loss = torch.tensor([0.0]).cuda()
            for cl in range(12):
                src_bb_feats_cl, src_btl_feats_cl, src_logits_cl = src_bb_feats[labels == cl], \
                    src_btl_feats[labels == cl], src_logits[labels == cl]
                trgt_bb_feats_cl, trgt_btl_feats_cl, trgt_logits_cl = trgt_bb_feats[labels == cl], \
                    trgt_btl_feats[labels == cl], trgt_logits[labels == cl]
                if self.asymmetric:
                    src_bb_feats_cl, src_btl_feats_cl, src_logits_cl = \
                        src_bb_feats_cl.clone().detach(), src_btl_feats_cl.clone().detach(), \
                        src_logits_cl.clone().detach()
                if src_logits_cl.shape[0] > 1:
                    loss = self.jmmd_loss(
                        (src_bb_feats_cl, src_btl_feats_cl, func.softmax(src_logits_cl, dim=1)),
                        (trgt_bb_feats_cl, trgt_btl_feats_cl, func.softmax(trgt_logits_cl, dim=1)))
                    jmmd_loss += loss / 12
            total_loss = src_loss + alfa * jmmd_loss
            total_loss.backward()
            optimizer.step()

            ce_loss_total += src_loss.item()
            jmmd_loss_total += jmmd_loss.item()
            comb_loss_total += total_loss.item()
            num_batches += 1
            total_batches += 1

            with torch.no_grad():
                dist = ((src_bb_feats - trgt_bb_feats) ** 2).sum(-1)
                dist = torch.mean(dist**0.5)
                d_total += dist.item()

            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE: {:.4f}  JMMD: {:.4f}  "
                                 "Tot: {:.4f}  alfa: {:.2f}  d: {:.2f}  T: {:.2f}s".format(
                                    ep, ep_total, step, steps,
                                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                    ce_loss_total / num_batches, jmmd_loss_total / num_batches,
                                    comb_loss_total / num_batches, alfa, d_total / num_batches,
                                    time.time() - start_time))
            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'ce_loss_ep': ce_loss_total / num_batches,
                        'ce_loss_batch': src_loss.item(),
                        'jmmd_loss_ep': jmmd_loss_total / num_batches,
                        'jmmd_loss_batch': jmmd_loss.item(),
                        'comb_loss_ep': comb_loss_total / num_batches,
                        'comb_loss_batch': total_loss.item(),
                    },
                    global_step=total_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'bottleneck': optimizer.param_groups[1]['lr'],
                        'fc': optimizer.param_groups[2]['lr'],
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

        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE: {:.4f}  JMMD: {:.4f}  "
                         "Tot: {:.4f}  alfa: {:.2f}  d: {:.2f}  T: {:.2f}s".format(
                            ep, ep_total, steps, steps,
                            optimizer.param_groups[0]['lr'], optimizer.param_groups[2]['lr'],
                            ce_loss_total / num_batches, jmmd_loss_total / num_batches, comb_loss_total / num_batches,
                            alfa, d_total / num_batches, time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        start_time = time.time()
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
                    _, _, logits = self.network.forward(images)
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
            losses.append(total_loss / num_batches)

        if not self.no_save:
            writer.add_scalars(main_tag="val_loss",
                               tag_scalar_dict={
                                   loader_names[0]: losses[0],
                                   loader_names[1]: losses[1]
                               },
                               global_step=ep * steps)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}      T: {:.2f}s".
                         format(loader_names[0], losses[0], loader_names[1], losses[1], time.time() - start_time))
        return losses

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                _, _, logits = self.network.forward(images)
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
                _, _, logits = self.network.forward(images)
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
                _, _, logits = self.network.forward(images)
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
