
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.tensorboard as tb

import utils
import mmd_util


class DualSSLOrientedModelV1:
    """Module implementing the SSL method for pretraining the DA model with both TB and IN-12 data"""

    def __init__(self, network, src_loader, trgt_loader, logger, no_save, tb_ssl_loss, in12_ssl_loss,
                 tb_alpha, in12_alpha, use_v2, orient_alpha, num_images_in_batch, ignore_orient_loss, use_cosine):
        self.network = network
        self.src_loader = utils.ForeverDataLoader(src_loader)
        self.trgt_loader = utils.ForeverDataLoader(trgt_loader)
        self.logger = logger
        self.network.cuda()
        self.network.freeze_train()
        self.no_save = no_save
        self.tb_ssl_loss = tb_ssl_loss
        self.in12_ssl_loss = in12_ssl_loss
        self.tb_alpha = tb_alpha
        self.in12_alpha = in12_alpha
        self.use_v2 = use_v2
        self.orient_alpha = orient_alpha
        self.num_images_in_batch = num_images_in_batch
        self.ignore_orient_loss = ignore_orient_loss
        self.use_cosine = use_cosine

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        if self.use_v2:
            self.orientation_loss = utils.OrientationLossV2(use_cosine=self.use_cosine)
        else:
            self.orientation_loss = utils.OrientationLossV1(num_images_in_batch=self.num_images_in_batch)
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_loss_total = 0.0
        trgt_loss_total = 0.0
        ssl_loss_total = 0.0
        orient_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.src_loader.get_next_batch()
            trgt_idx, trgt_images, trgt_labels = self.trgt_loader.get_next_batch()
            assert len(src_images) == 4 and len(trgt_images) == 2

            src_anchors, src_positives = torch.cat([src_images[0], src_images[2]], dim=0), \
                torch.cat([src_images[1], src_images[3]], dim=0)
            src_batch = torch.cat([src_anchors, src_positives], dim=0)
            trgt_anchors, trgt_positives = trgt_images[0], trgt_images[1]
            trgt_batch = torch.cat([trgt_anchors, trgt_positives], dim=0)

            images = torch.cat([src_batch, trgt_batch], dim=0)
            images = images.cuda()
            feats = self.network.forward(images)
            src_size = src_batch.shape[0]
            src_feats, trgt_feats = feats[:src_size], feats[src_size:]
            # print(src_feats.shape, trgt_feats.shape)
            if self.tb_ssl_loss == "simclr":
                src_logits, src_labels = utils.info_nce_loss(features=src_feats, temp=0.5)
                src_loss = criterion(src_logits, src_labels)
            elif self.tb_ssl_loss == "dcl":
                src_loss = utils.decoupled_contrastive_loss(features=src_feats, temp=0.1)
            else:
                concat_src_labels = torch.cat([src_labels for _ in range(4)], dim=0)
                # print(src_labels.shape, concat_src_labels.shape)
                src_loss = utils.sup_decoupled_contrastive_loss(features=src_feats, temp=0.1, labels=concat_src_labels)

            if self.in12_ssl_loss == "dcl":
                trgt_loss = utils.decoupled_contrastive_loss(features=trgt_feats, temp=0.1)
            else:

                trgt_logits, trgt_labels = utils.info_nce_loss(features=trgt_feats, temp=0.5)
                trgt_loss = criterion(trgt_logits, trgt_labels)

            num_images_src = len(src_images[0])
            src_feats_1, src_feats_2, src_feats_3, src_feats_4 = src_feats[:num_images_src], \
                src_feats[num_images_src:2 * num_images_src], src_feats[2 * num_images_src:3 * num_images_src], \
                src_feats[3 * num_images_src:4 * num_images_src]

            src_feats_stacked = torch.stack(
                tensors=(src_feats_1, src_feats_2, src_feats_3, src_feats_4),
                dim=1).view(4*num_images_src, -1)

            num_images_trgt = len(trgt_images[0])
            trgt_feats_1, trgt_feats_2 = trgt_feats[:num_images_trgt], \
                trgt_feats[num_images_trgt:2*num_images_trgt]
            trgt_feats_stacked = torch.stack(
                tensors=(trgt_feats_1, trgt_feats_2),
                dim=1).view(2*num_images_trgt, -1)
            if self.use_v2:
                orientation_loss = self.orientation_loss(src_feats=src_feats_stacked,
                                                         trgt_feats=trgt_feats_stacked,
                                                         src_labels=src_labels)
            else:
                orientation_loss = self.orientation_loss(src_feats=src_feats_stacked, trgt_feats=trgt_feats_stacked)

            if self.ignore_orient_loss:
                loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss
            else:
                loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss + self.orient_alpha * orientation_loss
            src_loss_total += src_loss.item()
            trgt_loss_total += trgt_loss.item()
            orient_loss_total += orientation_loss.item()
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ssl_loss_total += loss.item()
            num_batches += 1
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
                                 "OrL: {:.3f}  Loss: {:.3f}  T: {:.2f}s".format(
                                    ep, ep_total, step, steps,
                                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                    src_loss_total / num_batches, trgt_loss_total / num_batches,
                                    orient_loss_total / num_batches,
                                    ssl_loss_total / num_batches, time.time() - start_time)
                                 )

            if not self.no_save:
                scalar_dict = {
                        'src_ssl_loss_ep': src_loss_total / num_batches,
                        'src_ssl_loss_batch': src_loss.item(),
                        'trgt_ssl_loss_ep': trgt_loss_total / num_batches,
                        'trgt_ssl_loss_batch': trgt_loss.item(),
                        'orient_loss_ep': orient_loss_total / num_batches,
                        'orient_loss_batch': orientation_loss.item(),
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
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
                         "OrL: {:.3f}  Loss: {:.3f}  T: {:.2f}s".format(
                            ep, ep_total, steps, steps,
                            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                            src_loss_total / num_batches, trgt_loss_total / num_batches,
                            orient_loss_total / num_batches,
                            ssl_loss_total / num_batches, time.time() - start_time)
                         )


class DualSSLClassMMDModelV1:
    """Module implementing the SSL method for pretraining the DA model with both TB and IN-12 data"""

    def __init__(self, network, src_loader, trgt_loader, logger, no_save, tb_ssl_loss, in12_ssl_loss,
                 tb_alpha, in12_alpha, div_alpha, ignore_div_loss, asymmetric, use_ot, div_metric,
                 fixed_div_alpha, use_div_on_feats, combined_fwd_pass, queue_size, track_knn_acc):
        self.network = network
        self.src_loader = utils.ForeverDataLoader(src_loader)
        self.trgt_loader = utils.ForeverDataLoader(trgt_loader)
        self.logger = logger
        self.network.cuda()
        self.network.freeze_train()
        self.no_save = no_save
        self.tb_ssl_loss = tb_ssl_loss
        self.in12_ssl_loss = in12_ssl_loss
        self.tb_alpha = tb_alpha
        self.in12_alpha = in12_alpha
        self.div_alpha = div_alpha
        self.ignore_div_loss = ignore_div_loss
        self.asymmetric = asymmetric
        self.use_ot = use_ot
        self.div_metric = div_metric
        self.fixed_div_alpha = fixed_div_alpha
        self.use_div_on_feats = use_div_on_feats
        self.combined_fwd_pass = combined_fwd_pass
        self.queue_size = queue_size
        self.tb_feats_queue = None
        self.tb_labels_queue = None
        self.in12_feats_queue = None
        self.in12_labels_queue = None
        self.track_knn_acc = track_knn_acc

        self.emd_dist_loss = mmd_util.EMD1DLoss()
        self.mmd_dist_loss = mmd_util.JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([mmd_util.GaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                     ),
            linear=False,
        ).cuda()

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")
        torch.autograd.set_detect_anomaly(True)

    @staticmethod
    def get_paired_distance(src_feats, trgt_feats, metric):
        """Returns the distance matrix between the source features and the target features"""
        if metric == "cosine":
            dist = func.cosine_similarity(src_feats.unsqueeze(1), trgt_feats.unsqueeze(0), dim=-1)
        elif metric == "euclidean":
            dist = torch.linalg.norm(src_feats.unsqueeze(1) - trgt_feats.unsqueeze(0), dim=-1, ord=2)
        else:
            dist = torch.matmul(src_feats, trgt_feats.transpose(0, 1))
        assert dist.shape == (src_feats.shape[0], trgt_feats.shape[0]), f"{dist.shape}"
        return dist

    def get_distance(self, feats):
        if self.div_metric == "cosine":
            dist = func.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=-1)
        elif self.div_metric == "euclidean":
            dist = torch.linalg.norm(feats.unsqueeze(1) - feats.unsqueeze(0), dim=-1, ord=2)
        else:
            dist = torch.matmul(feats, feats.transpose(0, 1))
        assert dist.shape == (feats.shape[0], feats.shape[0]), f"{dist.shape}"
        return dist

    def get_div_alpha(self, step, steps, ep, ep_total):
        if self.ignore_div_loss:
            return 0.0
        if self.fixed_div_alpha:
            return self.div_alpha
        total_steps = steps * ep_total
        curr_step = steps * (ep - 1) + step
        frac = 1 - 0.5 * (1 + np.cos(curr_step * np.pi / total_steps))
        return self.div_alpha * frac

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""

        self.network.set_train()
        num_batches = 0
        src_loss_total = 0.0
        trgt_loss_total = 0.0
        ssl_loss_total = 0.0
        div_loss_mmd_total = 0.0
        div_loss_emd_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        src_acc_total = 0.0
        src_neg_acc_total = 0.0
        trgt_acc_total = 0.0
        trgt_neg_acc_total = 0.0
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.src_loader.get_next_batch()
            trgt_idx, trgt_images, trgt_labels = self.trgt_loader.get_next_batch()
            assert len(src_images) == 4 and len(trgt_images) == 2

            src_anchors, src_positives = torch.cat([src_images[0], src_images[2]], dim=0), \
                torch.cat([src_images[1], src_images[3]], dim=0)
            src_batch = torch.cat([src_anchors, src_positives], dim=0)
            trgt_anchors, trgt_positives = trgt_images[0], trgt_images[1]
            trgt_batch = torch.cat([trgt_anchors, trgt_positives], dim=0)
            src_labels, trgt_labels = src_labels.cuda(), trgt_labels.cuda()
            src_size, trgt_size = src_batch.shape[0], trgt_batch.shape[0]

            if self.combined_fwd_pass:
                images = torch.cat([src_batch, trgt_batch], dim=0)
                images = images.cuda()
                feats = self.network.forward(images)
                src_feats, trgt_feats = feats[:src_size], feats[src_size:]
            else:
                src_batch, trgt_batch = src_batch.cuda(), trgt_batch.cuda()
                src_feats = self.network.forward(src_batch)
                trgt_feats = self.network.forward(trgt_batch)

            src_anchor_feats = src_feats[:src_size//2]
            trgt_anchor_feats = trgt_feats[:trgt_size//2]
            # print(src_feats.shape, trgt_feats.shape)
            if self.tb_ssl_loss == "simclr":
                src_logits, src_labels_info_nce = utils.info_nce_loss(features=src_feats, temp=0.5)
                src_loss = criterion(src_logits, src_labels_info_nce)
            elif self.tb_ssl_loss == "dcl":
                src_loss = utils.decoupled_contrastive_loss(features=src_feats, temp=0.1)
            else:
                concat_src_labels = torch.cat([src_labels for _ in range(4)], dim=0)
                # print(src_labels.shape, concat_src_labels.shape)
                src_loss = utils.sup_decoupled_contrastive_loss(features=src_feats, temp=0.1, labels=concat_src_labels)

            if self.in12_ssl_loss == "dcl":
                trgt_loss = utils.decoupled_contrastive_loss(features=trgt_feats, temp=0.1)
            else:
                trgt_logits, trgt_labels = utils.info_nce_loss(features=trgt_feats, temp=0.5)
                trgt_loss = criterion(trgt_logits, trgt_labels)

            num_images_src = len(src_images[0])
            src_feats_1, src_feats_2 = src_feats[:num_images_src], src_feats[num_images_src:2 * num_images_src],
            src_feats_stacked = torch.stack(tensors=(src_feats_1, src_feats_2), dim=1).view(2 * num_images_src, -1)

            num_images_trgt = len(trgt_images[0])
            trgt_feats_1 = trgt_feats[:num_images_trgt]

            div_alpha = self.get_div_alpha(steps=steps, step=step, ep=ep, ep_total=ep_total)
            if self.ignore_div_loss:
                loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss
                with torch.no_grad():
                    if self.use_div_on_feats:
                        div_dist_emd_loss = torch.tensor([0.0])
                        div_dist_mmd_loss = self.mmd_dist_loss((src_feats_stacked,), (trgt_feats_1,))
                    else:
                        src_feats_dists = self.get_distance(src_feats_stacked)
                        if self.asymmetric:
                            src_feats_dists = torch.reshape(src_feats_dists, (-1, 1)).clone().detach()

                        else:
                            src_feats_dists = torch.reshape(src_feats_dists, (-1, 1))

                        trgt_feats_dists = self.get_distance(trgt_feats_1)
                        trgt_feats_dists = torch.reshape(trgt_feats_dists, (-1, 1))

                        div_dist_emd_loss = self.emd_dist_loss(src_feats_dists, trgt_feats_dists)
                        div_dist_mmd_loss = self.mmd_dist_loss((src_feats_dists,), (trgt_feats_dists,))
            else:
                if self.use_div_on_feats:
                    if self.asymmetric:
                        src_feats_dist = src_feats_stacked.clone().detach()
                    else:
                        src_feats_dist = src_feats_stacked
                    trgt_feats_dist = trgt_feats_1
                    if self.use_ot:
                        raise NotImplementedError()
                    else:
                        div_dist_emd_loss = torch.tensor([0.])
                        div_dist_mmd_loss = self.mmd_dist_loss((src_feats_dist,), (trgt_feats_dist,))
                        loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss + div_alpha * div_dist_mmd_loss
                else:
                    src_feats_dists = self.get_distance(src_feats_stacked)
                    if self.asymmetric:
                        src_feats_dists = torch.reshape(src_feats_dists, (-1, 1)).clone().detach()
                    else:
                        src_feats_dists = torch.reshape(src_feats_dists, (-1, 1))

                    trgt_feats_dists = self.get_distance(trgt_feats_1)
                    trgt_feats_dists = torch.reshape(trgt_feats_dists, (-1, 1))
                    if self.use_ot:
                        div_dist_emd_loss = self.emd_dist_loss(src_feats_dists, trgt_feats_dists)
                        div_dist_mmd_loss = torch.tensor([0.])
                        loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss + div_alpha * div_dist_emd_loss
                    else:
                        div_dist_emd_loss = torch.tensor([0.])
                        div_dist_mmd_loss = self.mmd_dist_loss((src_feats_dists, ), (trgt_feats_dists, ))
                        loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss + div_alpha * div_dist_mmd_loss

            src_loss_total += src_loss.item()
            trgt_loss_total += trgt_loss.item()
            div_loss_emd_total += div_dist_emd_loss.item()
            div_loss_mmd_total += div_dist_mmd_loss.item()
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ssl_loss_total += loss.item()
            num_batches += 1

            if self.track_knn_acc:
                # Track within-batch accuracy using KNNs
                # NOTE: This only considers the anchor images. Positive pairs are not considered for KNN acc.
                # calculation
                with torch.no_grad():
                    dupl_src_labels = torch.cat([src_labels for _ in range(2)], dim=0)
                    if self.tb_feats_queue is None:
                        self.tb_feats_queue = src_anchor_feats
                        self.tb_labels_queue = dupl_src_labels
                        self.in12_feats_queue = trgt_anchor_feats
                        self.in12_labels_queue = trgt_labels
                    else:
                        self.tb_feats_queue = torch.cat((self.tb_feats_queue, src_anchor_feats))[-self.queue_size:, :]
                        self.tb_labels_queue = torch.cat((self.tb_labels_queue, dupl_src_labels))[-self.queue_size:]
                        self.in12_feats_queue = torch.cat((self.in12_feats_queue, trgt_anchor_feats))[
                                                -self.queue_size:, :]
                        self.in12_labels_queue = torch.cat((self.in12_labels_queue, trgt_labels))[-self.queue_size:]
                    # print(self.tb_feats_queue.shape, self.in12_feats_queue.shape, self.tb_labels_queue.shape,
                    #       self.in12_labels_queue.shape)

                    # src_dist_mat = self.get_distance(feats=src_anchor_feats, metric="cosine")
                    src_dist_mat = self.get_paired_distance(src_feats=src_anchor_feats,
                                                            trgt_feats=self.tb_feats_queue,
                                                            metric="cosine")
                    # print(src_dist_mat.shape)
                    # Get top-2 closest image and discard closest (this should be the image itself) for src feats
                    _, src_topk_closest_indices = torch.topk(src_dist_mat, k=2, largest=True)
                    src_topk_labels = self.tb_labels_queue[src_topk_closest_indices][:, 1]

                    # Get the furthest image for src feats
                    _, src_topk_farthest_indices = torch.topk(src_dist_mat, k=1, largest=False)
                    src_farthest_labels = self.tb_labels_queue[src_topk_farthest_indices[:, 0]]

                    # Calculate src accuracies
                    src_acc = 100 * torch.sum((src_topk_labels == dupl_src_labels).int()).item() / len(dupl_src_labels)
                    src_neg_acc = 100 * torch.sum((src_farthest_labels == dupl_src_labels).int()).item() / len(
                        dupl_src_labels)

                    # trgt_dist_mat = self.get_distance(feats=trgt_anchor_feats, metric="cosine")
                    trgt_dist_mat = self.get_paired_distance(src_feats=trgt_anchor_feats,
                                                             trgt_feats=self.in12_feats_queue,
                                                             metric="cosine")
                    # Get top-2 closes image and discard closest (this should be the image itself) for trgt feats
                    _, trgt_topk_closest_indices = torch.topk(trgt_dist_mat, k=2, largest=True)
                    trgt_topk_labels = self.in12_labels_queue[trgt_topk_closest_indices][:, 1]

                    # Get the furthest image for trgt feats
                    _, trgt_topk_farthest_indices = torch.topk(trgt_dist_mat, k=1, largest=False)
                    trgt_farthest_labels = self.in12_labels_queue[trgt_topk_farthest_indices][:, 0]

                    # Calculate trgt accuracies
                    trgt_acc = 100 * torch.sum((trgt_topk_labels == trgt_labels).int()).item() / len(trgt_labels)
                    trgt_neg_acc = 100 * torch.sum((trgt_farthest_labels == trgt_labels).int()).item() / len(
                        trgt_labels)

                    src_acc_total += src_acc
                    src_neg_acc_total += src_neg_acc
                    trgt_acc_total += trgt_acc
                    trgt_neg_acc_total += trgt_neg_acc
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
                                 "Div-A: {:3.2e}  EMD: {:.3f}  MMD: {:.3f}  Loss: {:.3f}  A1: {:.2f}  A2: {:.2f}  "
                                 "A3: {:.2f}  A4: {:.2f}  T: {:.2f}s".format(
                                    ep, ep_total, step, steps,
                                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                    src_loss_total / num_batches, trgt_loss_total / num_batches,
                                    div_alpha, div_loss_emd_total / num_batches, div_loss_mmd_total / num_batches,
                                    ssl_loss_total / num_batches,
                                    src_acc_total / num_batches, trgt_acc_total / num_batches,
                                    src_neg_acc_total / num_batches, trgt_neg_acc_total / num_batches,
                                    time.time() - start_time)
                                 )

            if not self.no_save:
                scalar_dict = {
                        'src_ssl_loss_ep': src_loss_total / num_batches,
                        'src_ssl_loss_batch': src_loss.item(),
                        'trgt_ssl_loss_ep': trgt_loss_total / num_batches,
                        'trgt_ssl_loss_batch': trgt_loss.item(),
                        'div_loss_emd_ep': div_loss_emd_total / num_batches,
                        'div_loss_emd_batch': div_dist_emd_loss.item(),
                        'div_loss_mmd_ep': div_loss_mmd_total / num_batches,
                        'div_loss_mmd_batch': div_dist_mmd_loss.item(),
                        'ssl_loss_ep': ssl_loss_total / num_batches,
                        'ssl_loss_batch': loss.item(),
                    }
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict=scalar_dict,
                    global_step=(ep - 1) * steps + num_batches,
                )
                # Track the accuracies from the KNN models
                if self.track_knn_acc:
                    writer.add_scalars(
                        main_tag="knn_acc",
                        tag_scalar_dict={
                            'src_closest_acc_batch': src_acc,
                            'src_furthest_acc_batch': src_neg_acc,
                            'trgt_closest_acc_batch': trgt_acc,
                            'trgt_furthest_acc_batch': trgt_neg_acc,
                            'src_closest_acc_ep': src_acc_total / num_batches,
                            'src_furthest_acc_ep': src_neg_acc_total / num_batches,
                            'trgt_closest_acc_ep': trgt_acc_total / num_batches,
                            'trgt_furthest_acc_ep': trgt_neg_acc_total / num_batches
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
        div_alpha = self.get_div_alpha(steps=steps, step=steps, ep=ep, ep_total=ep_total)
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
                         "Div-A: {:3.2e}  EMD: {:.3f}  MMD: {:.3f}  Loss: {:.3f}  A1: {:.2f}  A2: {:.2f}  "
                         "A3: {:.2f}  A4: {:.2f}  T: {:.2f}s".format(
                            ep, ep_total, steps, steps,
                            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                            src_loss_total / num_batches, trgt_loss_total / num_batches,
                            div_alpha, div_loss_emd_total / num_batches, div_loss_mmd_total / num_batches,
                            ssl_loss_total / num_batches,
                            src_acc_total / num_batches, trgt_acc_total / num_batches,
                            src_neg_acc_total / num_batches, trgt_neg_acc_total / num_batches,
                            time.time() - start_time)
                         )
