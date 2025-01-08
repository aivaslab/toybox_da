
import time
import numpy as np
from collections import defaultdict

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


class DualSSLJANBase:
    """Module implementing the SSL method for pretraining the DA model with both TB and IN-12 data"""

    def __init__(self, **kwargs):
        # def __init__(self, network, src_loader, trgt_loader, logger, no_save, tb_ssl_loss, in12_ssl_loss,
        #              tb_alpha, in12_alpha, div_alpha, asymmetric, use_ot, div_metric,
        #              fixed_div_alpha, use_div_on_feats, combined_fwd_pass, queue_size, track_knn_acc):
        self.network = kwargs["network"]
        self.src_loader = utils.ForeverDataLoader(kwargs["src_loader"])
        self.trgt_loader = utils.ForeverDataLoader(kwargs["trgt_loader"])
        self.logger = kwargs["logger"]
        self.network.cuda()
        self.network.freeze_train()
        self.no_save = kwargs["no_save"]
        self.tb_ssl_loss = kwargs["tb_ssl_loss"]
        self.in12_ssl_loss = kwargs["in12_ssl_loss"]
        self.tb_alpha = kwargs["tb_alpha"]
        self.in12_alpha = kwargs["in12_alpha"]
        self.combined_fwd_pass = kwargs["combined_fwd_pass"]
        self.tb_feats_queue = None
        self.tb_labels_queue = None
        self.in12_feats_queue = None
        self.in12_labels_queue = None
        self.use_bb_mmd = kwargs["use_bb_mmd"]
        self.track_knn_acc = kwargs["track_knn_acc"]
        self.queue_size = kwargs["queue_size"]
        self.asymmetric = kwargs["asymmetric"]
        self.div_alpha_schedule = kwargs["div_alpha_schedule"]
        self.div_alpha_start = kwargs["div_alpha_start"]
        self.div_alpha = kwargs["div_alpha"]

        self.emd_dist_loss = mmd_util.EMD1DLoss()
        self.mmd_dist_loss = mmd_util.JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([mmd_util.GaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                     ),
            linear=False,
        ).cuda()
        self.dist_frac = 0.05

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")
        torch.autograd.set_detect_anomaly(True)

    def get_paired_distance_chunked(self, src_feats, trgt_feats, metric):
        n1, dim1 = src_feats.size(0), src_feats.size(1)
        n2, dim2 = trgt_feats.size(0), trgt_feats.size(1)
        assert dim1 == dim2
        all_dists = torch.zeros((n1, n2), dtype=torch.float32).cuda()
        num_chunks = 16
        src_chunk_size, trgt_chunk_size = n1 // num_chunks, n2 // num_chunks
        src_chunks, trgt_chunks = torch.chunk(src_feats, num_chunks, dim=0), torch.chunk(trgt_feats, num_chunks, dim=0)
        for chunk in src_chunks:
            assert chunk.shape == (src_chunk_size, dim1)
        for chunk in trgt_chunks:
            assert chunk.shape == (trgt_chunk_size, dim1)

        for i, chunk_1 in enumerate(src_chunks):
            for j, chunk_2 in enumerate(trgt_chunks):
                dist = self.get_paired_distance(src_feats=chunk_1, trgt_feats=chunk_2, metric=metric)
                assert dist.shape == (src_chunk_size, trgt_chunk_size)
                all_dists[i * src_chunk_size: i * src_chunk_size + src_chunk_size,
                j * trgt_chunk_size: j * trgt_chunk_size + trgt_chunk_size] = dist

        return all_dists

    def get_div_alpha(self, step, steps, ep, ep_total):
        if ep <= self.div_alpha_start:
            frac = 0.0
        else:
            total_steps = steps * (ep_total - self.div_alpha_start)
            curr_step = steps * (ep - 1 - self.div_alpha_start) + step
            if self.div_alpha_schedule == "fixed":
                frac = 1.0
            elif self.div_alpha_schedule == "cosine":
                frac = 1 - 0.5 * (1 + np.cos(curr_step * np.pi / total_steps))
            else:
                frac = curr_step / total_steps
        return self.div_alpha * frac

    def calculate_domain_dist_matching_loss(self, src_feats, trgt_feats):
        if self.asymmetric:
            src_feats_cloned = src_feats.clone().detach()

        div_dist_loss = self.mmd_dist_loss((src_feats,), (trgt_feats,))
        return div_dist_loss

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

    @staticmethod
    def get_distance(feats, metric):
        if metric == "cosine":
            dist = func.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=-1)
        elif metric == "euclidean":
            dist = torch.linalg.norm(feats.unsqueeze(1) - feats.unsqueeze(0), dim=-1, ord=2)
        else:
            dist = torch.matmul(feats, feats.transpose(0, 1))
        assert dist.shape == (feats.shape[0], feats.shape[0]), f"{dist.shape}"
        return dist

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""

        self.network.set_train()
        num_batches = 0
        src_loss_total = 0.0
        trgt_loss_total = 0.0
        mmd_loss_total = 0.0
        loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        src_acc_total = 0.0
        src_neg_acc_total = 0.0
        trgt_acc_total = 0.0
        trgt_neg_acc_total = 0.0
        for step in range(1, steps + 1):
            logger_strs = [f"E:{ep}/{ep_total} b:{step}/{steps} LR:{optimizer.param_groups[0]['lr']:.3f}"]
            tb_scalar_dicts = defaultdict(dict)
            tb_scalar_dicts["training_lr"] = {
                'bb': optimizer.param_groups[0]['lr'],
                'ssl_head': optimizer.param_groups[1]['lr'],
            }
            num_batches += 1
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

            if self.combined_fwd_pass and self.tb_alpha > 0.0:
                images = torch.cat([src_batch, trgt_batch], dim=0)
                images = images.cuda()
                bb_feats, ssl_feats = self.network.forward_w_feats(images)
                src_bb_feats, trgt_bb_feats = bb_feats[:src_size], bb_feats[src_size:]
                src_ssl_feats, trgt_ssl_feats = ssl_feats[:src_size], ssl_feats[src_size:]
            else:
                src_batch, trgt_batch = src_batch.cuda(), trgt_batch.cuda()
                if self.tb_alpha == 0.0:
                    if step == 1 and ep == 1:
                        print("Using separate fwd passes because tb-alpha=0.0")
                    with torch.no_grad():
                        src_bb_feats, src_ssl_feats = self.network.forward_w_feats(src_batch)
                else:
                    src_bb_feats, src_ssl_feats = self.network.forward_w_feats(src_batch)
                trgt_bb_feats, trgt_ssl_feats = self.network.forward_w_feats(trgt_batch)

            src_anchor_feats_ssl = src_ssl_feats[:src_size//2]
            src_anchor_feats_bb = src_bb_feats[:src_size//2]
            trgt_anchor_feats_ssl = trgt_ssl_feats[:trgt_size//2]
            trgt_anchor_feats_bb = trgt_bb_feats[:trgt_size//2]

            if self.tb_ssl_loss == "simclr":
                src_logits, src_labels_info_nce = utils.info_nce_loss(features=src_ssl_feats, temp=0.5)
                src_loss = criterion(src_logits, src_labels_info_nce)
            elif self.tb_ssl_loss == "dcl":
                src_loss = utils.decoupled_contrastive_loss(features=src_ssl_feats, temp=0.1)
            else:
                concat_src_labels = torch.cat([src_labels for _ in range(4)], dim=0)
                # print(src_labels.shape, concat_src_labels.shape)
                src_loss = utils.sup_decoupled_contrastive_loss(features=src_ssl_feats, temp=0.1,
                                                                labels=concat_src_labels)

            if self.in12_ssl_loss == "dcl":
                trgt_loss = utils.decoupled_contrastive_loss(features=trgt_ssl_feats, temp=0.1)
            else:
                trgt_logits, trgt_labels_info_nce = utils.info_nce_loss(features=trgt_ssl_feats, temp=0.5)
                trgt_loss = criterion(trgt_logits, trgt_labels_info_nce)
            src_loss_total += src_loss.item()
            trgt_loss_total += trgt_loss.item()

            logger_strs.append(f"SSL1:{src_loss_total / num_batches:.2f} SSL2:{trgt_loss_total / num_batches:.2f}")
            num_images_src = len(src_images[0])
            num_images_trgt = len(trgt_images[0])
            if self.use_bb_mmd:
                src_feats_1, src_feats_2 = src_bb_feats[:num_images_src], \
                                        src_bb_feats[num_images_src:2 * num_images_src]
                trgt_feats_1 = trgt_bb_feats[:num_images_trgt]
            else:
                src_feats_1, src_feats_2 = src_ssl_feats[:num_images_src], \
                    src_ssl_feats[num_images_src:2 * num_images_src]
                trgt_feats_1 = trgt_ssl_feats[:num_images_trgt]
            src_feats_stacked = torch.stack(tensors=(src_feats_1, src_feats_2), dim=1).view(2 * num_images_src, -1)
            domain_dist_loss = self.calculate_domain_dist_matching_loss(src_feats=src_feats_stacked,
                                                                        trgt_feats=trgt_feats_1)

            div_alpha = self.get_div_alpha(steps=steps, step=step, ep=ep, ep_total=ep_total)
            mmd_loss_total += domain_dist_loss.item()
            logger_strs.append(f"MMD:{mmd_loss_total / num_batches:.4f}")
            loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss + div_alpha * domain_dist_loss

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_total += loss.item()
            logger_strs.append(f"alfa: {div_alpha:1.1e} Loss:{loss_total/num_batches:.2f}")

            tb_scalar_dicts["training_loss"] = {
                'src_ssl_loss_ep': src_loss_total / num_batches,
                'src_ssl_loss_batch': src_loss.item(),
                'trgt_ssl_loss_ep': trgt_loss_total / num_batches,
                'trgt_ssl_loss_batch': trgt_loss.item(),
                'mmd_loss_ep': mmd_loss_total / num_batches,
                'mmd_loss_batch': domain_dist_loss.item(),
                'loss_ep': loss_total / num_batches,
                'loss_batch': loss.item(),
            }

            if self.track_knn_acc:
                if self.use_bb_mmd:
                    src_anchor_feats, trgt_anchor_feats = src_anchor_feats_bb, trgt_anchor_feats_bb
                else:
                    src_anchor_feats, trgt_anchor_feats = src_anchor_feats_ssl, trgt_anchor_feats_ssl
                # Track within-batch accuracy using KNNs
                # NOTE: This only considers the anchor images. Positive pairs are not considered for KNN acc.
                # calculation
                with (torch.no_grad()):
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

                    src_dist_mat = self.get_paired_distance_chunked(src_feats=src_anchor_feats,
                                                                    trgt_feats=self.tb_feats_queue,
                                                                    metric="cosine")
                    trgt_dist_mat = self.get_paired_distance_chunked(src_feats=trgt_anchor_feats,
                                                                     trgt_feats=self.in12_feats_queue,
                                                                     metric="cosine")

                    src_k = max(int(self.dist_frac * len(self.tb_feats_queue)), 1)

                    # Get top-2 closest image and discard closest (this should be the image itself) for src feats
                    src_topk_closest_indices = torch.topk(src_dist_mat, k=src_k+1, largest=True).indices[:, 1:]
                    src_topk_labels = self.tb_labels_queue[src_topk_closest_indices]
                    src_topk_matches = torch.sum((src_topk_labels == dupl_src_labels.unsqueeze(1)).int())
                    # print(src_topk_matches)

                    # Get the furthest image for src feats
                    _, src_topk_farthest_indices = torch.topk(src_dist_mat, k=src_k, largest=False)
                    src_farthest_labels = self.tb_labels_queue[src_topk_farthest_indices]
                    src_farthest_matches = torch.sum((src_farthest_labels == dupl_src_labels.unsqueeze(1)).int())

                    # Calculate src accuracies
                    src_acc = round(100 * src_topk_matches.item() / (src_k * len(dupl_src_labels)), 2)
                    src_neg_acc = round(100 * src_farthest_matches.item() / (src_k * len(dupl_src_labels)), 2)

                    trgt_k = max(int(self.dist_frac * len(self.in12_feats_queue)), 1)
                    # Get top-2 closes image and discard closest (this should be the image itself) for trgt feats
                    trgt_topk_closest_indices = torch.topk(trgt_dist_mat, k=trgt_k+1, largest=True).indices[:, 1:]
                    trgt_topk_labels = self.in12_labels_queue[trgt_topk_closest_indices]
                    trgt_topk_matches = torch.sum((trgt_topk_labels == trgt_labels.unsqueeze(1)).int())
                    # trgt_topk_labels = self.in12_labels_queue[trgt_topk_closest_indices][:, 1]

                    # Get the furthest image for trgt feats
                    _, trgt_topk_farthest_indices = torch.topk(trgt_dist_mat, k=trgt_k, largest=False)
                    trgt_farthest_labels = self.in12_labels_queue[trgt_topk_farthest_indices]
                    trgt_farthest_matches = torch.sum((trgt_farthest_labels == trgt_labels.unsqueeze(1)).int())
                    # trgt_farthest_labels = self.in12_labels_queue[trgt_topk_farthest_indices][:, 0]

                    # Calculate trgt accuracies
                    trgt_acc = round(100 * trgt_topk_matches.item() / (trgt_k * len(trgt_labels)), 2)
                    trgt_neg_acc = round(100 * trgt_farthest_matches.item() / (trgt_k * len(trgt_labels)), 2)

                    src_acc_total += src_acc
                    src_neg_acc_total += src_neg_acc
                    trgt_acc_total += trgt_acc
                    trgt_neg_acc_total += trgt_neg_acc
                    logger_strs.append(f"A1:{src_acc_total / num_batches:.2f} A2:{trgt_acc_total / num_batches:.2f} "
                                       f"A3:{src_neg_acc_total / num_batches:.2f} "
                                       f"A4:{trgt_neg_acc_total / num_batches:.2f}")
                    if step == steps:
                        trgt_q_dist_mat = self.get_paired_distance_chunked(src_feats=self.in12_feats_queue,
                                                                           trgt_feats=self.in12_feats_queue,
                                                                           metric="cosine")

                        # Calculate mean within-class and across class distances for source queue
                        src_q_dist_mat = self.get_paired_distance_chunked(src_feats=self.tb_feats_queue,
                                                                          trgt_feats=self.tb_feats_queue,
                                                                          metric="cosine")

                        src_q_cl_match_matrix = (self.tb_labels_queue.unsqueeze(1) ==
                                                 self.tb_labels_queue.unsqueeze(0)).int()
                        src_q_cl_mismatch_matrix = torch.ones_like(src_q_cl_match_matrix) - src_q_cl_match_matrix
                        assert torch.sum(src_q_cl_match_matrix) + torch.sum(src_q_cl_mismatch_matrix) == \
                               src_q_cl_match_matrix.shape[0] * src_q_cl_match_matrix.shape[1]

                        tb_cl_match_dists = src_q_dist_mat * src_q_cl_match_matrix
                        tb_cl_mismatch_dists = src_q_dist_mat * src_q_cl_mismatch_matrix

                        tb_cl_match_ave_dist = torch.sum(tb_cl_match_dists) / torch.sum(src_q_cl_match_matrix)
                        tb_cl_mismatch_ave_dist = torch.sum(tb_cl_mismatch_dists) / torch.sum(src_q_cl_mismatch_matrix)

                        # Calculate mean within-class and across class distances for source queue
                        trgt_q_cl_match_matrix = (self.in12_labels_queue.unsqueeze(1) ==
                                                  self.in12_labels_queue.unsqueeze(0)).int()
                        trgt_q_cl_mismatch_matrix = torch.ones_like(trgt_q_cl_match_matrix) - trgt_q_cl_match_matrix
                        assert torch.sum(trgt_q_cl_match_matrix) + torch.sum(trgt_q_cl_mismatch_matrix) == \
                               trgt_q_cl_match_matrix.shape[0] * trgt_q_cl_match_matrix.shape[1]

                        in12_cl_match_dists = trgt_q_dist_mat * trgt_q_cl_match_matrix
                        in12_cl_mismatch_dists = trgt_q_dist_mat * trgt_q_cl_mismatch_matrix

                        in12_cl_match_ave_dist = torch.sum(in12_cl_match_dists) / torch.sum(trgt_q_cl_match_matrix)
                        in12_cl_mismatch_ave_dist = \
                            torch.sum(in12_cl_mismatch_dists) / torch.sum(trgt_q_cl_mismatch_matrix)

                        logger_strs.append(
                            f"D:[{tb_cl_match_ave_dist:.2f} {tb_cl_mismatch_ave_dist:.2f} "
                            f"{in12_cl_match_ave_dist:.2f} {in12_cl_mismatch_ave_dist:.2f}]"
                        )
                tb_scalar_dicts["knn_queue"] = {
                    'tb_queue_size': len(self.tb_labels_queue),
                    'in12_queue_size': len(self.in12_labels_queue),
                }
                tb_scalar_dicts["knn_acc"] = {
                    'src_closest_acc_batch': src_acc,
                    'src_furthest_acc_batch': src_neg_acc,
                    'trgt_closest_acc_batch': trgt_acc,
                    'trgt_furthest_acc_batch': trgt_neg_acc,
                    'src_closest_acc_ep': src_acc_total / num_batches,
                    'src_furthest_acc_ep': src_neg_acc_total / num_batches,
                    'trgt_closest_acc_ep': trgt_acc_total / num_batches,
                    'trgt_furthest_acc_ep': trgt_neg_acc_total / num_batches
                }
            assert optimizer.param_groups[1]['lr'] == optimizer.param_groups[0]['lr']
            if 0 <= step - halfway < 1 or step == steps:
                logger_strs.append(f"T:{time.time() - start_time:.0f}s")
                logger_str = " ".join(logger_strs)
                self.logger.info(logger_str)

            if not self.no_save:
                for key, val_dict in tb_scalar_dicts.items():
                    writer.add_scalars(
                        main_tag=key,
                        tag_scalar_dict=val_dict,
                        global_step=(ep - 1) * steps + num_batches
                    )


class DualSSLWithinDomainDistMatchingModelBase:
    """Module implementing the SSL method for pretraining the DA model with both TB and IN-12 data"""

    def __init__(self, **kwargs):
        # def __init__(self, network, src_loader, trgt_loader, logger, no_save, tb_ssl_loss, in12_ssl_loss,
        #              tb_alpha, in12_alpha, div_alpha, asymmetric, use_ot, div_metric,
        #              fixed_div_alpha, use_div_on_feats, combined_fwd_pass, queue_size, track_knn_acc):
        self.network = kwargs["network"]
        self.src_loader = utils.ForeverDataLoader(kwargs["src_loader"])
        self.trgt_loader = utils.ForeverDataLoader(kwargs["trgt_loader"])
        self.logger = kwargs["logger"]
        self.network.cuda()
        self.network.freeze_train()
        self.no_save = kwargs["no_save"]
        self.tb_ssl_loss = kwargs["tb_ssl_loss"]
        self.in12_ssl_loss = kwargs["in12_ssl_loss"]
        self.tb_alpha = kwargs["tb_alpha"]
        self.in12_alpha = kwargs["in12_alpha"]
        self.div_metric = kwargs["div_metric"]
        self.combined_fwd_pass = kwargs["combined_fwd_pass"]
        self.queue_size = kwargs["queue_size"]
        self.tb_feats_queue = None
        self.tb_labels_queue = None
        self.in12_feats_queue = None
        self.in12_labels_queue = None
        self.track_knn_acc = kwargs["track_knn_acc"]
        self.ind_mmd_loss = kwargs["ind_mmd_loss"]
        self.use_bb_mmd = kwargs["use_bb_mmd"]
        self.dist_frac = 0.05
        self.neg_dcl_weight_min = 0.1
        self.neg_dcl_weight_max = 5.0

        self.emd_dist_loss = mmd_util.EMD1DLoss()
        if self.ind_mmd_loss:
            self.mmd_dist_loss = mmd_util.IndividualMMDLoss(
                kernels=[mmd_util.IndGaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                linear=False,
            ).cuda()
        else:
            self.mmd_dist_loss = mmd_util.JointMultipleKernelMaximumMeanDiscrepancy(
                kernels=([mmd_util.GaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                         ),
                linear=False,
            ).cuda()

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")
        torch.autograd.set_detect_anomaly(True)

    def get_paired_distance_chunked(self, src_feats, trgt_feats, metric):
        n1, dim1 = src_feats.size(0), src_feats.size(1)
        n2, dim2 = trgt_feats.size(0), trgt_feats.size(1)
        assert dim1 == dim2
        all_dists = torch.zeros((n1, n2), dtype=torch.float32).cuda()
        num_chunks = 16
        src_chunk_size, trgt_chunk_size = n1 // num_chunks, n2 // num_chunks
        src_chunks, trgt_chunks = torch.chunk(src_feats, num_chunks, dim=0), torch.chunk(trgt_feats, num_chunks, dim=0)
        for chunk in src_chunks:
            assert chunk.shape == (src_chunk_size, dim1)
        for chunk in trgt_chunks:
            assert chunk.shape == (trgt_chunk_size, dim1)

        for i, chunk_1 in enumerate(src_chunks):
            for j, chunk_2 in enumerate(trgt_chunks):
                dist = self.get_paired_distance(src_feats=chunk_1, trgt_feats=chunk_2, metric=metric)
                assert dist.shape == (src_chunk_size, trgt_chunk_size)
                all_dists[i * src_chunk_size: i * src_chunk_size + src_chunk_size,
                j * trgt_chunk_size: j * trgt_chunk_size + trgt_chunk_size] = dist

        return all_dists

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

    @staticmethod
    def get_distance(feats, metric):
        if metric == "cosine":
            dist = func.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=-1)
        elif metric == "euclidean":
            dist = torch.linalg.norm(feats.unsqueeze(1) - feats.unsqueeze(0), dim=-1, ord=2)
        else:
            dist = torch.matmul(feats, feats.transpose(0, 1))
        assert dist.shape == (feats.shape[0], feats.shape[0]), f"{dist.shape}"
        return dist

    def track_domain_dist_matching_loss(self, src_feats, trgt_feats, tracking_vars, tb_scalar_dicts, logger_strs):
        with torch.no_grad():
            src_feats_dists = self.get_distance(src_feats, metric=self.div_metric)
            src_feats_dists_reshaped = torch.reshape(src_feats_dists, (-1, 1))
            num_samples = 1024
            src_sample_idxs = torch.randperm(src_feats_dists_reshaped.size(0))[:num_samples]
            src_samples = src_feats_dists_reshaped[src_sample_idxs]
            # print(src_feats_dists_reshaped.shape, src_samples.shape,
            #       torch.mean(src_feats_dists_reshaped), torch.std(src_feats_dists_reshaped),
            #       torch.mean(src_samples), torch.std(src_samples))

            trgt_feats_dists = self.get_distance(trgt_feats, metric=self.div_metric)
            trgt_feats_dists_reshaped = torch.reshape(trgt_feats_dists, (-1, 1))
            trgt_sample_idxs = torch.randperm(trgt_feats_dists_reshaped.size(0))[:num_samples]
            trgt_samples = trgt_feats_dists_reshaped[trgt_sample_idxs]
            # print(trgt_feats_dists_reshaped.shape, trgt_samples.shape,
            #       torch.mean(trgt_feats_dists_reshaped), torch.std(trgt_feats_dists_reshaped),
            #       torch.mean(trgt_samples), torch.std(trgt_samples))

            div_dist_emd_loss = self.emd_dist_loss(src_samples, trgt_samples)
            if self.ind_mmd_loss:
                div_dist_mmd_loss = self.mmd_dist_loss(src_feats_dists, trgt_feats_dists)
            else:
                div_dist_mmd_loss = self.mmd_dist_loss((src_samples,), (trgt_samples,))
            tracking_vars["div_loss_emd_total"] += div_dist_emd_loss.item()
            tracking_vars["div_loss_mmd_total"] += div_dist_mmd_loss.item()
            tracking_vars["src_dist_total"] += torch.mean(src_feats_dists).item()
            tracking_vars["trgt_dist_total"] += torch.mean(trgt_feats_dists).item()
            tracking_vars["num_batches"] += 1

        logger_strs.append(f"EMD:{tracking_vars['div_loss_emd_total'] / tracking_vars['num_batches']:.3f} "
                           f"MMD:{tracking_vars['div_loss_mmd_total'] / tracking_vars['num_batches']:.3f} "
                           f"D:[{tracking_vars['src_dist_total'] / tracking_vars['num_batches']:.2f} "
                           f"{tracking_vars['trgt_dist_total'] / tracking_vars['num_batches']:.2f}]"
                           )
        tb_scalar_dicts["dom_dist_loss"] = {
            'div_loss_emd_ep': tracking_vars["div_loss_emd_total"] / tracking_vars["num_batches"],
            'div_loss_emd_batch': div_dist_emd_loss.item(),
            'div_loss_mmd_ep': tracking_vars["div_loss_mmd_total"] / tracking_vars["num_batches"],
            'div_loss_mmd_batch': div_dist_mmd_loss.item(),
        }

        tb_scalar_dicts["dom_dist"] = {
            "src_dist": tracking_vars["src_dist_total"] / tracking_vars["num_batches"],
            "trgt_dist": tracking_vars["trgt_dist_total"] / tracking_vars["num_batches"],
        }

    def calculate_domain_dist_matching_loss(self, src_feats, trgt_feats):
        return torch.tensor(0.0, dtype=torch.float, device=src_feats.device)

    def get_div_alpha(self, step, steps, ep, ep_total):
        return 0.0

    def get_dcl_alpha(self, ep, step, steps, ep_total):
        total_steps = ep_total * steps
        curr_step = (ep - 1) * steps + step
        ratio = (self.neg_dcl_weight_min / self.neg_dcl_weight_max) ** (1 / total_steps)
        return self.neg_dcl_weight_max * (ratio ** (curr_step - 1))

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""

        self.network.set_train()
        num_batches = 0
        src_loss_total = 0.0
        trgt_loss_total = 0.0
        loss_total = 0.0
        tracking_vars = defaultdict(float)
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        src_acc_total = 0.0
        src_neg_acc_total = 0.0
        trgt_acc_total = 0.0
        trgt_neg_acc_total = 0.0
        for step in range(1, steps + 1):
            logger_strs = [f"E:{ep}/{ep_total} b:{step}/{steps} LR:{optimizer.param_groups[0]['lr']:.3f}"]
            tb_scalar_dicts = defaultdict(dict)
            tb_scalar_dicts["training_lr"] = {
                'bb': optimizer.param_groups[0]['lr'],
                'ssl_head': optimizer.param_groups[1]['lr'],
            }
            num_batches += 1
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

            if self.combined_fwd_pass and self.tb_alpha > 0.0:
                images = torch.cat([src_batch, trgt_batch], dim=0)
                images = images.cuda()
                bb_feats, ssl_feats = self.network.forward_w_feats(images)
                src_bb_feats, trgt_bb_feats = bb_feats[:src_size], bb_feats[src_size:]
                src_ssl_feats, trgt_ssl_feats = ssl_feats[:src_size], ssl_feats[src_size:]
            else:
                src_batch, trgt_batch = src_batch.cuda(), trgt_batch.cuda()
                if self.tb_alpha == 0.0:
                    if step == 1 and ep == 1:
                        print("Using separate fwd passes because tb-alpha=0.0")
                    with torch.no_grad():
                        src_bb_feats, src_ssl_feats = self.network.forward_w_feats(src_batch)
                else:
                    src_bb_feats, src_ssl_feats = self.network.forward_w_feats(src_batch)
                trgt_bb_feats, trgt_ssl_feats = self.network.forward_w_feats(trgt_batch)

            src_anchor_feats_ssl = src_ssl_feats[:src_size//2]
            src_anchor_feats_bb = src_bb_feats[:src_size//2]
            trgt_anchor_feats_ssl = trgt_ssl_feats[:trgt_size//2]
            trgt_anchor_feats_bb = trgt_bb_feats[:trgt_size//2]

            if self.tb_ssl_loss == "simclr":
                src_logits, src_labels_info_nce = utils.info_nce_loss(features=src_ssl_feats, temp=0.5)
                src_loss = criterion(src_logits, src_labels_info_nce)
            elif self.tb_ssl_loss == "dcl":
                src_loss = utils.decoupled_contrastive_loss(features=src_ssl_feats, temp=0.1)
            else:
                concat_src_labels = torch.cat([src_labels for _ in range(4)], dim=0)
                # print(src_labels.shape, concat_src_labels.shape)
                src_loss = utils.sup_decoupled_contrastive_loss(features=src_ssl_feats, temp=0.1,
                                                                labels=concat_src_labels)

            if self.in12_ssl_loss == "nwdcl":
                neg_alpha = self.get_dcl_alpha(ep=ep, ep_total=ep_total, step=step, steps=steps)
                trgt_loss = utils.neg_weighted_dcl(features=trgt_ssl_feats, temp=0.1, neg_weights_temp=neg_alpha)
                logger_strs.append(f"tau: {neg_alpha:.2f}")
            elif self.in12_ssl_loss == "dcl":
                trgt_loss = utils.decoupled_contrastive_loss(features=trgt_ssl_feats, temp=0.1)
            else:
                trgt_logits, trgt_labels_info_nce = utils.info_nce_loss(features=trgt_ssl_feats, temp=0.5)
                trgt_loss = criterion(trgt_logits, trgt_labels_info_nce)
            src_loss_total += src_loss.item()
            trgt_loss_total += trgt_loss.item()

            logger_strs.append(f"SSL1:{src_loss_total / num_batches:.2f} SSL2:{trgt_loss_total / num_batches:.2f}")
            num_images_src = len(src_images[0])
            num_images_trgt = len(trgt_images[0])
            if self.use_bb_mmd:
                src_feats_1, src_feats_2 = src_bb_feats[:num_images_src], \
                                        src_bb_feats[num_images_src:2 * num_images_src]
                trgt_feats_1 = trgt_bb_feats[:num_images_trgt]
            else:
                src_feats_1, src_feats_2 = src_ssl_feats[:num_images_src], \
                    src_ssl_feats[num_images_src:2 * num_images_src]
                trgt_feats_1 = trgt_ssl_feats[:num_images_trgt]
            src_feats_stacked = torch.stack(tensors=(src_feats_1, src_feats_2), dim=1).view(2 * num_images_src, -1)
            self.track_domain_dist_matching_loss(src_feats=src_feats_stacked, trgt_feats=trgt_feats_1,
                                                 tracking_vars=tracking_vars, logger_strs=logger_strs,
                                                 tb_scalar_dicts=tb_scalar_dicts)

            domain_dist_loss = self.calculate_domain_dist_matching_loss(src_feats=src_feats_stacked,
                                                                        trgt_feats=trgt_feats_1)

            div_alpha = self.get_div_alpha(steps=steps, step=step, ep=ep, ep_total=ep_total)
            loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss + div_alpha * domain_dist_loss

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_total += loss.item()
            logger_strs.append(f"alfa: {div_alpha:1.1e} Loss:{loss_total/num_batches:.2f}")

            tb_scalar_dicts["training_loss"] = {
                'src_ssl_loss_ep': src_loss_total / num_batches,
                'src_ssl_loss_batch': src_loss.item(),
                'trgt_ssl_loss_ep': trgt_loss_total / num_batches,
                'trgt_ssl_loss_batch': trgt_loss.item(),
                'loss_ep': loss_total / num_batches,
                'loss_batch': loss.item(),
            }

            if self.track_knn_acc:
                if self.use_bb_mmd:
                    src_anchor_feats, trgt_anchor_feats = src_anchor_feats_bb, trgt_anchor_feats_bb
                else:
                    src_anchor_feats, trgt_anchor_feats = src_anchor_feats_ssl, trgt_anchor_feats_ssl
                # print(src_anchor_feats.shape, trgt_anchor_feats.shape)
                # Track within-batch accuracy using KNNs
                # NOTE: This only considers the anchor images. Positive pairs are not considered for KNN acc.
                # calculation
                with (torch.no_grad()):
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
                    # print(len(self.tb_feats_queue), len(self.in12_feats_queue))
                    src_dist_mat = self.get_paired_distance_chunked(src_feats=src_anchor_feats,
                                                                    trgt_feats=self.tb_feats_queue,
                                                                    metric="cosine")
                    trgt_dist_mat = self.get_paired_distance_chunked(src_feats=trgt_anchor_feats,
                                                                     trgt_feats=self.in12_feats_queue,
                                                                     metric="cosine")

                    if step == steps and len(self.in12_feats_queue) > 10000:
                        acc_log_strs = ["Accs:"]
                        src_accs, trgt_accs, src_neg_accs, trgt_neg_accs, dist_fracs = [], [], [], [], []
                        for dist_frac in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
                            src_k = max(int(dist_frac * len(self.tb_feats_queue)), 1)

                            # Get top-2 closest image and discard closest
                            # (this should be the image itself) for src feats
                            src_topk_closest_indices = torch.topk(src_dist_mat,
                                                                  k=min(src_k + 1, len(self.tb_feats_queue)),
                                                                  largest=True).indices
                            src_topk_labels = self.tb_labels_queue[src_topk_closest_indices]
                            src_topk_matches = torch.sum((src_topk_labels == dupl_src_labels.unsqueeze(1)).int())
                            # print(src_topk_matches)

                            # Get the furthest image for src feats
                            _, src_topk_farthest_indices = torch.topk(src_dist_mat, k=src_k, largest=False)
                            src_farthest_labels = self.tb_labels_queue[src_topk_farthest_indices]
                            src_farthest_matches = torch.sum(
                                (src_farthest_labels == dupl_src_labels.unsqueeze(1)).int())

                            # Calculate src accuracies
                            src_acc = round(100 * src_topk_matches.item() / (src_k * len(dupl_src_labels)), 2)
                            src_neg_acc = round(100 * src_farthest_matches.item() / (src_k * len(dupl_src_labels)), 2)

                            trgt_k = int(dist_frac * len(self.in12_feats_queue))

                            # Get top-2 closes image and discard closest
                            # (this should be the image itself) for trgt feats
                            trgt_topk_closest_indices = torch.topk(trgt_dist_mat,
                                                                   k=min(trgt_k + 1, len(self.in12_feats_queue)),
                                                                   largest=True).indices
                            trgt_topk_labels = self.in12_labels_queue[trgt_topk_closest_indices]
                            trgt_topk_matches = torch.sum((trgt_topk_labels == trgt_labels.unsqueeze(1)).int())
                            # trgt_topk_labels = self.in12_labels_queue[trgt_topk_closest_indices][:, 1]

                            # Get the furthest image for trgt feats
                            _, trgt_topk_farthest_indices = torch.topk(trgt_dist_mat, k=trgt_k, largest=False)
                            trgt_farthest_labels = self.in12_labels_queue[trgt_topk_farthest_indices]
                            trgt_farthest_matches = torch.sum((trgt_farthest_labels == trgt_labels.unsqueeze(1)).int())
                            # trgt_farthest_labels = self.in12_labels_queue[trgt_topk_farthest_indices][:, 0]

                            # Calculate trgt accuracies
                            trgt_acc = round(100 * trgt_topk_matches.item() / (trgt_k * len(trgt_labels)), 2)
                            trgt_neg_acc = round(100 * trgt_farthest_matches.item() / (trgt_k * len(trgt_labels)), 2)
                            dist_fracs.append(dist_frac)
                            src_accs.append(src_acc)
                            src_neg_accs.append(src_neg_acc)
                            trgt_accs.append(trgt_acc)
                            trgt_neg_accs.append(trgt_neg_acc)
                            acc_log_strs.append(f"{dist_frac:.2f}: [{src_acc:.1f} {trgt_acc:.1f} {src_neg_acc:.1f}"
                                                f" {trgt_neg_acc:.1f}]")
                        self.logger.info(" ".join(acc_log_strs))
                        tb_scalar_dicts["knn_src_acc_distribution"] = {}
                        tb_scalar_dicts["knn_src_neg_acc_distribution"] = {}
                        tb_scalar_dicts["knn_trgt_acc_distribution"] = {}
                        tb_scalar_dicts["knn_trgt_neg_acc_distribution"] = {}
                        for i in range(len(dist_fracs)):
                            dist_frac, src_acc, src_neg_acc, trgt_acc, trgt_neg_acc = dist_fracs[i], src_accs[i], \
                                src_neg_accs[i], trgt_accs[i], trgt_neg_accs[i]
                            tb_scalar_dicts["knn_src_acc_distribution"][str(dist_frac)] = src_acc
                            tb_scalar_dicts["knn_trgt_acc_distribution"][str(dist_frac)] = trgt_acc
                            tb_scalar_dicts["knn_src_neg_acc_distribution"][str(dist_frac)] = src_neg_acc
                            tb_scalar_dicts["knn_trgt_neg_acc_distribution"][str(dist_frac)] = trgt_neg_acc

                    src_k = max(int(self.dist_frac * len(self.tb_feats_queue)), 1)

                    # Get top-2 closest image and discard closest (this should be the image itself) for src feats
                    src_topk_closest_indices = torch.topk(src_dist_mat, k=src_k+1, largest=True).indices[:, 1:]
                    src_topk_labels = self.tb_labels_queue[src_topk_closest_indices]
                    src_topk_matches = torch.sum((src_topk_labels == dupl_src_labels.unsqueeze(1)).int())
                    # print(src_topk_matches)

                    # Get the furthest image for src feats
                    _, src_topk_farthest_indices = torch.topk(src_dist_mat, k=src_k, largest=False)
                    src_farthest_labels = self.tb_labels_queue[src_topk_farthest_indices]
                    src_farthest_matches = torch.sum((src_farthest_labels == dupl_src_labels.unsqueeze(1)).int())

                    # Calculate src accuracies
                    src_acc = round(100 * src_topk_matches.item() / (src_k * len(dupl_src_labels)), 2)
                    src_neg_acc = round(100 * src_farthest_matches.item() / (src_k * len(dupl_src_labels)), 2)

                    trgt_k = max(int(self.dist_frac * len(self.in12_feats_queue)), 1)
                    # Get top-2 closes image and discard closest (this should be the image itself) for trgt feats
                    trgt_topk_closest_indices = torch.topk(trgt_dist_mat, k=trgt_k+1, largest=True).indices[:, 1:]
                    trgt_topk_labels = self.in12_labels_queue[trgt_topk_closest_indices]
                    trgt_topk_matches = torch.sum((trgt_topk_labels == trgt_labels.unsqueeze(1)).int())
                    # trgt_topk_labels = self.in12_labels_queue[trgt_topk_closest_indices][:, 1]

                    # Get the furthest image for trgt feats
                    _, trgt_topk_farthest_indices = torch.topk(trgt_dist_mat, k=trgt_k, largest=False)
                    trgt_farthest_labels = self.in12_labels_queue[trgt_topk_farthest_indices]
                    trgt_farthest_matches = torch.sum((trgt_farthest_labels == trgt_labels.unsqueeze(1)).int())
                    # trgt_farthest_labels = self.in12_labels_queue[trgt_topk_farthest_indices][:, 0]

                    # Calculate trgt accuracies
                    trgt_acc = round(100 * trgt_topk_matches.item() / (trgt_k * len(trgt_labels)), 2)
                    trgt_neg_acc = round(100 * trgt_farthest_matches.item() / (trgt_k * len(trgt_labels)), 2)

                    src_acc_total += src_acc
                    src_neg_acc_total += src_neg_acc
                    trgt_acc_total += trgt_acc
                    trgt_neg_acc_total += trgt_neg_acc
                    logger_strs.append(f"A1:{src_acc_total / num_batches:.2f} A2:{trgt_acc_total / num_batches:.2f} "
                                       f"A3:{src_neg_acc_total / num_batches:.2f} "
                                       f"A4:{trgt_neg_acc_total / num_batches:.2f}")
                    if step == steps:
                        trgt_q_dist_mat = self.get_paired_distance_chunked(src_feats=self.in12_feats_queue,
                                                                           trgt_feats=self.in12_feats_queue,
                                                                           metric="cosine")

                        # Calculate mean within-class and across class distances for source queue
                        src_q_dist_mat = self.get_paired_distance_chunked(src_feats=self.tb_feats_queue,
                                                                          trgt_feats=self.tb_feats_queue,
                                                                          metric="cosine")

                        src_q_cl_match_matrix = (self.tb_labels_queue.unsqueeze(1) ==
                                                 self.tb_labels_queue.unsqueeze(0)).int()
                        src_q_cl_mismatch_matrix = torch.ones_like(src_q_cl_match_matrix) - src_q_cl_match_matrix
                        assert torch.sum(src_q_cl_match_matrix) + torch.sum(src_q_cl_mismatch_matrix) == \
                               src_q_cl_match_matrix.shape[0] * src_q_cl_match_matrix.shape[1]

                        tb_cl_match_dists = src_q_dist_mat * src_q_cl_match_matrix
                        tb_cl_mismatch_dists = src_q_dist_mat * src_q_cl_mismatch_matrix

                        tb_cl_match_ave_dist = torch.sum(tb_cl_match_dists) / torch.sum(src_q_cl_match_matrix)
                        tb_cl_mismatch_ave_dist = torch.sum(tb_cl_mismatch_dists) / torch.sum(src_q_cl_mismatch_matrix)

                        # Calculate mean within-class and across class distances for source queue
                        trgt_q_cl_match_matrix = (self.in12_labels_queue.unsqueeze(1) ==
                                                  self.in12_labels_queue.unsqueeze(0)).int()
                        trgt_q_cl_mismatch_matrix = torch.ones_like(trgt_q_cl_match_matrix) - trgt_q_cl_match_matrix
                        assert torch.sum(trgt_q_cl_match_matrix) + torch.sum(trgt_q_cl_mismatch_matrix) == \
                               trgt_q_cl_match_matrix.shape[0] * trgt_q_cl_match_matrix.shape[1]

                        in12_cl_match_dists = trgt_q_dist_mat * trgt_q_cl_match_matrix
                        in12_cl_mismatch_dists = trgt_q_dist_mat * trgt_q_cl_mismatch_matrix

                        in12_cl_match_ave_dist = torch.sum(in12_cl_match_dists) / torch.sum(trgt_q_cl_match_matrix)
                        in12_cl_mismatch_ave_dist = \
                            torch.sum(in12_cl_mismatch_dists) / torch.sum(trgt_q_cl_mismatch_matrix)

                        logger_strs.append(
                            f"D:[{tb_cl_match_ave_dist:.2f} {tb_cl_mismatch_ave_dist:.2f} "
                            f"{in12_cl_match_ave_dist:.2f} {in12_cl_mismatch_ave_dist:.2f}]"
                        )
                tb_scalar_dicts["knn_queue"] = {
                    'tb_queue_size': len(self.tb_labels_queue),
                    'in12_queue_size': len(self.in12_labels_queue),
                }
                tb_scalar_dicts["knn_acc"] = {
                    'src_closest_acc_batch': src_acc,
                    'src_furthest_acc_batch': src_neg_acc,
                    'trgt_closest_acc_batch': trgt_acc,
                    'trgt_furthest_acc_batch': trgt_neg_acc,
                    'src_closest_acc_ep': src_acc_total / num_batches,
                    'src_furthest_acc_ep': src_neg_acc_total / num_batches,
                    'trgt_closest_acc_ep': trgt_acc_total / num_batches,
                    'trgt_furthest_acc_ep': trgt_neg_acc_total / num_batches
                }
            assert optimizer.param_groups[1]['lr'] == optimizer.param_groups[0]['lr']
            if 0 <= step - halfway < 1 or step == steps:
                logger_strs.append(f"T:{time.time() - start_time:.0f}s")
                logger_str = " ".join(logger_strs)
                self.logger.info(logger_str)

            if not self.no_save:
                for key, val_dict in tb_scalar_dicts.items():
                    writer.add_scalars(
                        main_tag=key,
                        tag_scalar_dict=val_dict,
                        global_step=(ep - 1) * steps + num_batches
                    )


class DualSSLWithinDomainAllDistMatchingModel(DualSSLWithinDomainDistMatchingModelBase):
    def __init__(self, **kwargs):
        self.use_ot = kwargs['use_ot']
        self.asymmetric = kwargs["asymmetric"]
        self.div_alpha_schedule = kwargs["div_alpha_schedule"]
        self.div_alpha_start = kwargs["div_alpha_start"]
        self.div_alpha = kwargs["div_alpha"]
        super().__init__(**kwargs)

    def get_div_alpha(self, step, steps, ep, ep_total):
        if ep <= self.div_alpha_start:
            frac = 0.0
        else:
            total_steps = steps * (ep_total - self.div_alpha_start)
            curr_step = steps * (ep - 1 - self.div_alpha_start) + step
            if self.div_alpha_schedule == "fixed":
                frac = 1.0
            elif self.div_alpha_schedule == "cosine":
                frac = 1 - 0.5 * (1 + np.cos(curr_step * np.pi / total_steps))
            else:
                frac = curr_step / total_steps
        return self.div_alpha * frac

    def calculate_domain_dist_matching_loss(self, src_feats, trgt_feats):
        src_feats_dists = self.get_distance(src_feats, metric=self.div_metric)
        if self.asymmetric:
            src_feats_dists = src_feats_dists.clone().detach()

        trgt_feats_dists = self.get_distance(trgt_feats, metric=self.div_metric)

        src_feats_dists_reshaped = torch.reshape(src_feats_dists, (-1, 1))
        trgt_feats_dists_reshaped = torch.reshape(trgt_feats_dists, (-1, 1))
        if self.use_ot:
            div_dist_loss = self.emd_dist_loss(src_feats_dists_reshaped, trgt_feats_dists_reshaped)
        else:
            if self.ind_mmd_loss:
                div_dist_loss = self.mmd_dist_loss(src_feats_dists, trgt_feats_dists)
            else:
                div_dist_loss = self.mmd_dist_loss((src_feats_dists_reshaped,), (trgt_feats_dists_reshaped,))
        return div_dist_loss


class DualSSLWithinDomainAllDistNormMatchingModel(DualSSLWithinDomainAllDistMatchingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_domain_dist_matching_loss(self, src_feats, trgt_feats):
        src_feats_dists = self.get_distance(src_feats, metric=self.div_metric)
        if self.asymmetric:
            src_feats_dists = src_feats_dists.clone().detach()

        trgt_feats_dists = self.get_distance(trgt_feats, metric=self.div_metric)

        src_feats_dists_reshaped = torch.reshape(src_feats_dists, (-1, 1))
        trgt_feats_dists_reshaped = torch.reshape(trgt_feats_dists, (-1, 1))
        num_samples = src_feats_dists_reshaped.shape[0] // 4

        src_sample_idxs = torch.randperm(src_feats_dists_reshaped.size(0))[:num_samples]
        src_samples = src_feats_dists_reshaped[src_sample_idxs]
        trgt_sample_idxs = torch.randperm(trgt_feats_dists_reshaped.size(0))[:num_samples]
        trgt_samples = trgt_feats_dists_reshaped[trgt_sample_idxs]
        with torch.no_grad():
            src_feats_dists_min = torch.min(src_samples)
            trgt_feats_dists_min = torch.min(trgt_samples)
        src_samples -= src_feats_dists_min
        trgt_samples -= trgt_feats_dists_min
        if self.use_ot:
            div_dist_loss = self.emd_dist_loss(src_samples, trgt_samples)
        else:
            if self.ind_mmd_loss:
                div_dist_loss = self.mmd_dist_loss(src_feats_dists, trgt_feats_dists)
            else:
                div_dist_loss = self.mmd_dist_loss((src_samples,), (trgt_samples,))
        return div_dist_loss



class DualSSLWithinDomainSplitDistMatchingModelBase(DualSSLWithinDomainDistMatchingModelBase):
    def __init__(self, **kwargs):
        self.num_near_images = kwargs['num_near_images']
        self.num_far_images = kwargs['num_far_images']
        super().__init__(**kwargs)

    def track_domain_dist_matching_loss(self, src_feats, trgt_feats, tracking_vars, tb_scalar_dicts, logger_strs):
        near_k, far_k = self.num_near_images, self.num_far_images
        with torch.no_grad():
            src_feats_dists = self.get_distance(src_feats, metric=self.div_metric)
            trgt_feats_dists = self.get_distance(trgt_feats, metric=self.div_metric)

            if self.div_metric == "euclidean":
                src_closest_dists = torch.topk(src_feats_dists, k=near_k+1, largest=False).values[:, 1:]
                src_farthest_dists = torch.topk(src_feats_dists, k=far_k, largest=True).values[:, 0:]
                trgt_closest_dists = torch.topk(trgt_feats_dists, k=near_k+1, largest=False).values[:, 1:]
                trgt_farthest_dists = torch.topk(trgt_feats_dists, k=far_k, largest=True).values[:, 0:]
            else:
                src_closest_dists = torch.topk(src_feats_dists, k=near_k+1, largest=True).values[:, 1:]
                src_farthest_dists = torch.topk(src_feats_dists, k=far_k, largest=False).values[:, 0:]
                trgt_closest_dists = torch.topk(trgt_feats_dists, k=near_k+1, largest=True).values[:, 1:]
                trgt_farthest_dists = torch.topk(trgt_feats_dists, k=far_k, largest=False).values[:, 0:]

            src_closest_dists = torch.reshape(src_closest_dists, (-1, 1))
            src_farthest_dists = torch.reshape(src_farthest_dists, (-1, 1))
            trgt_closest_dists = torch.reshape(trgt_closest_dists, (-1, 1))
            trgt_farthest_dists = torch.reshape(trgt_farthest_dists, (-1, 1))
            # print(src_closest_dists.shape, src_farthest_dists.shape, trgt_closest_dists.shape,
            #       trgt_farthest_dists.shape)

            closest_div_dist_emd_loss = self.emd_dist_loss(src_closest_dists, trgt_closest_dists)
            farthest_div_dist_emd_loss = self.emd_dist_loss(src_farthest_dists, trgt_farthest_dists)

            closest_div_dist_mmd_loss = self.mmd_dist_loss((src_closest_dists,), (trgt_closest_dists,))
            farthest_div_dist_mmd_loss = self.mmd_dist_loss((src_farthest_dists,), (trgt_farthest_dists,))

            tracking_vars["closest_div_loss_emd_total"] += closest_div_dist_emd_loss.item()
            tracking_vars["farthest_div_loss_emd_total"] += farthest_div_dist_emd_loss.item()
            tracking_vars["closest_div_loss_mmd_total"] += closest_div_dist_mmd_loss.item()
            tracking_vars["farthest_div_loss_mmd_total"] += farthest_div_dist_mmd_loss.item()
            tracking_vars["closest_src_dist_total"] += torch.mean(src_closest_dists).item()
            tracking_vars["farthest_src_dist_total"] += torch.mean(src_farthest_dists).item()
            tracking_vars["closest_trgt_dist_total"] += torch.mean(trgt_closest_dists).item()
            tracking_vars["farthest_trgt_dist_total"] += torch.mean(trgt_farthest_dists).item()
            tracking_vars["num_batches"] += 1

        logger_strs.append(f"EMD1:{tracking_vars['closest_div_loss_emd_total'] / tracking_vars['num_batches']:.3f} "
                           f"EMD2:{tracking_vars['farthest_div_loss_emd_total'] / tracking_vars['num_batches']:.3f} "
                           f"MMD1:{tracking_vars['closest_div_loss_mmd_total'] / tracking_vars['num_batches']:.3f} "
                           f"MMD2:{tracking_vars['farthest_div_loss_mmd_total'] / tracking_vars['num_batches']:.3f}"
                           # f"D:[{tracking_vars['closest_src_dist_total'] / tracking_vars['num_batches']:.2f} "
                           # f"{tracking_vars['closest_trgt_dist_total'] / tracking_vars['num_batches']:.2f} "
                           # f"{tracking_vars['farthest_src_dist_total'] / tracking_vars['num_batches']:.2f} "
                           # f"{tracking_vars['farthest_trgt_dist_total'] / tracking_vars['num_batches']:.2f}]"
                           )
        tb_scalar_dicts["dom_dist_loss"] = {
            'closest_div_loss_emd_ep': tracking_vars["closest_div_loss_emd_total"] / tracking_vars["num_batches"],
            'closest_div_loss_emd_batch': closest_div_dist_emd_loss.item(),
            'farthest_div_loss_emd_ep': tracking_vars["farthest_div_loss_emd_total"] / tracking_vars["num_batches"],
            'farthest_div_loss_emd_batch': farthest_div_dist_emd_loss.item(),
            'closest_div_loss_mmd_ep': tracking_vars["closest_div_loss_mmd_total"] / tracking_vars["num_batches"],
            'closest_div_loss_mmd_batch': closest_div_dist_mmd_loss.item(),
            'farthest_div_loss_mmd_ep': tracking_vars["farthest_div_loss_mmd_total"] / tracking_vars["num_batches"],
            'farthest_div_loss_mmd_batch': farthest_div_dist_mmd_loss.item(),
        }
        tb_scalar_dicts["closest_queue_dist"] = {
            "src_closest_dist": tracking_vars["closest_src_dist_total"] / tracking_vars["num_batches"],
            "src_farthest_dist": tracking_vars["farthest_src_dist_total"] / tracking_vars["num_batches"],
            "trgt_closest_dist": tracking_vars["closest_trgt_dist_total"] / tracking_vars["num_batches"],
            "trgt_farthest_dist": tracking_vars["farthest_trgt_dist_total"] / tracking_vars["num_batches"],
        }

    def calculate_domain_dist_matching_loss(self, src_feats, trgt_feats):
        return torch.tensor(0.0, device=src_feats.device, dtype=src_feats.dtype)


class DualSSLWithinDomainSplitDistMatchingModel(DualSSLWithinDomainSplitDistMatchingModelBase):
    def __init__(self, **kwargs):
        self.use_ot = kwargs['use_ot']
        self.asymmetric = kwargs["asymmetric"]
        self.div_alpha_schedule = kwargs["div_alpha_schedule"]
        self.div_alpha_start = kwargs["div_alpha_start"]
        self.div_alpha = kwargs["div_alpha"]
        self.split_div_type = kwargs["split_div_type"]
        assert self.split_div_type.lower() in ["farthest", "closest", "both"]
        super().__init__(**kwargs)

    def get_div_alpha(self, step, steps, ep, ep_total):
        if ep <= self.div_alpha_start:
            frac = 0.0
        else:
            total_steps = steps * (ep_total - self.div_alpha_start)
            curr_step = steps * (ep - 1 - self.div_alpha_start) + step
            if self.div_alpha_schedule == "fixed":
                frac = 1.0
            elif self.div_alpha_schedule == "cosine":
                frac = 1 - 0.5 * (1 + np.cos(curr_step * np.pi / total_steps))
            else:
                frac = curr_step / total_steps
        return self.div_alpha * frac

    def calculate_domain_dist_matching_loss(self, src_feats, trgt_feats):
        near_k, far_k = self.num_near_images, self.num_far_images
        src_feats_dists = self.get_distance(src_feats, metric=self.div_metric)
        trgt_feats_dists = self.get_distance(trgt_feats, metric=self.div_metric)

        if self.div_metric == "euclidean":
            src_closest_dists = torch.topk(src_feats_dists, k=near_k+1, largest=False).values[:, 1:]
            src_farthest_dists = torch.topk(src_feats_dists, k=far_k, largest=True).values[:, 0:]
            trgt_closest_dists = torch.topk(trgt_feats_dists, k=near_k+1, largest=False).values[:, 1:]
            trgt_farthest_dists = torch.topk(trgt_feats_dists, k=far_k, largest=True).values[:, 0:]
        else:
            src_closest_dists = torch.topk(src_feats_dists, k=near_k+1, largest=True).values[:, 1:]
            src_farthest_dists = torch.topk(src_feats_dists, k=far_k, largest=False).values[:, 0:]
            trgt_closest_dists = torch.topk(trgt_feats_dists, k=near_k+1, largest=True).values[:, 1:]
            trgt_farthest_dists = torch.topk(trgt_feats_dists, k=far_k, largest=False).values[:, 0:]

        if self.asymmetric:
            src_closest_dists = torch.reshape(src_closest_dists, (-1, 1)).clone().detach()
            src_farthest_dists = torch.reshape(src_farthest_dists, (-1, 1)).clone().detach()
        else:
            src_closest_dists = torch.reshape(src_closest_dists, (-1, 1))
            src_farthest_dists = torch.reshape(src_farthest_dists, (-1, 1))
        trgt_closest_dists = torch.reshape(trgt_closest_dists, (-1, 1))
        trgt_farthest_dists = torch.reshape(trgt_farthest_dists, (-1, 1))
        # print(src_closest_dists.shape, trgt_closest_dists.shape)

        closest_div_dist_loss = torch.tensor([0.0], dtype=trgt_closest_dists.dtype).cuda()
        farthest_div_dist_loss = torch.tensor([0.0], dtype=trgt_closest_dists.dtype).cuda()

        if self.split_div_type != "farthest":
            if self.use_ot:
                closest_div_dist_loss = self.emd_dist_loss(src_closest_dists, trgt_closest_dists)
            else:
                closest_div_dist_loss = self.mmd_dist_loss((src_closest_dists,), (trgt_closest_dists,))
        if self.split_div_type != "closest":
            if self.use_ot:
                farthest_div_dist_loss = self.emd_dist_loss(src_farthest_dists, trgt_farthest_dists)
            else:
                farthest_div_dist_loss = self.mmd_dist_loss((src_farthest_dists,), (trgt_farthest_dists,))

        return closest_div_dist_loss + farthest_div_dist_loss
