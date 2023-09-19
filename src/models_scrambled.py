"""Module containing models with pairwise class information for target dataset"""
import torch
import torch.nn as nn
import time
import torch.utils.tensorboard as tb
import torch.nn.functional as func

import tllib.alignment.jan as jan
import tllib.modules.kernels as kernels

import utils


class DualSupModelWithScrambledTargetClasses:
    """Module implementing the combined supervised pretraining on source and target"""
    
    def __init__(self, network, source_loader, target_loader, combined_batch, logger, scrambler_seed=None):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.combined_batch = combined_batch
        self.network.cuda()
        
        self.scrambler_seed = scrambler_seed if scrambler_seed is not None else 42
        import scramble_labels
        self.scrambler = scramble_labels.RandomToyboxScrambler(seed=self.scrambler_seed)
    
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
                _, _, src_logits, trgt_logits = self.network.forward(comb_images)
            else:
                _, src_logits = self.network.forward_1(src_images)
                _, trgt_logits = self.network.forward_2(trgt_images)
            
            scrambled_trgt_labels = self.scrambler.scramble(trgt_labels)
            
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
                if "in12" in loader_names[idx]:
                    scrambled_labels = self.scrambler.scramble(labels)
                else:
                    scrambled_labels = labels.clone()
                with torch.no_grad():
                    _, logits = self.network.forward_2(images) if "in12" in loader_names[idx] \
                        else self.network.forward_1(images)
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
    
    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        n_total_2 = 0
        n_correct_2 = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            scrambled_labels = self.scrambler.scramble(labels)
            with torch.no_grad():
                _, logits_1 = self.network.forward_1(images)
                _, logits_2 = self.network.forward_2(images)
            top, pred = utils.calc_accuracy(output=logits_1, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]

            top_2, pred_2 = utils.calc_accuracy(output=logits_2, target=scrambled_labels, topk=(1,))
            n_correct_2 += top_2[0].item() * pred_2.shape[0]
            n_total_2 += pred_2.shape[0]
        acc = n_correct / n_total
        acc_2 = n_correct_2 / n_total_2
        return round(acc, 2), round(acc_2, 2)

