"""Module containing the SSL models"""
import torch
import torch.utils.tensorboard as tb
import torch.nn as nn

import time

import utils


class SimCLRModel:
    """Class definition for the SimCLR model"""
    def __init__(self, network, source_loader, target_loader, temp, logger, no_save=False):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.temp = temp
        self.logger = logger
        self.no_save = no_save
        self.network.cuda()
        
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train the model for one epoch"""
        self.network.unfreeze_student()
        self.network.student_backbone.train()
        self.network.student_projection.train()
        num_batches = 0
        src_info_loss_total = 0.0
        trgt_info_loss_total = 0.0
        sum_info_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            src_idx, src_images = self.source_loader.get_next_batch()
            trgt_idx, trgt_images = self.target_loader.get_next_batch()
            src_images = torch.cat(src_images, dim=0)
            trgt_images = torch.cat(trgt_images, dim=0)
            all_images = torch.cat([src_images, trgt_images], dim=0)
            all_images = all_images.cuda()
            all_feats = self.network.student_forward(all_images)
            
            src_size = src_images.size(0)
            src_feats, trgt_feats = all_feats[:src_size], all_feats[src_size:]
            src_logits, src_labels = utils.info_nce_loss(src_feats, temp=self.temp)
            src_loss = criterion(src_logits, src_labels)
            trgt_logits, trgt_labels = utils.info_nce_loss(trgt_feats, temp=self.temp)
            trgt_loss = criterion(trgt_logits, trgt_labels)
            
            total_loss = src_loss + trgt_loss
            
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            num_batches += 1
            src_info_loss_total += src_loss.item()
            trgt_info_loss_total += trgt_loss.item()
            sum_info_loss_total += total_loss.item()
            
            self.network.update_beta(step=(ep-1)*steps+num_batches, total_steps=ep_total*steps)
            self.network.update_teacher_weights()
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  LR: {:.3f}  "
                                 "SL: {:.3f}  TL: {:.3f}  T: {:.2f}s"
                                 .format(ep, ep_total, step, steps, optimizer.param_groups[0]['lr'],
                                         src_info_loss_total / num_batches, trgt_info_loss_total / num_batches,
                                         time.time() - start_time))
                
            if not self.no_save:
                writer.add_scalars(
                    main_tag='training_loss',
                    tag_scalar_dict={
                        'src_info_loss_ep': src_info_loss_total / num_batches,
                        'src_info_loss_batch': src_loss.item(),
                        'trgt_info_loss_ep': trgt_info_loss_total / num_batches,
                        'trgt_info_loss_batch': trgt_loss.item(),
                        'total_loss_ep': sum_info_loss_total / num_batches,
                        'total_loss_batch': total_loss.item(),
                    },
                    global_step=(ep-1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag='training_details',
                    tag_scalar_dict={
                        'lr': optimizer.param_groups[0]['lr'],
                        'beta': self.network.beta
                    },
                    global_step=(ep-1) * steps + num_batches,
                    
                )

        self.logger.info("Ep: {}/{}  Step: {}/{}  LR: {:.3f}  SL: {:.3f}  TL: {:.3f}  T: {:.2f}s"
                         .format(ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'],
                                 src_info_loss_total / num_batches, trgt_info_loss_total / num_batches,
                                 time.time() - start_time))
                