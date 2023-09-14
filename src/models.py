"""Module implementing the MTL pretraining task for Toybox->IN-12"""
import time
import math

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
import torch.nn.functional as func

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


class SSLModel:
    """Module implementing the SSL method for pretraining the DA model"""
    
    def __init__(self, network, loader, logger):
        self.network = network
        self.loader = utils.ForeverDataLoader(loader)
        self.logger = logger
        self.network.cuda()
    
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


class SupModel:
    """Module implementing the supervised pretraining on source"""
    
    def __init__(self, network, source_loader, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.logger = logger
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
            src_logits = self.network.forward(src_images)
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


class DualSupModel:
    """Module implementing the combined supervised pretraining on source and target"""
    
    def __init__(self, network, source_loader, target_loader, combined_batch, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.combined_batch = combined_batch
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
                logits = self.network.forward(comb_images)
                src_size = src_images.shape[0]
                src_logits = logits[:src_size]
                trgt_logits = logits[src_size:]
            else:
                src_logits = self.network.forward(src_images)
                trgt_logits = self.network.forward(trgt_images)
            
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
                with torch.no_grad():
                    logits = self.network.forward(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
            losses.append(total_loss/num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
        writer.add_scalars(main_tag="val_loss",
                           tag_scalar_dict={
                               loader_names[0]: losses[0],
                               loader_names[1]: losses[1]
                           },
                           global_step=ep*steps)
    
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
                 scrambler_seed=None, lmbda=0.05):
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
        self.scrambler_seed = scrambler_seed if scrambler_seed else 42
        import scramble_labels
        self.scrambler = \
            scramble_labels.RandomToyboxScrambler(seed=self.scrambler_seed) if self.scramble_labels else None
        
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
            if self.scrambler is not None:
                scrambled_trgt_labels = self.scrambler.scramble(labels=trgt_labels)
            else:
                scrambled_trgt_labels = trgt_labels.clone()
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
