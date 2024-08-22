"""
Module containing different models for evaluating trained models
"""
import time
import math

import torch
import torch.nn as nn

import utils


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


class ModelKNNEval:
    """model for knn eval of trained network"""

    def __init__(self, network, train_loader, logger, dim, num_classes=12):
        self.network = network
        self.train_loader = train_loader
        self.logger = logger
        self.network.cuda()
        self.network.set_eval()
        self.dim = dim
        self.train_activations = None
        self.train_labels = None
        self.num_classes = num_classes

        self.train_size = len(self.train_loader.dataset)

    def train(self):
        for _, (idxs, images, labels) in enumerate(self.train_loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                feats = self.network.forward(images)
                if self.train_activations is None:
                    self.train_activations = feats.detach().clone()
                    self.train_labels = labels.detach().clone()
                else:
                    self.train_activations = torch.cat((self.train_activations, feats.detach().clone()), dim=0)
                    self.train_labels = torch.cat((self.train_labels, labels.detach().clone()), dim=0)
        # print(self.train_activations.shape, self.train_labels.size())
        assert self.train_activations.shape == (self.train_size, self.dim)
        assert self.train_labels.size(0) == self.train_size

    def eval(self, loader, num_neighbors, use_cosine):
        """Evaluate the model on the provided dataloader"""
        eval_activations = None
        eval_labels = None
        eval_size = len(loader.dataset)
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                feats = self.network.forward(images)
                if eval_activations is None:
                    eval_activations = feats.detach().clone()
                    eval_labels = labels.detach().clone()
                else:
                    eval_activations = torch.cat((eval_activations, feats.detach().clone()), dim=0)
                    eval_labels = torch.cat((eval_labels, labels.detach().clone()), dim=0)
        assert eval_activations.shape == (eval_size, self.dim)
        assert eval_labels.shape[0] == eval_size

        distance_matrix = torch.zeros((eval_size, self.train_size), dtype=torch.float16)
        chunk_size = 1000
        n_train_chunks, n_eval_chunks = math.ceil(self.train_size / chunk_size), math.ceil(eval_size / chunk_size)
        train_chunks = torch.chunk(self.train_activations, n_train_chunks, dim=0)
        eval_chunks = torch.chunk(eval_activations, n_eval_chunks, dim=0)

        train_chunk_size = train_chunks[0].size(0)
        eval_chunk_size = eval_chunks[0].size(0)
        for train_chunk_idx, train_chunk in enumerate(train_chunks):
            for eval_chunk_idx, eval_chunk in enumerate(eval_chunks):
                if use_cosine:
                    dist_mat = torch.matmul(eval_chunk, train_chunk.transpose(0, 1))
                else:
                    dist_mat = ((eval_chunk.unsqueeze(1) - train_chunk.unsqueeze(0)) ** 2).sum(-1)
                distance_matrix[
                    eval_chunk_idx * eval_chunk_size:
                        min(eval_chunk_idx * eval_chunk_size + eval_chunk_size, eval_size),
                    train_chunk_idx*train_chunk_size:
                        min(train_chunk_idx*train_chunk_size + train_chunk_size, self.train_size)] \
                    = dist_mat
        accuracies_list = []
        for n_neighbor in num_neighbors:
            if use_cosine:
                topk_closest_images, topk_closest_indices = torch.topk(distance_matrix, k=n_neighbor, largest=True)
            else:
                topk_closest_images, topk_closest_indices = torch.topk(distance_matrix, k=n_neighbor, largest=False)
            topk_labels = self.train_labels[topk_closest_indices]

            preds = torch.mode(topk_labels, dim=1)
            num_equal = (preds.values == eval_labels).sum().item()
            acc = round(100 * num_equal / eval_size, 2)
            accuracies_list.append(acc)
        # print(accuracies_list)
        return accuracies_list

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
