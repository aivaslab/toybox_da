"""
Module for project utils
"""
import torch
import argparse
import logging
import torch.nn.functional as func
import torch.nn as nn
from typing import Tuple
from PIL import Image


COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}
LOG_FORMAT_TERMINAL = '%(asctime)s:' + COLOR['GREEN'] + '%(filename)s' + COLOR['ENDC'] + ':%(lineno)s:' + COLOR['RED'] \
                      + '%(levelname)s' + COLOR['ENDC'] + ': %(message)s'
LOG_FORMAT_FILE = '%(asctime)s:%(filename)s:%(lineno)s:%(levelname)s:%(message)s'


def create_logger(log_level_str: str, log_file_name: str, no_save: bool = False):
    """Create and return logger"""
    log_level = getattr(logging, log_level_str.upper())
    logging.basicConfig(format=LOG_FORMAT_TERMINAL, level=log_level)
    logger = logging.getLogger(__name__)
    if not no_save:
        logfile_handler = logging.FileHandler(log_file_name)
        logging_formatter = logging.Formatter(LOG_FORMAT_FILE)
        logfile_handler.setFormatter(logging_formatter)
        logger.addHandler(logfile_handler)
    return logger


def save_args(path, args):
    """Save the experiment args in json file"""
    import json
    json_str = json.dumps(args)
    out_file = open(path + "exp_args.json", "w")
    out_file.write(json_str)
    out_file.close()


class ForeverDataLoader:
    """Class that returns the next batch of data"""
    
    def __init__(self, loader):
        self.loader = loader
        self.loader_iter = iter(self.loader)
    
    def get_next_batch(self):
        """Return next_batch"""
        try:
            ret = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            ret = next(self.loader_iter)
        
        return ret


def decoupled_contrastive_loss(features, temp):
    """Implement the decoupled contrastive loss
    https://arxiv.org/pdf/2110.06848"""
    dev = torch.device('cuda:0')
    batch_size = features.shape[0] / 2
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dev)
    features = func.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(dev)
    labels = labels[~mask].view(labels.shape[0], -1).type(torch.uint8)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    positives = positives / temp
    negatives = negatives / temp

    pos_loss = -positives
    neg_loss = torch.logsumexp(negatives, dim=1, keepdim=True)
    total_loss = (pos_loss + neg_loss).mean()

    return total_loss


def info_nce_loss(features, temp):
    """Implement the info_nce loss"""
    dev = torch.device('cuda:0')
    batch_size = features.shape[0] / 2
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(dev)
    features = func.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1))

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(dev)
    labels = labels[~mask].view(labels.shape[0], -1).type(torch.uint8)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dev)
    logits = logits / temp
    return logits, labels


def sup_con_loss(features, labels, temp):
    """Method for calculating the supervised contrastive loss"""
    device = torch.device('cuda:0')
    same_cl_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    same_cl_matrix = same_cl_matrix.to(device)
    features = func.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, torch.transpose(features, 0, 1)).to(device)
    # print(similarity_matrix)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    same_cl_matrix = same_cl_matrix[~mask].view(same_cl_matrix.shape[0], -1).type(torch.uint8)
    # print(same_cl_matrix, "same_cls")
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # print(similarity_matrix, "without diagonal")
    similarity_matrix /= temp
    denom_logsum = torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
    # print(denom_logsum, "denom_logsum")
    all_loss_ratio = similarity_matrix - denom_logsum
    # print(all_loss_ratio, "loss_ratios for all items")
    positive_loss_ratio = same_cl_matrix * all_loss_ratio
    # print(positive_loss_ratio, "loss ratios for positive items")
    positive_count = torch.sum(same_cl_matrix, dim=1)
    # print(positive_count)
    positives_loss_sum = torch.sum(positive_loss_ratio, dim=1)
    positives_loss_ave = torch.nan_to_num(torch.divide(positives_loss_sum, positive_count), nan=0)
    return -torch.mean(positives_loss_ave)


class OrientationLossV1(nn.Module):
    def __init__(self, num_images_in_batch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_images_in_batch = num_images_in_batch
        self.mask = self.get_mask()

    def get_mask(self):
        bsize = self.num_images_in_batch // 4
        base_mask = -1 * torch.eye(bsize) + 1 - torch.eye(bsize)
        mask_rep_y = torch.repeat_interleave(base_mask, 4, dim=1)
        mask_rep_x = torch.repeat_interleave(mask_rep_y, 4, dim=0)

        return mask_rep_x.unsqueeze(0).cuda()

    def forward(self, src_feats, trgt_feats):
        # print(src_feats.shape, trgt_feats.shape)
        dist_matrix = trgt_feats.unsqueeze(1) - src_feats.unsqueeze(0)
        # print(dist_matrix.shape)
        # vec_sim = torch.mul(dist_matrix.unsqueeze(1), dist_matrix.unsqueeze(2)).sum(dim=-1)
        vec_sim = func.cosine_similarity(dist_matrix.unsqueeze(1), dist_matrix.unsqueeze(2), dim=-1)
        mask = self.mask
        # print(mask.shape)
        loss = torch.mul(vec_sim, mask).sum()
        # print(loss.shape, loss)
        return loss / (src_feats.size(0) * src_feats.size(0) * trgt_feats.size(0))
        
        
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
        
def partial_mixup(input: torch.Tensor,
                  gamma: float,
                  indices: torch.Tensor
                  ) -> torch.Tensor:
    """Util method for mixup regularization"""
    if input.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = input[indices]
    return input.mul(gamma).add(perm_input, alpha=1 - gamma)


def mixup(input: torch.Tensor,
          target: torch.Tensor,
          gamma: float,
          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Do mixup"""
    indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
    return partial_mixup(input, gamma, indices), partial_mixup(target, gamma, indices)


def calc_accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batchSize = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        target_reshaped = torch.reshape(target, (1, -1)).repeat(maxk, 1)
        correct_top_k = torch.eq(pred, target_reshaped)
        pred_1 = pred[0]
        res = []
        for k in topk:
            correct_k = correct_top_k[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(torch.mul(correct_k, 100.0 / batchSize))
        return res, pred_1


def get_parser():
    """
    Use argparse to get parser
    """
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--log-level", "-l", default="warning", type=str)
    parser.add_argument("--config-file", "-conf", default="./default_config.yaml", type=str)
    parser.add_argument("--seed", "-s", default=0, type=int)
    return parser.parse_args()


def try_assert(fn, arg, msg):
    """
    Util method for assert followed by exception
    """
    try:
        assert fn(arg), msg
    except AssertionError:
        logging.error(msg)
        raise


def weights_init(m):
    """
    Reset weights of the network
    """
    import torch.nn as nn
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    
    for _, data, _ in loader:
        # print(data.shape)
        if isinstance(data, list):
            data = torch.concatenate(data)
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        
        cnt += nb_pixels
    
    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


def concat_images_horizontally(images, gap=5):
    """Concatenate images in the list horizontally"""
    num_images = len(images)
    h, w = images[0].height, images[1].width
    height, width = h, num_images * w + (num_images + 1) * gap
    dst = Image.new('RGB', (width, height), (0, 0, 0))
    for idx, image in enumerate(images):
        dst.paste(image, ((idx + 1) * gap + idx * w, 0))

    return dst


def concat_images_vertically(images, gap=5):
    """Concatenate images in the list vertically"""
    num_images = len(images)
    h, w = images[0].height, images[1].width
    height, width = num_images * h + (num_images + 1) * gap, w
    dst = Image.new('RGB', (width, height), (0, 0, 0))
    for idx, image in enumerate(images):
        dst.paste(image, (0, (idx + 1) * gap + idx * h))

    return dst


def get_images(images, mean, std, aug=True):
    """
    Return images from the tensor provided.
    Inverse Normalize the images if aug is True
    """
    import torchvision.transforms as transforms
    trnsfrms = []
    if aug:
        trnsfrms.append(transforms.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]))
    import torchvision.transforms as transforms
    
    trnsfrms.append(transforms.ToPILImage())
    tr = transforms.Compose(trnsfrms)
    ims1 = []
    for img_idx in range(len(images)):
        pil_img1 = tr(images[img_idx])
        ims1.append(pil_img1)
    
    def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
        """hconcat images"""
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    def get_concat_h_multi_blank(im_list):
        """sdfs"""
        _im = im_list.pop(0)
        for im in im_list:
            _im = get_concat_h_blank(_im, im)
        return _im
    
    def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
        """hconcat images"""
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst
    
    def get_concat_v_multi_blank(im_list):
        """sdfs"""
        _im = im_list.pop(0)
        for im in im_list:
            _im = get_concat_v_blank(_im, im)
        return _im
    
    hor_images = []
    for i in range(len(ims1) // 8):
        if len(ims1[i * 8:min(i * 8 + 8, len(ims1))]) == 8:
            hor_images.append(get_concat_h_multi_blank(ims1[i * 8:min(i * 8 + 8, len(ims1))]))
    
    return get_concat_v_multi_blank(hor_images)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.6f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
