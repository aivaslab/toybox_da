import torchvision.transforms as transforms

import datasets


def get_ssl_transform_w_gblur(dset):
    """Train transform for SSL experiments with color jitter and gaussian blur"""
    assert dset in ["toybox", "in12"]
    if dset == "toybox":
        norm_transform = transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)
    else:
        norm_transform = transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)
    prob = 0.2
    color_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.8)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(hue=0.5)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(saturation=0.8)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(contrast=0.8)], p=prob),
                        transforms.RandomEqualize(p=prob),
                        transforms.RandomPosterize(bits=4, p=prob),
                        transforms.RandomAutocontrast(p=prob)
                        ]
    gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    all_transforms = transforms.Compose([transforms.ToPILImage(),
                                         transforms.RandomOrder(color_transforms),
                                         transforms.RandomGrayscale(p=0.2),
                                         transforms.RandomApply([gaussian_blur], p=0.2),
                                         transforms.Resize(256),
                                         transforms.RandomResizedCrop(size=224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         norm_transform,
                                         transforms.RandomErasing(p=0.5)])
    return all_transforms


def get_ssl_transform(dset):
    """Train transform for SSL experiments with color jitter"""
    assert dset in ["toybox", "in12"]
    if dset == "toybox":
        norm_transform = transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)
    else:
        norm_transform = transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)
    prob = 0.2
    color_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.8)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(hue=0.5)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(saturation=0.8)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(contrast=0.8)], p=prob),
                        transforms.RandomEqualize(p=prob),
                        transforms.RandomPosterize(bits=4, p=prob),
                        transforms.RandomAutocontrast(p=prob)
                        ]
    all_transforms = transforms.Compose([transforms.ToPILImage(),
                                         transforms.RandomOrder(color_transforms),
                                         transforms.RandomGrayscale(p=0.2),
                                         transforms.Resize(256),
                                         transforms.RandomResizedCrop(size=224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         norm_transform,
                                         transforms.RandomErasing(p=0.5)])
    return all_transforms


def get_eval_transform(dset):
    """Bare-bones eval transform for model evaluation and dataset testing"""
    assert dset in ["toybox", "in12"]
    if dset == "toybox":
        norm_transform = transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)
    else:
        norm_transform = transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)
    all_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform,
        ])
    return all_transforms
