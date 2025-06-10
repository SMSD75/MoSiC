import random
from PIL import Image
# Custom function to apply resizing and cropping with probability
import torchvision.transforms.functional as F
from torchvision import transforms
import torch
import torchvision.transforms as trn


def random_resize_crop(image, target, size=(256, 256), scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    ## convert target to tensor
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    image = F.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
    target = F.resized_crop(target, i, j, h, w, size, interpolation=Image.NEAREST)
    return image, target

def resize(image, target, size=(256, 256)):
    ## convert target to tensor
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    image = F.resize(image, size, interpolation=Image.BILINEAR)
    target = F.resize(target, size, interpolation=Image.NEAREST)
    return image, target


def apply_horizontal_flip(image, target):
    # Generate a random seed for the transformation
    if not isinstance(target, torch.Tensor):
        target = transforms.ToTensor()(target)
    seed = torch.randint(0, 2**32, size=(1,)).item()
    torch.manual_seed(seed)

    # Apply horizontal flip to the image
    image = F.hflip(image)

    # Use the same seed for the target to ensure consistent flip
    torch.manual_seed(seed)
    target = F.hflip(target)

    return image, target


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.5, 2), ratio=(3. / 4., 4. / 3.), probability=1.0):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.probability = probability

    def __call__(self, img, target):
        if random.random() < self.probability:
            return random_resize_crop(img, target, self.size, self.scale, self.ratio)
        return img, target

class RandomHorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, img, target):
        if random.random() < self.probability:
            return apply_horizontal_flip(img, target)
        return img, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Resize(object):
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, target):
        return resize(img, target, self.size)


class CombTransforms(object):
    def __init__(self, img_transform=None, tgt_transform=None, img_tgt_transform=None):
        self.img_transform = img_transform
        self.tgt_transform = tgt_transform
        self.img_tgt_transform = img_tgt_transform

    def __call__(self, img, tgt):
        if self.img_transform:
            img = self.img_transform(img)
        if self.tgt_transform:
            tgt = self.tgt_transform(tgt)
        if self.img_tgt_transform:
            return self.img_tgt_transform(img, tgt)
        else:
            return img, tgt



IMAGNET_MEAN = [0.485, 0.456, 0.406]
IMAGNET_STD = [0.229, 0.224, 0.255]

def get_hbird_train_transforms_for_imgs(input_size = 224,
        min_scale_factor = 0.5,
        max_scale_factor = 2.0,
        brightness_jitter_range = 0.1,
        contrast_jitter_range = 0.1,
        saturation_jitter_range = 0.1,
        hue_jitter_range = 0.1,
        brightness_jitter_probability = 0.5,
        contrast_jitter_probability = 0.5,
        saturation_jitter_probability = 0.5,
        hue_jitter_probability = 0.5,
        img_mean = IMAGNET_MEAN,
        img_std = IMAGNET_STD):


    # 1. Image transformations for training
    image_train_global_transforms = [trn.RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor))]
    image_train_local_transforms = [
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=img_mean, std=img_std)
    ]
    image_train_transform = trn.Compose([*image_train_global_transforms, *image_train_local_transforms])

    # 3. Return the transformations in dictionaries for training and validation
    train_transforms = {"img": image_train_transform, "target": None, "shared": None}
    return train_transforms

def get_hbird_transforms(input_size = 224,
        min_scale_factor = 0.5,
        max_scale_factor = 2.0,
        brightness_jitter_range = 0.1,
        contrast_jitter_range = 0.1,
        saturation_jitter_range = 0.1,
        hue_jitter_range = 0.1,
        brightness_jitter_probability = 0.5,
        contrast_jitter_probability = 0.5,
        saturation_jitter_probability = 0.5,
        hue_jitter_probability = 0.5,
        img_mean = IMAGNET_MEAN,
        img_std = IMAGNET_STD):
    # 1. Return the transformations in dictionaries for training, validation, and testing
    train_transforms = get_hbird_train_transforms(input_size, 
                                                    min_scale_factor, 
                                                    max_scale_factor, 
                                                    brightness_jitter_range, 
                                                    contrast_jitter_range, 
                                                    saturation_jitter_range, 
                                                    hue_jitter_range, 
                                                    brightness_jitter_probability, 
                                                    contrast_jitter_probability, 
                                                    saturation_jitter_probability, 
                                                    hue_jitter_probability, 
                                                    img_mean, 
                                                    img_std)
    val_transforms = get_hbird_val_transforms(input_size, img_mean, img_std)
    test_transforms = get_hbird_val_transforms(input_size, img_mean, img_std)
    return train_transforms, val_transforms, test_transforms

def get_hbird_train_transforms(input_size = 224,
        min_scale_factor = 0.5,
        max_scale_factor = 2.0,
        brightness_jitter_range = 0.1,
        contrast_jitter_range = 0.1,
        saturation_jitter_range = 0.1,
        hue_jitter_range = 0.1,
        brightness_jitter_probability = 0.5,
        contrast_jitter_probability = 0.5,
        saturation_jitter_probability = 0.5,
        hue_jitter_probability = 0.5,
        img_mean = IMAGNET_MEAN,
        img_std = IMAGNET_STD):

    # 1. Image transformations for training
    image_train_transform = trn.Compose([
        trn.RandomApply([trn.ColorJitter(brightness=brightness_jitter_range)], p=brightness_jitter_probability),
        trn.RandomApply([trn.ColorJitter(contrast=contrast_jitter_range)], p=contrast_jitter_probability),
        trn.RandomApply([trn.ColorJitter(saturation=saturation_jitter_range)], p=saturation_jitter_probability),
        trn.RandomApply([trn.ColorJitter(hue=hue_jitter_range)], p=hue_jitter_probability),
        trn.ToTensor(),
        trn.Normalize(mean=img_mean, std=img_std)
    ])

    # 2. Shared transformations for training
    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)),
        # RandomHorizontalFlip(probability=0.1),
    ])

    # 3. Return the transformations in dictionaries for training and validation
    train_transforms = {"img": image_train_transform, "target": None, "shared": shared_train_transform}
    return train_transforms


def get_hbird_val_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD):
    return get_default_val_transforms(input_size, img_mean, img_std)

def get_default_train_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD,
                    min_scale_factor = 0.5,
                    max_scale_factor = 2.0):
    # 1. Image transformations for training
    image_train_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize(mean=img_mean, std=img_std)
    ])

    # 2. Shared transformations for training
    shared_train_transform = Compose([
        RandomResizedCrop(size=(input_size, input_size), scale=(min_scale_factor, max_scale_factor)),
    ])
    # 3. Return the transformations in dictionary for training
    return {"img": image_train_transform, "target": None, "shared": shared_train_transform}

def get_default_val_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD):
    # 1. Image transformations for validation
    if img_mean is None or img_std is None:
        image_val_transform = trn.Compose([trn.Resize((input_size, input_size)), trn.ToTensor()])
    else:
        image_val_transform = trn.Compose([trn.Resize((input_size, input_size)), trn.ToTensor(), trn.Normalize(mean=img_mean, std=img_std)])

    # 2. Shared transformations for validation
    shared_val_transform = Compose([
        Resize(size=(input_size, input_size)),
    ])

    # 3. Return the transformations in a dictionary for validation
    val_transforms = {"img": image_val_transform, "target": None , "shared": shared_val_transform}
    return val_transforms

def get_default_transforms(input_size = 224,
                    img_mean = IMAGNET_MEAN,
                    img_std = IMAGNET_STD):
    # 1. Return the transformations in dictionaries for training, validation, and testing
    train_transforms = get_default_train_transforms(input_size, img_mean, img_std)
    val_transforms = get_default_val_transforms(input_size, img_mean, img_std)
    test_transforms = get_default_val_transforms(input_size, img_mean, img_std)
    return train_transforms, val_transforms, test_transforms