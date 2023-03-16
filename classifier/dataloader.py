# --------------------------------------------------------------------------
#                         Copyright © by
#           Ignitarium Technology Solutions Pvt. Ltd.
#                         All rights reserved.
#  This file contains confidential information that is proprietary to
#  Ignitarium Technology Solutions Pvt. Ltd. Distribution,
#  disclosure or reproduction of this file in part or whole is strictly
#  prohibited without prior written consent from Ignitarium.
# --------------------------------------------------------------------------
#  Filename    : dataloader.py
# --------------------------------------------------------------------------
#  Description :
# --------------------------------------------------------------------------
"""
get dataloader for train, val and test.
"""
import os
from glob import glob
import torchvision
import torch

from torch.utils.data import Dataset, WeightedRandomSampler
import cv2

class CustomDataset(Dataset):
    """
    Class for creating custom dataloader
    Args:
        data_path: path of directory contains images
        transformation: image transformation/augmentations
    Returns:
        image: transformed image
        label: ground truth target value
    """
    def __init__(self, data_path, transformation):
        self.data_path = data_path
        self.transform = transformation
        self.classes = sorted(os.listdir(self.data_path))
        self.idx_to_class = dict(enumerate(self.classes))
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}
        self.images = glob(os.path.join(self.data_path, "*/*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filepath = self.images[idx]
        image = cv2.imread(image_filepath)
        image = self.transform(image)

        label = image_filepath.split("/")[-2]
        label = self.class_to_idx[label]

        return image, label

def get_image_preprocess_transforms(image_size):
    """
    Helper function for preprocessing in data
    Return:
        preprocess: preprocessing transformation which applied on image data.
    """
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(),
        ]
    )
    return preprocess

def get_data_augmentation_preprocess(trans):
    """
    Helper function to apply data augmentation:
    Args:
        trans: transformed data
    Return:
        data_augmented_transforms: transformation after applying various data
                                   augmentation on image
    """
    data_augmented_transforms = torchvision.transforms.Compose(
        [
            trans,
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(20, fill=(0, 0, 0)),
        ]
    )
    return data_augmented_transforms

def get_dataloader(dataset, batch_size, shuffle=False, sampler=None):
    """
    Helper function to create dataloader
    Args:
        dataset: object of custom dataset class
        batch_size: batch size for data loader
        shuffle: if images needs to be shuffle
        sampler: weighted random sampler for handling class imbalance
    Return:
        loader: Dataloader object
    """
    if sampler is not None:
        loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        sampler=sampler)
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=shuffle)
    return loader

def get_data_generators(root_path, image_size, class_imbalance_method, batch_size):
    """
    Helper function which creates train, val, and test data loaders
    """
    image_transforms = get_image_preprocess_transforms(image_size)
    aug_transform = get_data_augmentation_preprocess(image_transforms)

    # prepare train dataloader
    train_data_path = os.path.join(root_path, "train")
    train_dataset = CustomDataset(
        data_path=train_data_path,
        transformation=aug_transform
    )
    sampler=None
    class_weights = []
    if class_imbalance_method in ("upsampler", "class_weight"):

        # loop through each subdirectory and calculate the class weight
        # that is 1 / len(files) in that subdirectory
        classes = train_dataset.classes
        for cls in classes:
            files = os.listdir(os.path.join(train_data_path, cls))
            class_weights.append(1 / len(files))

        if class_imbalance_method == "upsampler":
            sample_weights = [0] * len(train_dataset)

            for idx, (data, label) in enumerate(train_dataset):
                class_weight = class_weights[label]
                sample_weights[idx] = class_weight
            num_samples = len(train_dataset)

            sampler = WeightedRandomSampler(
                sample_weights, num_samples=num_samples, replacement=True
            )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler
    )

    # prepare val dataloader
    val_data_path = os.path.join(root_path, "val")
    val_dataset = CustomDataset(
        data_path=val_data_path,
        transformation=image_transforms
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # prepare test dataloader
    test_data_path = os.path.join(root_path, "test")
    test_dataset = CustomDataset(
        data_path=test_data_path,
        transformation=image_transforms
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, class_weights
