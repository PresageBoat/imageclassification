#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from core.config import cfg
from datasets.imagenet import ImageNet
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


# Supported datasets
_DATASETS = {"imagenet": ImageNet}

# Default data directory (/path/pycls/pycls/datasets/data)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Relative data paths to default data directory
_PATHS = {"imagenet": "imagenet"}


def _construct_loader(dataset,split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # Construct the dataset
    dataset = _DATASETS["imagenet"](dataset,split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if len(cfg.GPU_DEVICE_IDS) > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / len(cfg.GPU_DEVICE_IDS)),
        shuffle=True,
        drop_last=True,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / len(cfg.GPU_DEVICE_IDS)),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """ "Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
