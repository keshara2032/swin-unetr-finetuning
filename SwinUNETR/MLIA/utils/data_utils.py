# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import math
import os
from pathlib import Path
from typing import List, Tuple
from functools import partial

import numpy as np
import torch

from monai import data, transforms
from monai.data import ITKReader


def repeat_channels(x, target_channels: int):
    """
    Repeat single-channel input to the desired channel count if needed.
    """
    if x.shape[0] == target_channels:
        return x
    return np.repeat(x, target_channels, axis=0)


def add_depth(x, depth: int):
    """
    Ensure a depth dimension exists and matches the desired pseudo 3D depth.
    """
    if x.ndim == 3:
        x = np.expand_dims(x, -1)
    if x.shape[-1] == depth:
        return x
    return np.repeat(x, depth, axis=-1)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def _collect_mlia_pairs(split_root: Path) -> List[dict]:
    """
    Build paired file list for MLIA data layout.

    Expected structure:
    split_root/
        Brains/mri_<id>.mhd (+ .raw alongside)
        Labels/seg_<id>.mhd (+ .raw alongside)
    """
    brain_dir = split_root / "Brains"
    label_dir = split_root / "Labels"
    pairs = []
    for img_path in sorted(glob.glob(str(brain_dir / "mri_*.mhd"))):
        img_path = Path(img_path)
        case_id = img_path.stem.split("_")[-1]
        label_path = label_dir / f"seg_{case_id}.mhd"
        if label_path.exists():
            pairs.append({"image": str(img_path), "label": str(label_path), "case_id": case_id})
    return pairs


def build_mlia_splits(
    data_root: str, fold: int = 0, num_folds: int = 5, seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Deterministically split MLIA data into train/val folds.

    We shuffle all available pairs with a fixed seed, then hold out one fold
    (index ``fold``) for validation and use the remainder for training.
    """
    train_root = Path(data_root) / "Training"
    all_pairs = _collect_mlia_pairs(train_root)
    if len(all_pairs) == 0:
        raise RuntimeError(f"No MLIA pairs found under {train_root}")

    rng = np.random.RandomState(seed)
    rng.shuffle(all_pairs)

    fold = fold % max(1, num_folds)
    fold_sizes = [len(all_pairs) // num_folds] * num_folds
    for i in range(len(all_pairs) % num_folds):
        fold_sizes[i] += 1
    start = sum(fold_sizes[:fold])
    end = start + fold_sizes[fold]
    val = all_pairs[start:end]
    tr = all_pairs[:start] + all_pairs[end:]
    return tr, val


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list

    # If a json is provided and exists, respect it; otherwise build from MLIA layout.
    if datalist_json and os.path.exists(datalist_json):
        train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    else:
        train_files, validation_files = build_mlia_splits(
            data_root=data_dir, fold=args.fold, num_folds=args.num_folds, seed=args.seed
        )

    itk_reader = ITKReader()

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], reader=itk_reader),
            transforms.EnsureChannelFirstd(keys="image"),
            transforms.Lambdad(keys="image", func=partial(repeat_channels, target_channels=args.in_channels)),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.Lambdad(keys=["image", "label"], func=partial(add_depth, depth=args.pseudo_3d_depth)),
            transforms.Transposed(keys=["image", "label"], indices=[0, 3, 1, 2]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[args.roi_z, args.roi_x, args.roi_y],
                pos=1,
                neg=1,
                num_samples=1,
                image_threshold=0,
                allow_smaller=True,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.Lambdad(keys=["image", "label"], func=np.ascontiguousarray),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], reader=itk_reader),
            transforms.EnsureChannelFirstd(keys="image"),
            transforms.Lambdad(keys="image", func=partial(repeat_channels, target_channels=args.in_channels)),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.Lambdad(keys=["image", "label"], func=partial(add_depth, depth=args.pseudo_3d_depth)),
            transforms.Transposed(keys=["image", "label"], indices=[0, 3, 1, 2]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.Lambdad(keys=["image", "label"], func=np.ascontiguousarray),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], reader=itk_reader),
            transforms.EnsureChannelFirstd(keys="image"),
            transforms.Lambdad(keys="image", func=partial(repeat_channels, target_channels=args.in_channels)),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.Lambdad(keys=["image", "label"], func=partial(add_depth, depth=args.pseudo_3d_depth)),
            transforms.Transposed(keys=["image", "label"], indices=[0, 3, 1, 2]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.Lambdad(keys=["image", "label"], func=np.ascontiguousarray),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        if os.path.exists(Path(data_dir) / "Testing"):
            test_pairs = _collect_mlia_pairs(Path(data_dir) / "Testing")
            validation_files = test_pairs if len(test_pairs) > 0 else validation_files
        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
