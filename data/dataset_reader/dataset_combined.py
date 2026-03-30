import os

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *


def _gather_loli_street(path):
    """Collect (low, high) image paths from LoLI_Street directory structure."""
    low_dir = os.path.join(path, 'low')
    high_dir = os.path.join(path, 'high')
    lows = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir) if not f.startswith('.')])
    highs = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir) if not f.startswith('.')])
    return lows, highs


def _gather_ve_lol_l(path, variant, split):
    """Collect (low, high) image paths from VE-LOL-L directory structure.
    variant: 'Cap' or 'Syn'
    split: 'train' or 'test'
    """
    low_dir = os.path.join(path, f'VE-LOL-L-{variant}-Low_{split}')
    normal_dir = os.path.join(path, f'VE-LOL-L-{variant}-Normal_{split}')
    lows = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir) if not f.startswith('.')])
    highs = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if not f.startswith('.')])
    return lows, highs


def main_dataset_combined(rank=0,
                          test_paths=None,
                          train_paths=None,
                          batch_size_train=4,
                          batch_size_test=1,
                          verbose=False,
                          cropsize=256,
                          flips=None,
                          num_workers=1,
                          crop_type='Random',
                          world_size=1):
    """
    Combined dataset loader for LoLI_Street + VE-LOL-L-CAP + VE-LOL-L-SYN.

    test_paths / train_paths: dict with keys like:
        {
            'LoLI_Street': '/path/to/lolistreet/Val',
            'VE_LOL_L_CAP': '/path/to/VE-LOL-L-Cap-Full',
            'VE_LOL_L_SYN': '/path/to/VE-LOL-L-Syn',
        }
    """
    # Mapping from dataset name to (gather_fn, extra_args_for_split)
    gatherers = {
        'LoLI_Street': lambda path, split: _gather_loli_street(path),
        'VE_LOL_L_CAP': lambda path, split: _gather_ve_lol_l(path, 'Cap', split),
        'VE_LOL_L_SYN': lambda path, split: _gather_ve_lol_l(path, 'Syn', split),
    }

    # --- Collect test paths ---
    all_low_valid, all_high_valid = [], []
    for ds_name, ds_path in test_paths.items():
        lows, highs = gatherers[ds_name](ds_path, 'test')
        all_low_valid.extend(lows)
        all_high_valid.extend(highs)
    check_paths([all_low_valid, all_high_valid])

    # --- Collect train paths ---
    all_low_train, all_high_train = [], []
    has_train = train_paths is not None
    if has_train:
        for ds_name, ds_path in train_paths.items():
            lows, highs = gatherers[ds_name](ds_path, 'train')
            all_low_train.extend(lows)
            all_high_train.extend(highs)
        check_paths([all_low_train, all_high_train])

    if verbose:
        print('Combined dataset images:')
        print(f"    -Validation low: {len(all_low_valid)}, high: {len(all_high_valid)}")
        if has_train:
            print(f"    -Training low: {len(all_low_train)}, high: {len(all_high_train)}")

    tensor_transform = transforms.ToTensor()
    flip_transform = None
    if flips:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])

    test_dataset = MyDataset_Crop(all_low_valid, all_high_valid, cropsize=None,
                                  tensor_transform=tensor_transform, test=True)

    train_loader = None
    if has_train:
        train_dataset = MyDataset_Crop(all_low_train, all_high_train, cropsize=cropsize,
                                       tensor_transform=tensor_transform, test=False,
                                       flips=flip_transform, crop_type=crop_type)

    if world_size > 1:
        samplers = []
        if has_train:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, shuffle=True, rank=rank)
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=False,
                                     num_workers=num_workers, pin_memory=True, drop_last=False, sampler=train_sampler)
            samplers.append(train_sampler)

        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, shuffle=True, rank=rank)
        samplers.append(test_sampler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    else:
        if has_train:
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_test, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=False)
        samplers = None

    return train_loader, test_loader, samplers
