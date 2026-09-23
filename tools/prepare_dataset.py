# Copyright (c) DADNet Authors. All rights reserved.
"""Prepare a defect-segmentation dataset for DADNet.

The DADNet dataset class (``SD900Dataset``) reads its image/annotation pairs
from a JSON-Lines split file -- one JSON object per line::

    {"image_path": "img_dir/train/foo.bmp", "seg_mask_path": "ann_dir/train/foo.png"}

This script builds those files.  Paths are written **relative to
``--data-root``** so the prepared dataset can be moved or copied to another
machine without editing anything.

Two input layouts are supported.

1. The dataset is already split into sub-directories::

       data/NEU-Seg/
       ├── img_dir/{train,val,test}/*.bmp
       └── ann_dir/{train,val,test}/*.png

   .. code-block:: shell

       python tools/prepare_dataset.py --data-root data/NEU-Seg \
           --splits train val test

2. The dataset is a flat directory of images and masks that still needs to be
   split.  Images and masks are paired by file stem and distributed according
   to ``--split-ratio``::

       data/NEU-Seg/
       ├── img_dir/*.bmp
       └── ann_dir/*.png

   .. code-block:: shell

       python tools/prepare_dataset.py --data-root data/NEU-Seg \
           --split-ratio 0.7 0.15 0.15 --seed 0

   The split is written to ``img_dir/{train,val,test}`` /
   ``ann_dir/{train,val,test}``; use ``--dry-run`` to preview the counts
   without touching the filesystem.
"""

import argparse
import os
import os.path as osp
import random
import shutil
import sys
from collections import defaultdict

# Suffixes to consider when scanning for images / annotations.
IMG_SUFFIXES = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
SEG_SUFFIXES = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')


def build_index(directory, suffixes):
    """Map ``stem.lower() -> filename`` for every file with a known suffix."""
    index = {}
    for name in os.listdir(directory):
        stem, ext = osp.splitext(name)
        if ext.lower() not in suffixes:
            continue
        index[stem.lower()] = name
    return index


def pair_by_stem(img_dir, ann_dir):
    """Pair images with annotations by file stem.

    Returns:
        list[tuple[str, str]]: ``(image_filename, ann_filename)`` pairs,
        sorted by image filename for reproducibility.
    """
    imgs = build_index(img_dir, IMG_SUFFIXES)
    anns = build_index(ann_dir, SEG_SUFFIXES)
    if not imgs:
        sys.exit(f'error: no images found in {img_dir}')
    if not anns:
        sys.exit(f'error: no annotations found in {ann_dir}')

    missing = sorted(set(imgs) - set(anns))
    if missing:
        preview = ', '.join(missing[:5])
        sys.exit(
            f'error: {len(missing)} image(s) in {img_dir} have no matching '
            f'annotation in {ann_dir} (matched by file stem). '
            f'Examples: {preview}')
    orphan = sorted(set(anns) - set(imgs))
    if orphan:
        print(f'warning: {len(orphan)} annotation(s) in {ann_dir} have no '
              f'matching image and will be ignored.')

    return sorted((imgs[s], anns[s]) for s in set(imgs) & set(anns))


def write_split_json(path, data_root, pairs, img_subdir, ann_subdir):
    """Write one JSON-Lines split file with paths relative to ``data_root``."""
    with open(path, 'w') as f:
        for img_name, ann_name in pairs:
            img_rel = osp.join(img_subdir, img_name)
            ann_rel = osp.join(ann_subdir, ann_name)
            f.write(
                '{"image_path": "%s", "seg_mask_path": "%s"}\n' %
                (img_rel, ann_rel))


def resolve(data_root, subdir):
    """Return ``data_root/subdir`` if it exists, else ``None``."""
    full = osp.join(data_root, subdir)
    return full if osp.isdir(full) else None


def split_existing(data_root, splits, dry_run):
    """Handle layout 1: sub-directories per split already exist."""
    for split in splits:
        img_dir = resolve(data_root, osp.join('img_dir', split))
        ann_dir = resolve(data_root, osp.join('ann_dir', split))
        if img_dir is None or ann_dir is None:
            sys.exit(f'error: expected both img_dir/{split} and ann_dir/{split} '
                     f'under {data_root}')
        pairs = pair_by_stem(img_dir, ann_dir)
        out = osp.join(data_root, f'{split}.json')
        print(f'{split:6s}: {len(pairs):6d} pairs -> {out}')
        if not dry_run:
            write_split_json(out, data_root, pairs,
                             osp.join('img_dir', split),
                             osp.join('ann_dir', split))


def split_flat(data_root, ratio, seed, dry_run, move):
    """Handle layout 2: one flat pool, split it into train/val/test."""
    if len(ratio) != 3 or any(r < 0 for r in ratio):
        sys.exit('error: --split-ratio needs three non-negative numbers, '
                 'e.g. --split-ratio 0.7 0.15 0.15')
    total_ratio = sum(ratio)
    if total_ratio <= 0:
        sys.exit('error: --split-ratio must sum to a positive number')

    flat_img = resolve(data_root, 'img_dir')
    flat_ann = resolve(data_root, 'ann_dir')
    if flat_img is None or flat_ann is None:
        sys.exit(f'error: expected img_dir/ and ann_dir/ directly under '
                 f'{data_root}')

    pairs = pair_by_stem(flat_img, flat_ann)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * ratio[0] / total_ratio))
    n_val = int(round(n * ratio[1] / total_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    chunks = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:],
    }

    for split, chunk in chunks.items():
        out = osp.join(data_root, f'{split}.json')
        print(f'{split:6s}: {len(chunk):6d} pairs -> {out}')
        if dry_run:
            continue
        write_split_json(out, data_root, chunk,
                         osp.join('img_dir', split),
                         osp.join('ann_dir', split))
        if move:
            for sub, names in (('img_dir', [p[0] for p in chunk]),
                               ('ann_dir', [p[1] for p in chunk])):
                dst = osp.join(data_root, sub, split)
                os.makedirs(dst, exist_ok=True)
                for name in names:
                    shutil.move(osp.join(data_root, sub, name),
                                osp.join(dst, name))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build train/val/test JSON-Lines split files for DADNet.')
    parser.add_argument('--data-root', required=True,
                        help='dataset root, e.g. data/NEU-Seg')
    parser.add_argument('--splits', nargs='+', default=None,
                        metavar='NAME',
                        help='existing split sub-directories, e.g. '
                             '--splits train val test')
    parser.add_argument('--split-ratio', nargs=3, type=float, default=None,
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='split a flat img_dir/ann_dir pool, e.g. '
                             '--split-ratio 0.7 0.15 0.15')
    parser.add_argument('--seed', type=int, default=0,
                        help='shuffle seed for --split-ratio (default: 0)')
    parser.add_argument('--move', action='store_true',
                        help='with --split-ratio, also move files into '
                             'img_dir/<split> and ann_dir/<split>')
    parser.add_argument('--dry-run', action='store_true',
                        help='only report the pair counts, write nothing')
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    if not osp.isdir(data_root):
        sys.exit(f'error: --data-root {data_root} is not a directory')

    if args.splits and args.split_ratio:
        sys.exit('error: --splits and --split-ratio are mutually exclusive')
    if args.splits:
        split_existing(data_root, args.splits, args.dry_run)
    elif args.split_ratio:
        if args.dry_run:
            args.move = False
        split_flat(data_root, args.split_ratio, args.seed, args.dry_run,
                   args.move)
    else:
        sys.exit('error: pass either --splits or --split-ratio')

    if args.dry_run:
        print('\n(dry run -- no files were written)')
    else:
        print('\ndone. Point data_root in your config at this directory.')


if __name__ == '__main__':
    main()
