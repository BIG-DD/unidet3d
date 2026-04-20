#!/usr/bin/env python
"""
Merge multiple custom RGB-D training data directories into one.

Usage:
    python tools/merge_custom_data.py \
        --src data/custom data/custom_batch2 data/custom_batch3 \
        --dst data/custom_merged \
        --val-ratio 0.15
"""

import argparse
import os
import shutil
import pickle
import numpy as np
from glob import glob


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def remap_scene_paths(data_list, old_data_root, new_data_root, subdirs):
    """Update absolute/relative paths inside each info dict."""
    updated = []
    for info in data_list:
        info = dict(info)
        # lidar_path stores relative path like 'points/scene_0000.bin'
        for key in ('lidar_path', 'pts_instance_mask_path',
                    'pts_semantic_mask_path', 'super_pts_path'):
            if key in info:
                fname = os.path.basename(info[key])
                subdir = os.path.dirname(info[key])
                info[key] = os.path.join(subdir, fname)
        updated.append(info)
    return updated


def merge(src_dirs, dst_dir, val_ratio):
    os.makedirs(dst_dir, exist_ok=True)
    subdirs = ['points', 'instance_mask', 'semantic_mask', 'super_points']
    for sd in subdirs:
        os.makedirs(os.path.join(dst_dir, sd), exist_ok=True)

    all_infos = []
    scene_counter = 0

    for src in src_dirs:
        # collect train + val infos
        for split in ('train', 'val'):
            pkl_path = os.path.join(src, f'custom_infos_{split}.pkl')
            if not os.path.exists(pkl_path):
                print(f"  [skip] {pkl_path} not found")
                continue
            pkg = load_pkl(pkl_path)
            metainfo = pkg.get('metainfo', {})
            classes = metainfo.get('classes', [])
            print(f"  Loading {pkl_path}: {len(pkg['data_list'])} scenes, classes={classes}")

            for info in pkg['data_list']:
                info = dict(info)
                old_name = os.path.basename(info['lidar_path'])
                new_name = f'scene_{scene_counter:04d}.bin'
                scene_counter += 1

                for sd in subdirs:
                    old_p = os.path.join(src, sd, old_name)
                    new_p = os.path.join(dst_dir, sd, new_name)
                    if os.path.exists(old_p):
                        shutil.copy2(old_p, new_p)
                    else:
                        print(f"  [warn] missing {old_p}")

                # update paths in info
                for key, sd in [('lidar_path', 'points'),
                                  ('pts_instance_mask_path', 'instance_mask'),
                                  ('pts_semantic_mask_path', 'semantic_mask'),
                                  ('super_pts_path', 'super_points')]:
                    if key in info:
                        info[key] = os.path.join(sd, new_name)

                all_infos.append(info)

    print(f"\nTotal scenes collected: {len(all_infos)}")

    # split train / val by val_ratio (last N scenes → val)
    np.random.seed(42)
    indices = np.random.permutation(len(all_infos))
    n_val = max(1, int(len(all_infos) * val_ratio))
    val_idx   = set(indices[-n_val:].tolist())
    train_idx = set(indices[:-n_val].tolist())

    train_list = [all_infos[i] for i in sorted(train_idx)]
    val_list   = [all_infos[i] for i in sorted(val_idx)]

    print(f"Train: {len(train_list)}  Val: {len(val_list)}")

    # read metainfo from first available pkl for classes
    meta = {}
    for src in src_dirs:
        for split in ('train', 'val'):
            p = os.path.join(src, f'custom_infos_{split}.pkl')
            if os.path.exists(p):
                meta = load_pkl(p).get('metainfo', {})
                break
        if meta:
            break

    save_pkl({'metainfo': meta, 'data_list': train_list},
             os.path.join(dst_dir, 'custom_infos_train.pkl'))
    save_pkl({'metainfo': meta, 'data_list': val_list},
             os.path.join(dst_dir, 'custom_infos_val.pkl'))

    print(f"\nSaved to {dst_dir}/")
    print(f"  custom_infos_train.pkl  ({len(train_list)} scenes)")
    print(f"  custom_infos_val.pkl    ({len(val_list)} scenes)")
    print(f"  classes: {meta.get('classes', 'unknown')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', nargs='+', required=True,
                   help='Source data directories (e.g. data/custom data/custom_batch2)')
    p.add_argument('--dst', required=True,
                   help='Destination merged directory (e.g. data/custom_merged)')
    p.add_argument('--val-ratio', type=float, default=0.15,
                   help='Fraction of scenes to put in val split (default: 0.15)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    merge(args.src, args.dst, args.val_ratio)