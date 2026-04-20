#!/usr/bin/env python
"""
UniDet3D inference on custom RGB-D data.

Usage:
    python tools/infer_rgbd.py \
        --data-dir /home/rossi/dataset/private/1 \
        --checkpoint work_dirs/tmp/unidet3d.pth \
        --config configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py \
        --out-dir results/rgbd \
        --score-thr 0.3 \
        --frame-idx 0
"""

import sys
import os
import argparse
import json
import glob
import re

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── colour palette for 18 ScanNet classes ─────────────────────────────────────
SCANNET_CLASSES = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
    'otherfurniture',
]

CLASS_COLORS_BGR = np.array([
    [42,  75, 215],   # cabinet  – blue
    [80, 175,  76],   # bed      – green
    [ 0,  60, 255],   # chair    – red-orange
    [220, 140,  30],  # sofa     – steel blue
    [ 35, 200, 250],  # table    – yellow
    [220,  30, 220],  # door     – magenta
    [ 30, 220, 100],  # window   – lime
    [100,  30, 220],  # bookshelf
    [220, 220,  30],  # picture
    [ 60, 180, 180],  # counter
    [ 30,  30, 200],  # desk
    [ 90, 200,  60],  # curtain
    [200,  90,  60],  # refrigerator
    [200,  60, 160],  # showercurtrain
    [ 70, 200, 200],  # toilet
    [  0, 130, 255],  # sink
    [180,  80, 180],  # bathtub
    [150, 150, 150],  # otherfurniture
], dtype=np.uint8)


# ── preprocessing ──────────────────────────────────────────────────────────────

def load_frame_pair(data_dir, frame_idx=0):
    """Load matched rgb/depth pair by sorted index.

    Searches for RGB in data_dir and data_dir/images/; depth in data_dir and
    data_dir/depth/. Accepts both rgb_*.jpg and rgb_d2c_*.jpg patterns.
    """
    rgb_patterns = [
        os.path.join(data_dir, 'rgb_*.jpg'),
        os.path.join(data_dir, 'images', 'rgb_*.jpg'),
        os.path.join(data_dir, 'images', '*.jpg'),
    ]
    depth_patterns = [
        os.path.join(data_dir, 'depth_*.npy'),
        os.path.join(data_dir, 'depth', 'depth_*.npy'),
        os.path.join(data_dir, 'depth', '*.npy'),
    ]
    rgb_files = []
    for p in rgb_patterns:
        rgb_files = sorted(glob.glob(p))
        if rgb_files:
            break
    depth_files = []
    for p in depth_patterns:
        depth_files = sorted(glob.glob(p))
        if depth_files:
            break
    if not rgb_files:
        raise FileNotFoundError(f"No RGB jpg files found under {data_dir}")
    idx = min(frame_idx, len(rgb_files) - 1)
    rgb   = cv2.cvtColor(cv2.imread(rgb_files[idx]), cv2.COLOR_BGR2RGB)
    depth = np.load(depth_files[idx]).astype(np.float32)  # mm
    return rgb, depth, rgb_files[idx]


def depth_to_pointcloud(depth_mm, rgb, fx, fy, cx, cy,
                         depth_min_mm=100, depth_max_mm=5000):
    """Back-project depth image to (N,6) point cloud [x,y,z,R,G,B]."""
    H, W = depth_mm.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth_mm.copy()
    valid = (z > depth_min_mm) & (z < depth_max_mm)

    z_m  = z  / 1000.0
    x_m  = (uu - cx) * z_m / fx
    y_m  = (vv - cy) * z_m / fy

    xyz  = np.stack([x_m, y_m, z_m], axis=-1)          # (H,W,3)
    rgb_ = rgb.astype(np.float32)                         # (H,W,3)
    xyzrgb = np.concatenate([xyz, rgb_], axis=-1)        # (H,W,6)

    return xyzrgb[valid].astype(np.float32), valid        # (N,6), (H,W)


def generate_superpoints_slic(rgb, valid_mask, n_segments=500, compactness=10):
    """Superpixels via SLIC → superpoint IDs for valid depth pixels."""
    from skimage.segmentation import slic
    from skimage.util import img_as_float
    sp = slic(img_as_float(rgb), n_segments=n_segments,
              compactness=compactness, sigma=1, start_label=0,
              channel_axis=-1)
    return sp[valid_mask].astype(np.int64)


# ── model loading ──────────────────────────────────────────────────────────────

def load_model(config_path, checkpoint_path, device='cuda:0'):
    from mmengine.config import Config
    from mmdet3d.utils import register_all_modules
    from mmdet3d.registry import MODELS
    register_all_modules()

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)

    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)} (ignored)")

    model = model.to(device).eval()
    return model


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(model, points_raw, sp_mask, dataset_tag='scannet',
                  device='cuda:0'):
    """
    Run UniDet3D on a single frame.

    points_raw : (N,6) float32 – already colour-normalised
    sp_mask    : (N,)  int64
    """
    from mmdet3d.structures import Det3DDataSample, PointData

    pts_t = torch.from_numpy(points_raw).float().to(device)
    sp_t  = torch.from_numpy(sp_mask).long().to(device)

    data_sample = Det3DDataSample()
    data_sample.set_metainfo({'lidar_path': f'/data/{dataset_tag}/scene'})
    gt_pts_seg = PointData()
    gt_pts_seg.sp_pts_mask = sp_t
    data_sample.gt_pts_seg = gt_pts_seg

    batch_inputs = {'points': [pts_t]}
    with torch.no_grad():
        results = model.predict(batch_inputs, [data_sample])
    return results[0]


# ── visualisation ──────────────────────────────────────────────────────────────

def project_box_to_image(center, size, intrinsics, R_cam=None):
    """
    Project an AABB [cx,cy,cz, dx,dy,dz] to 8 image points.

    Camera frame: x-right, y-down, z-forward (standard depth camera).
    Returns (8,2) pixel coords or None if box is behind camera.
    """
    cx, cy, cz = center
    dx, dy, dz = size

    # 8 corners in camera space
    corners = np.array([
        [cx + sx * dx/2, cy + sy * dy/2, cz + sz * dz/2]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])  # (8,3)

    if R_cam is not None:
        corners = corners @ R_cam.T

    valid = corners[:, 2] > 0.05
    if not valid.all():
        return None

    fx = intrinsics['fx']; fy = intrinsics['fy']
    px = intrinsics['cx']; py = intrinsics['cy']

    u = corners[:, 0] / corners[:, 2] * fx + px
    v = corners[:, 1] / corners[:, 2] * fy + py
    return np.stack([u, v], axis=1).astype(int)   # (8,2)


BOX_EDGES = [
    (0,1),(1,3),(3,2),(2,0),   # front face
    (4,5),(5,7),(7,6),(6,4),   # back face
    (0,4),(1,5),(2,6),(3,7),   # connecting edges
]


def draw_box_on_image(img, corners_2d, color_bgr, label, score):
    """Draw 3-D bounding-box wireframe and label onto img (in-place)."""
    for a, b in BOX_EDGES:
        pa = tuple(np.clip(corners_2d[a], -1, max(img.shape[:2])*2))
        pb = tuple(np.clip(corners_2d[b], -1, max(img.shape[:2])*2))
        cv2.line(img, pa, pb, color_bgr.tolist(), 2, cv2.LINE_AA)

    # label near the top corner
    top = corners_2d[np.argmin(corners_2d[:, 1])]
    text = f'{label} {score:.2f}'
    cv2.putText(img, text,
                (int(top[0]), max(int(top[1]) - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color_bgr.tolist(), 2, cv2.LINE_AA)


def visualize_2d(rgb, result, intrinsics, score_thr, out_path, classes):
    """Project 3-D boxes onto the RGB image and save."""
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pred = result.pred_instances_3d
    bboxes = pred.bboxes_3d.tensor.cpu().numpy()   # (M,6)
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()

    drawn = 0
    for bbox, score, label in zip(bboxes, scores, labels):
        if score < score_thr:
            continue
        center = bbox[:3].tolist()
        size   = bbox[3:6].tolist()
        c2d = project_box_to_image(center, size, intrinsics)
        if c2d is None:
            continue
        # clip to image bounds check
        H, W = rgb.shape[:2]
        if np.all((c2d[:, 0] < 0) | (c2d[:, 0] >= W)) or \
           np.all((c2d[:, 1] < 0) | (c2d[:, 1] >= H)):
            continue
        color = CLASS_COLORS_BGR[label % len(CLASS_COLORS_BGR)]
        draw_box_on_image(vis, c2d, color,
                          classes[label] if label < len(classes) else str(label),
                          score)
        drawn += 1

    print(f"  → {drawn} boxes drawn on 2-D image (thr={score_thr})")
    cv2.imwrite(out_path, vis)
    return vis


def visualize_3d(points_raw, result, out_path, classes, score_thr,
                 color_mean=127.5):
    """Save matplotlib 3-D scatter + bounding-box wireframes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patches as mpatches

    pts  = points_raw.copy()
    xyz  = pts[:, :3]
    rgb_norm = np.clip((pts[:, 3:] + color_mean) / 255.0, 0, 1)

    # Downsample for plotting
    stride = max(1, len(xyz) // 20000)
    xyz_d  = xyz[::stride]
    rgb_d  = rgb_norm[::stride]

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(xyz_d[:, 0], xyz_d[:, 2], -xyz_d[:, 1],   # x, z, -y → x,depth,up
               c=rgb_d, s=0.4, depthshade=True)

    pred   = result.pred_instances_3d
    bboxes = pred.bboxes_3d.tensor.cpu().numpy()
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()

    legend_handles = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if score < score_thr:
            continue
        cx, cy, cz, dx, dy, dz = bbox[:6]
        # corners  x,y,z in camera frame; plot as x,z,-y
        x0, x1 = cx-dx/2, cx+dx/2
        y0, y1 = cy-dy/2, cy+dy/2
        z0, z1 = cz-dz/2, cz+dz/2
        # edges: 12 lines
        edges = [
            ([x0,x1],[z0,z0],[-y0,-y0]), ([x0,x1],[z1,z1],[-y0,-y0]),
            ([x0,x1],[z0,z0],[-y1,-y1]), ([x0,x1],[z1,z1],[-y1,-y1]),
            ([x0,x0],[z0,z1],[-y0,-y0]), ([x1,x1],[z0,z1],[-y0,-y0]),
            ([x0,x0],[z0,z1],[-y1,-y1]), ([x1,x1],[z0,z1],[-y1,-y1]),
            ([x0,x0],[z0,z0],[-y0,-y1]), ([x1,x1],[z0,z0],[-y0,-y1]),
            ([x0,x0],[z1,z1],[-y0,-y1]), ([x1,x1],[z1,z1],[-y0,-y1]),
        ]
        color_rgb = CLASS_COLORS_BGR[label % len(CLASS_COLORS_BGR)][[2,1,0]] / 255.0
        for ex, ey, ez in edges:
            ax.plot(ex, ey, ez, color=color_rgb, linewidth=1.5)
        name = classes[label] if label < len(classes) else str(label)
        patch = mpatches.Patch(color=color_rgb, label=f'{name} {score:.2f}')
        if not any(h.get_label() == patch.get_label() for h in legend_handles):
            legend_handles.append(patch)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Depth Z (m)'); ax.set_zlabel('Up -Y (m)')
    ax.set_title('UniDet3D – 3-D Point Cloud with Detections')
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → 3-D view saved: {out_path}")


def save_detections_txt(result, classes, score_thr, out_path):
    """Save detection results as human-readable text."""
    pred = result.pred_instances_3d
    bboxes = pred.bboxes_3d.tensor.cpu().numpy()
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()

    lines = ['class, score, cx, cy, cz, dx, dy, dz']
    for bbox, score, label in sorted(
            zip(bboxes, scores, labels), key=lambda x: -x[1]):
        if score < score_thr:
            continue
        name = classes[label] if label < len(classes) else str(label)
        cx, cy, cz, dx, dy, dz = bbox[:6]
        lines.append(
            f'{name}, {score:.3f}, '
            f'{cx:.3f}, {cy:.3f}, {cz:.3f}, '
            f'{dx:.3f}, {dy:.3f}, {dz:.3f}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  → detections saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',   default='/home/rossi/dataset/private/sampled_data')
    p.add_argument('--checkpoint', default='work_dirs/unidet3d_custom/epoch_64.pth')
    p.add_argument('--config',
        default='configs/unidet3d_custom.py')
    p.add_argument('--out-dir',    default='results/rgbd')
    p.add_argument('--score-thr',  type=float, default=0.3)
    p.add_argument('--frame-idx',  type=int,   default=0,
                   help='Index of frame to use (0=first)')
    p.add_argument('--n-segments', type=int,   default=500,
                   help='SLIC superpixel count for superpoints')
    p.add_argument('--device',     default='cuda:0')
    p.add_argument('--dataset-tag', default='scannet',
                   help='Dataset name tag (determines class vocabulary)')
    p.add_argument('--classes', nargs='+', default=None,
                   help='Custom class names (overrides built-in vocab). '
                        'E.g. --classes car chair person')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load camera intrinsics ────────────────────────────────────────────────
    intr_path = os.path.join(args.data_dir, 'camera_intrinsics.json')
    with open(intr_path) as f:
        intrinsics = json.load(f)
    print(f"Camera: fx={intrinsics['fx']:.1f}  fy={intrinsics['fy']:.1f}  "
          f"cx={intrinsics['cx']:.1f}  cy={intrinsics['cy']:.1f}")

    # ── select class vocab based on dataset tag ───────────────────────────────
    if args.classes is not None:
        classes = args.classes
    elif args.dataset_tag == 'scannet':
        classes = SCANNET_CLASSES
    else:
        classes = SCANNET_CLASSES   # fallback

    # ── load frame ────────────────────────────────────────────────────────────
    print(f"\nLoading frame #{args.frame_idx} …")
    rgb, depth, rgb_path = load_frame_pair(args.data_dir, args.frame_idx)
    print(f"  RGB  {rgb.shape}  |  depth {depth.shape}  "
          f"range [{depth.min():.0f}, {depth.max():.0f}] mm")

    # ── generate point cloud ──────────────────────────────────────────────────
    print("Generating point cloud …")
    points, valid_mask = depth_to_pointcloud(
        depth, rgb,
        intrinsics['fx'], intrinsics['fy'],
        intrinsics['cx'], intrinsics['cy'],
        depth_min_mm=200, depth_max_mm=4000)
    print(f"  {len(points):,} valid points")

    # ── generate superpoints ──────────────────────────────────────────────────
    print(f"Generating superpoints (SLIC n={args.n_segments}) …")
    sp_mask = generate_superpoints_slic(rgb, valid_mask, args.n_segments)
    print(f"  {sp_mask.max()+1} superpoints")

    # ── colour normalisation (ScanNet convention) ─────────────────────────────
    points_norm = points.copy()
    points_norm[:, 3:] -= 127.5

    # ── load model ────────────────────────────────────────────────────────────
    print("\nLoading model …")
    model = load_model(args.config, args.checkpoint, args.device)
    print("  Model loaded.")

    # ── inference ─────────────────────────────────────────────────────────────
    print("Running inference …")
    result = run_inference(model, points_norm, sp_mask,
                           args.dataset_tag, args.device)

    pred = result.pred_instances_3d
    n_all  = len(pred.scores_3d)
    n_kept = (pred.scores_3d.cpu().numpy() >= args.score_thr).sum()
    print(f"  {n_all} raw detections → {n_kept} above thr={args.score_thr}")

    # print top detections (always show top-15 regardless of threshold)
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()
    bboxes = pred.bboxes_3d.tensor.cpu().numpy()
    order  = np.argsort(-scores)
    print("\n  Top detections (top-15 regardless of threshold):")
    print(f"  {'class':<20} {'score':>6}  {'center (m)':>25}  {'size (m)':>20}")
    for i in order[:15]:
        name = classes[labels[i]] if labels[i] < len(classes) else str(labels[i])
        cx,cy,cz,dx,dy,dz = bboxes[i][:6]
        print(f"  {name:<20} {scores[i]:>6.3f}  "
              f"({cx:6.2f},{cy:6.2f},{cz:6.2f})  "
              f"({dx:.2f}x{dy:.2f}x{dz:.2f})")

    # ── save results ──────────────────────────────────────────────────────────
    frame_name = os.path.splitext(os.path.basename(rgb_path))[0]

    # 2-D projection on RGB
    out_2d = os.path.join(args.out_dir, f'{frame_name}_2d.jpg')
    print(f"\nRendering 2-D visualisation → {out_2d}")
    visualize_2d(rgb, result, intrinsics, args.score_thr, out_2d, classes)

    # 3-D Open3D view
    out_3d = os.path.join(args.out_dir, f'{frame_name}_3d.png')
    print(f"Rendering 3-D visualisation → {out_3d}")
    visualize_3d(points_norm, result, out_3d, classes, args.score_thr)

    # Text detections
    out_txt = os.path.join(args.out_dir, f'{frame_name}_detections.txt')
    save_detections_txt(result, classes, args.score_thr, out_txt)

    # Also save original RGB for reference
    cv2.imwrite(os.path.join(args.out_dir, f'{frame_name}_rgb.jpg'),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    print(f"\nDone. Results in: {args.out_dir}/")


if __name__ == '__main__':
    main()
