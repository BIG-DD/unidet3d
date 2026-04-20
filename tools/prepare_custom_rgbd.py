"""
RGB-D + labelme polygon → UniDet3D 训练数据 + Open3D 可视化

深度增强策略（三级）:
  1. Depth Anything V2（DA2）神经网络估计稠密相对深度
     → 最小二乘标定到传感器 metric 尺度（scale + shift）
     → 与传感器深度加权融合（近处信任传感器，远处信任 DA2）
  2. 飞点清理：mask 腐蚀 → 统计离群点 → DBSCAN 取最大簇 → 百分位裁剪
  3. 类别先验尺寸兜底：框尺寸异常时用先验约束

用法:
    python tools/prepare_custom_rgbd.py \
        --data-dir /home/rossi/dataset/private/sampled_data \
        --out-dir  data/custom \
        --vis-dir  results/custom_vis
"""

import argparse, glob, json, os
import cv2
import mmengine
import numpy as np
import open3d as o3d
from skimage.segmentation import slic
from skimage.util import img_as_float

# ── 类别 ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['car', 'chair', 'person']
NAME2ID     = {n: i for i, n in enumerate(CLASS_NAMES)}
CLASS_COLORS_RGB = {
    'car':    np.array([255, 100,  50], dtype=np.float64) / 255.,
    'chair':  np.array([ 50, 210,  50], dtype=np.float64) / 255.,
    'person': np.array([ 80, 160, 255], dtype=np.float64) / 255.,
}

# [dx(lateral), dy(vertical-down), dz(depth)] 典型尺寸 (m)
CLASS_PRIORS = {
    'car':    np.array([1.8, 1.5, 4.2]),
    'chair':  np.array([0.6, 0.9, 0.6]),
    'person': np.array([0.5, 1.7, 0.4]),
}

DEPTH_MIN_MM   = 200
DEPTH_MAX_MM   = 8000
MAX_INST_DEPTH = 8.0
MIN_BOX_SIZE   = 0.10

EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
         (4,5),(4,6),(5,7),(6,7)]


# ─────────────────────────────────────────────────────────────────────────────
# Depth Anything V2 深度估计 + 传感器融合
# ─────────────────────────────────────────────────────────────────────────────

def load_depth_model(device='cuda:0'):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch
    print("加载 Depth Anything V2 Small ...")
    proc  = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()
    print("  DA2 加载完成")
    return proc, model


def fuse_depth(sensor_mm, rgb, proc, model, device='cuda:0'):
    """
    DA2 输出 affine-invariant 深度（可能是视差型，近处值大）。
    标定流程：
      1. 用传感器有效像素计算 DA2 与 sensor_m 的相关系数
      2. 若负相关 → DA2 是视差，先取倒数 (1/da2) 再做线性标定
      3. 若正相关 → DA2 是深度，直接线性标定
      4. 加权融合：近处用传感器，远处用 DA2
    """
    from PIL import Image
    import torch

    H, W = sensor_mm.shape

    # ── DA2 推理 ──────────────────────────────────────────────────────────────
    pil = Image.fromarray(rgb)
    inputs = proc(images=pil, return_tensors='pt').to(device)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth   # (1, H', W')
    pred_np = pred.squeeze().cpu().float().numpy()
    pred_np = cv2.resize(pred_np, (W, H), interpolation=cv2.INTER_LINEAR)

    sensor_m = sensor_mm / 1000.0
    valid    = (sensor_mm > DEPTH_MIN_MM) & (sensor_mm < DEPTH_MAX_MM)
    if valid.sum() < 200:
        return sensor_mm.copy()

    s_raw = pred_np[valid].astype(np.float64)
    d     = sensor_m[valid].astype(np.float64)

    # ── 判断 DA2 是视差还是深度 ───────────────────────────────────────────────
    corr = float(np.corrcoef(s_raw, d)[0, 1])
    if corr < 0:
        # 视差型：值越大 = 越近；取倒数转成"相对深度"
        s_calib = 1.0 / (s_raw + 1e-6)
        da2_for_calib = 1.0 / (pred_np.astype(np.float64) + 1e-6)
    else:
        s_calib = s_raw
        da2_for_calib = pred_np.astype(np.float64)

    # ── 最小二乘：sensor_m = a * s_calib + b ─────────────────────────────────
    A = np.stack([s_calib, np.ones_like(s_calib)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if a <= 0:
        a = float(np.median(d / (s_calib + 1e-8)))
        b = 0.0

    da2_metric_mm = (a * da2_for_calib + b) * 1000.0

    # ── 加权融合：近处信任传感器，远处信任 DA2 ───────────────────────────────
    sensor_z_m = np.where(valid, sensor_m, 0.0)
    w_sensor   = np.where(valid,
                          np.clip(1.0 - (sensor_z_m - 1.0) / 7.0, 0.1, 0.9),
                          0.0)
    fused = np.where(valid,
                     w_sensor * sensor_mm + (1 - w_sensor) * da2_metric_mm,
                     da2_metric_mm)
    fused = np.clip(fused, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# 几何工具
# ─────────────────────────────────────────────────────────────────────────────

def polygon_to_mask(points, H, W, erode_px=2):
    pts  = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    if erode_px > 0:
        area = int(mask.sum())
        px   = 0 if area < 200 else (max(1, erode_px // 2) if area < 800 else erode_px)
        if px > 0:
            k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px*2+1, px*2+1))
            mask = cv2.erode(mask, k, iterations=1)
    return mask.astype(bool)


def depth_to_xyz(depth_mm, fx, fy, cx, cy):
    H, W = depth_mm.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    z = depth_mm / 1000.0
    return np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], axis=-1)


def clean_and_bbox(pts3d, nb_neighbors=20, std_ratio=1.5):
    """统计离群点 → DBSCAN 最大簇 → 2/98 百分位 AABB。"""
    if len(pts3d) < nb_neighbors + 1:
        mn, mx = pts3d.min(0), pts3d.max(0)
        return pts3d, np.concatenate([(mn+mx)/2, np.maximum(mx-mn, 0.02)]).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd_s, _  = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    clean     = np.asarray(pcd_s.points) if len(pcd_s.points) >= 10 else pts3d

    if len(clean) >= 20:
        z_med  = float(np.median(clean[:, 2]))
        eps    = max(z_med * 0.025, 0.06)
        pcd2   = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(clean)
        labels = np.array(pcd2.cluster_dbscan(eps=eps, min_points=8,
                                               print_progress=False))
        if labels.max() >= 0:
            biggest = np.bincount(labels[labels >= 0]).argmax()
            cluster = clean[labels == biggest]
            if len(cluster) >= 10:
                clean = cluster

    mn = np.percentile(clean, 2,  axis=0)
    mx = np.percentile(clean, 98, axis=0)
    c  = (mn + mx) / 2
    s  = np.maximum(mx - mn, 0.02)
    return clean, np.concatenate([c, s]).astype(np.float32)


def prior_bbox(pts3d, poly_mask_2d, label, K):
    """
    深度覆盖不足时的兜底方案：
      - 中心位置由有效点中值确定
      - 框尺寸取 max(图像投影估算, 先验下限)
    """
    z   = float(np.median(pts3d[:, 2]))
    x   = float(np.median(pts3d[:, 0]))
    y   = float(np.median(pts3d[:, 1]))
    fx, fy = K['fx'], K['fy']

    rows, cols = np.where(poly_mask_2d)
    if len(cols) > 0:
        # 图像多边形范围投影到 3D（用有效点中值深度）
        proj_dx = (cols.max() - cols.min()) / fx * z
        proj_dy = (rows.max() - rows.min()) / fy * z
    else:
        proj_dx = proj_dy = 0.0

    prior = CLASS_PRIORS.get(label, np.array([1.0, 1.0, 1.0]))
    # 取图像投影和先验的较大值（防止截断），但不超过先验的 2 倍
    dx = float(np.clip(max(proj_dx, prior[0] * 0.5), prior[0]*0.4, prior[0]*2.0))
    dy = float(np.clip(max(proj_dy, prior[1] * 0.5), prior[1]*0.4, prior[1]*2.0))
    dz = float(prior[2])   # 深度方向无法从单帧投影估算，直接用先验

    return np.array([x, y, z, dx, dy, dz], dtype=np.float32)


def apply_prior_sanity(bbox, label):
    """用先验尺寸约束框大小（最小×0.3，最大×3），防止仍然异常。"""
    prior = CLASS_PRIORS.get(label, np.array([1.0, 1.0, 1.0]))
    for i, p in enumerate(prior):
        lo, hi = p * 0.25, p * 3.5
        bbox[3 + i] = float(np.clip(bbox[3 + i], lo, hi))
    return bbox


def superpoints_slic(rgb, valid_mask, n_segments=500):
    sp = slic(img_as_float(rgb), n_segments=n_segments,
              compactness=10, sigma=1, start_label=0)
    return sp[valid_mask].astype(np.int64)


def bbox_corners(bbox):
    bcx, bcy, bcz, bdx, bdy, bdz = bbox
    off = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                     [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=np.float32)
    return off * np.array([bdx/2, bdy/2, bdz/2]) + np.array([bcx, bcy, bcz])


def bbox_lineset(bbox, color):
    c  = bbox_corners(bbox)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(c),
        lines=o3d.utility.Vector2iVector(EDGES))
    ls.colors = o3d.utility.Vector3dVector([color] * len(EDGES))
    return ls


# ─────────────────────────────────────────────────────────────────────────────
# 单帧处理
# ─────────────────────────────────────────────────────────────────────────────

COVERAGE_THR = 0.25   # 有效深度覆盖率阈值：低于此值切换到先验方案

def process_frame(rgb_path, depth_path, ann_path, K,
                  scene_id, bins_dir, vis_dir,
                  erode_px=2, depth_proc=None):
    """
    depth_proc: (processor, model) 或 None（不使用 DA2）
    """
    fx, fy, cx_k, cy_k = K['fx'], K['fy'], K['cx'], K['cy']
    H, W = K['height'], K['width']

    rgb      = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth_mm = np.load(depth_path).astype(np.float32)
    with open(ann_path) as f:
        ann = json.load(f)

    # ── 深度增强 ──────────────────────────────────────────────────────────────
    if depth_proc is not None:
        proc, model = depth_proc
        depth_use = fuse_depth(depth_mm, rgb, proc, model)
        print("  [DA2] 深度融合完成，有效像素覆盖率："
              f"原始={((depth_mm>DEPTH_MIN_MM)&(depth_mm<DEPTH_MAX_MM)).mean()*100:.1f}%  "
              f"融合={((depth_use>DEPTH_MIN_MM)&(depth_use<DEPTH_MAX_MM)).mean()*100:.1f}%")
    else:
        depth_use = depth_mm

    valid   = (depth_use > DEPTH_MIN_MM) & (depth_use < DEPTH_MAX_MM)
    xyz_img = depth_to_xyz(depth_use, fx, fy, cx_k, cy_k)
    xyzrgb  = np.concatenate([xyz_img, rgb.astype(np.float32)], axis=-1)
    points  = xyzrgb[valid].astype(np.float32)
    sp_mask = superpoints_slic(rgb, valid)

    inst_img = np.full((H, W), -1, dtype=np.int64)
    sem_img  = np.full((H, W), -1, dtype=np.int64)
    instances_info = []
    inst_id  = 0

    for shape in ann['shapes']:
        label  = shape['label']
        cls_id = NAME2ID.get(label, -1)
        if cls_id < 0:
            continue

        poly_mask = polygon_to_mask(shape['points'], H, W, erode_px)
        fg        = poly_mask & valid
        n_pts     = int(fg.sum())
        n_poly    = int(poly_mask.sum())
        coverage  = n_pts / max(n_poly, 1)

        if n_pts < 10:
            print(f"    [{label}] 有效点 {n_pts} 太少，跳过")
            continue

        # 深度过滤
        pts3d  = xyz_img[fg]
        z_med  = float(np.median(pts3d[:, 2]))
        if z_med > MAX_INST_DEPTH:
            print(f"    [{label}] 中值深度 {z_med:.1f}m > {MAX_INST_DEPTH}m，跳过")
            continue

        # ── 选择 bbox 生成策略 ────────────────────────────────────────────────
        if coverage >= COVERAGE_THR:
            _, bbox = clean_and_bbox(pts3d)
            method = f"depth({coverage*100:.0f}%)"
        else:
            bbox   = prior_bbox(pts3d, poly_mask, label, K)
            method = f"prior({coverage*100:.0f}%)"

        # 先验尺寸约束兜底
        bbox = apply_prior_sanity(bbox, label)

        # 最小尺寸过滤
        if bbox[3:6].min() < MIN_BOX_SIZE:
            print(f"    [{label}] 框过小 {bbox[3]:.3f}x{bbox[4]:.3f}x{bbox[5]:.3f}m，跳过")
            continue

        inst_img[fg] = inst_id
        sem_img[fg]  = cls_id
        instances_info.append({
            'bbox_3d':       bbox[:6].tolist(),
            'bbox_label_3d': cls_id,
            'label':         label,
            'method':        method,
        })
        print(f"    [{label}]  z={z_med:.1f}m  {method}  "
              f"bbox=({bbox[0]:+.2f},{bbox[1]:+.2f},{bbox[2]:+.2f})  "
              f"size={bbox[3]:.2f}x{bbox[4]:.2f}x{bbox[5]:.2f}m")
        inst_id += 1

    inst_mask = inst_img[valid]
    sem_mask  = sem_img[valid]

    fname = f'{scene_id}.bin'
    points.tofile(os.path.join(bins_dir['points'],         fname))
    inst_mask.tofile(os.path.join(bins_dir['instance_mask'], fname))
    sem_mask.tofile(os.path.join(bins_dir['semantic_mask'],  fname))
    sp_mask.tofile(os.path.join(bins_dir['super_points'],    fname))

    if vis_dir:
        _vis_2d(rgb_path, ann['shapes'], xyz_img, valid, instances_info, K,
                os.path.join(vis_dir, f'{scene_id}_2d.jpg'), erode_px)
        # 深度对比图
        if depth_proc is not None:
            _vis_depth_compare(depth_mm, depth_use,
                               os.path.join(vis_dir, f'{scene_id}_depth.jpg'))
        _vis_3d_open3d(points, instances_info,
                       os.path.join(vis_dir, f'{scene_id}_3d.png'))

    return {
        'lidar_points':           {'num_pts_feats': 6, 'lidar_path': fname},
        'instances':              [{'bbox_3d': i['bbox_3d'],
                                    'bbox_label_3d': i['bbox_label_3d']}
                                   for i in instances_info],
        'pts_semantic_mask_path': fname,
        'pts_instance_mask_path': fname,
        'super_pts_path':         fname,
        'axis_align_matrix':      np.eye(4).tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def _project_corners(bbox, fx, fy, cx, cy):
    c = bbox_corners(bbox)
    z = np.clip(c[:, 2], 1e-3, None)
    u = (c[:, 0] / z * fx + cx).astype(int)
    v = (c[:, 1] / z * fy + cy).astype(int)
    return np.stack([u, v], axis=1)


def _vis_2d(rgb_path, shapes, xyz_img, valid, instances_info, K, out_path, erode_px):
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    H, W = K['height'], K['width']
    canvas  = cv2.imread(rgb_path)
    overlay = canvas.copy()
    for shape in shapes:
        c_bgr = (CLASS_COLORS_RGB.get(shape['label'], np.array([.8,.8,.8]))[::-1]*255).astype(int).tolist()
        pts   = np.array(shape['points'], dtype=np.int32).reshape(-1,1,2)
        cv2.fillPoly(overlay, [pts], c_bgr)
    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)

    for info in instances_info:
        color = CLASS_COLORS_RGB.get(info['label'], np.array([.8,.8,.8]))
        c_bgr = (color[::-1]*255).astype(int).tolist()
        try:
            corners = _project_corners(info['bbox_3d'], fx, fy, cx, cy)
        except Exception:
            continue
        for i, j in EDGES:
            p1 = tuple(np.clip(corners[i], -9999, 9999))
            p2 = tuple(np.clip(corners[j], -9999, 9999))
            cv2.line(canvas, p1, p2, c_bgr, 2, cv2.LINE_AA)
        b   = info['bbox_3d']
        u0  = int(np.clip(corners[:,0].min(), 0, W-1))
        v0  = int(np.clip(corners[:,1].min(), 10, H-1))
        txt = f"{info['label']} {b[3]:.2f}x{b[4]:.2f}x{b[5]:.2f}  [{info['method']}]"
        cv2.putText(canvas, txt, (u0, v0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, c_bgr, 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    print(f"  2D → {out_path}")


def _vis_depth_compare(sensor_mm, fused_mm, out_path):
    """左：传感器深度，右：融合深度，并排保存。"""
    def to_colormap(d_mm):
        d = np.clip(d_mm, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
        d = ((d - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
    left  = to_colormap(np.where(sensor_mm > 0, sensor_mm, 0))
    right = to_colormap(fused_mm)
    H, W  = left.shape[:2]
    cv2.putText(left,  'Sensor',  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(right, 'Fused(DA2+Sensor)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imwrite(out_path, np.hstack([left, right]))
    print(f"  depth cmp → {out_path}")


def _vis_3d_open3d(points, instances_info, out_path, render_w=1280, render_h=960):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.clip(points[:, 3:6]/255., 0, 1))
    pcd_vis = pcd.voxel_down_sample(0.02)

    render = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
    render.scene.set_background([0.10, 0.10, 0.10, 1.0])

    mat_p = o3d.visualization.rendering.MaterialRecord()
    mat_p.shader = "defaultUnlit"; mat_p.point_size = 2.0
    render.scene.add_geometry("pcd", pcd_vis, mat_p)

    mat_l = o3d.visualization.rendering.MaterialRecord()
    mat_l.shader = "unlitLine"; mat_l.line_width = 3.0
    for idx, info in enumerate(instances_info):
        color = CLASS_COLORS_RGB.get(info['label'], np.array([.8,.8,.8]))
        ls    = bbox_lineset(info['bbox_3d'], color)
        render.scene.add_geometry(f"b{idx}", ls, mat_l)

    # 相机位置
    p  = np.asarray(pcd_vis.points)
    xmin,ymin,zmin = p.min(0); xmax,ymax,zmax = p.max(0)
    ctr = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    xext,yext,zext = xmax-xmin, ymax-ymin, zmax-zmin
    pull = max(zext*0.8, 1.5)
    eye  = np.array([ctr[0]+xext*0.2, ctr[1]-yext*0.6, zmin-pull])
    render.scene.camera.look_at(ctr.tolist(), eye.tolist(), [0., -1., 0.])
    render.scene.camera.set_projection(
        55., render_w/render_h, 0.05, 200.,
        o3d.visualization.rendering.Camera.FovType.Vertical)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=min(min(xext,yext,zext)*0.3, 0.3), origin=ctr.tolist())
    mat_a = o3d.visualization.rendering.MaterialRecord()
    mat_a.shader = "defaultLit"
    render.scene.add_geometry("axes", axes, mat_a)

    o3d.io.write_image(out_path, render.render_to_image())
    print(f"  3D  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PKL
# ─────────────────────────────────────────────────────────────────────────────

def write_pkl(data_list, out_path):
    mmengine.dump({'metainfo': {'categories': NAME2ID,
                                'dataset': 'custom', 'info_version': '1.0'},
                   'data_list': data_list}, out_path, 'pkl')
    print(f"PKL → {out_path}  ({len(data_list)} scenes)")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',  default='/home/rossi/dataset/private/sampled_data')
    parser.add_argument('--out-dir',   default='data/custom')
    parser.add_argument('--vis-dir',   default='results/custom_vis')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--erode-px',  type=int,   default=2)
    parser.add_argument('--no-da2',    action='store_true',
                        help='跳过 Depth Anything V2，只用传感器深度')
    parser.add_argument('--device',    default='cuda:0')
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, 'camera_intrinsics.json')) as f:
        K = json.load(f)
    K['height'] = int(K.get('height', 480))
    K['width']  = int(K.get('width',  640))

    bins_dir = {}
    for sub in ['points','instance_mask','semantic_mask','super_points']:
        p = os.path.join(args.out_dir, sub)
        os.makedirs(p, exist_ok=True)
        bins_dir[sub] = p
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # 加载 DA2（共享，所有帧复用）
    depth_proc = None if args.no_da2 else load_depth_model(args.device)

    ann_files = sorted(glob.glob(os.path.join(args.data_dir, 'images', '*.json')))
    if not ann_files:
        raise FileNotFoundError(f"images/*.json not found in {args.data_dir}")

    all_infos = []
    for idx, ann_path in enumerate(ann_files):
        base_id    = os.path.splitext(os.path.basename(ann_path))[0].replace('rgb_', '')
        rgb_path   = os.path.join(args.data_dir, 'images', f'rgb_{base_id}.jpg')
        depth_path = os.path.join(args.data_dir, 'depth',  f'depth_{base_id}.npy')
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"[skip] 缺文件: {base_id}"); continue

        scene_id = f'scene_{idx:04d}'
        print(f"\n[{idx+1}/{len(ann_files)}] {scene_id}  {base_id}")
        info = process_frame(rgb_path, depth_path, ann_path, K,
                             scene_id, bins_dir, args.vis_dir,
                             erode_px=args.erode_px,
                             depth_proc=depth_proc)
        all_infos.append(info)

    n_val = max(1, int(len(all_infos) * args.val_ratio))
    write_pkl(all_infos[:-n_val],  os.path.join(args.out_dir, 'custom_infos_train.pkl'))
    write_pkl(all_infos[-n_val:],  os.path.join(args.out_dir, 'custom_infos_val.pkl'))
    print(f"\n完成！train={len(all_infos)-n_val}  val={n_val}")


if __name__ == '__main__':
    main()