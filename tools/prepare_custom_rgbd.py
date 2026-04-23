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
        --rgb-dir    /path/to/rgb_images \
        --ann-dir    /path/to/labelme_jsons \
        --depth-dir  /path/to/depth_npys \
        --intrinsics /path/to/camera_intrinsics.json \
        --out-dir    data/custom \
        --vis-dir    results/custom_vis

文件匹配规则：以 --ann-dir 中每个 .json 的文件名主干（stem）为基准，
分别在 --rgb-dir 中搜索同名的 .jpg/.jpeg/.png，在 --depth-dir 中搜索同名的 .npy。
"""

import argparse, glob, json, os
import cv2
import mmengine
import numpy as np
import open3d as o3d
import yaml
from skimage.segmentation import slic
from skimage.util import img_as_float

_DEFAULT_LABEL_CONFIG = os.path.join(os.path.dirname(__file__), 'custom_label_config.yaml')

# 运行时由 load_label_config() 填充
CLASS_NAMES      = []   # ['car', 'chair', ...]，顺序即 label id
NAME2ID          = {}   # {'car': 0, ...}
LABEL_MAP        = {}   # {'汽车': 'car', ...}
CLASS_COLORS_RGB = {}   # {'car': np.array([r,g,b]) / 255.}
CLASS_PRIORS     = {}   # {'car': np.array([dx,dy,dz])}


def load_label_config(config_path: str):
    """从 YAML 配置文件加载类别信息，填充全局映射表。"""
    global CLASS_NAMES, NAME2ID, LABEL_MAP, CLASS_COLORS_RGB, CLASS_PRIORS
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cats = cfg['categories']
    CLASS_NAMES = [c['name_en'] for c in cats]
    NAME2ID     = {n: i for i, n in enumerate(CLASS_NAMES)}
    LABEL_MAP   = {c['name_zh']: c['name_en'] for c in cats}
    CLASS_COLORS_RGB = {
        c['name_en']: np.array(c['color_rgb'], dtype=np.float64) / 255.
        for c in cats
    }
    CLASS_PRIORS = {
        c['name_en']: np.array(c.get('size_prior_m', [1.0, 1.0, 1.0]))
        for c in cats
    }
    print(f"[config] 加载标签配置: {config_path}")
    print(f"  类别({len(CLASS_NAMES)}): {CLASS_NAMES}")

DEPTH_MIN_MM   = 200
DEPTH_MAX_MM   = 8000
MAX_INST_DEPTH = 6.0
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


def _ground_anchored_center(mn, mx):
    """
    相机坐标系：x=右, y=下, z=深。
    物体底面对应 y 最大处（贴地），框中心 y = y_bottom - dy/2。
    x / z 取区间中点（轴对齐）。
    """
    s = np.maximum(mx - mn, 0.02)
    c = np.array([(mn[0] + mx[0]) / 2,   # x: 水平中点
                  mx[1] - s[1] / 2,        # y: 底面锚定
                  (mn[2] + mx[2]) / 2],    # z: 深度中点
                 dtype=np.float64)
    return c, s


def clean_and_bbox(pts3d, nb_neighbors=20, std_ratio=1.5):
    """统计离群点 → DBSCAN 最大簇 → 2/98 百分位 AABB，框底面贴地。"""
    if len(pts3d) < nb_neighbors + 1:
        mn, mx = pts3d.min(0), pts3d.max(0)
        c, s   = _ground_anchored_center(mn, mx)
        return pts3d, np.concatenate([c, s]).astype(np.float32)

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
    c, s = _ground_anchored_center(mn, mx)
    return clean, np.concatenate([c, s]).astype(np.float32)


def prior_bbox(pts3d, poly_mask_2d, label, K):
    """
    深度覆盖不足时的兜底方案：
      - x/z 中心取有效点中值
      - y 中心：底面锚定到点云最低处（最大 y），再上移 dy/2
      - 框尺寸取 max(图像投影估算, 先验下限)
    """
    z      = float(np.median(pts3d[:, 2]))
    x      = float(np.median(pts3d[:, 0]))
    y_bot  = float(np.percentile(pts3d[:, 1], 98))  # 底面：y 最大（最低）处
    fx, fy = K['fx'], K['fy']

    rows, cols = np.where(poly_mask_2d)
    if len(cols) > 0:
        proj_dx = (cols.max() - cols.min()) / fx * z
        proj_dy = (rows.max() - rows.min()) / fy * z
    else:
        proj_dx = proj_dy = 0.0

    prior = CLASS_PRIORS.get(label, np.array([1.0, 1.0, 1.0]))
    dx = float(np.clip(max(proj_dx, prior[0] * 0.5), prior[0]*0.4, prior[0]*2.0))
    dy = float(np.clip(max(proj_dy, prior[1] * 0.5), prior[1]*0.4, prior[1]*2.0))
    dz = float(prior[2])
    y  = y_bot - dy / 2   # 框中心 y：底面锚定

    return np.array([x, y, z, dx, dy, dz], dtype=np.float32)


def apply_prior_sanity(bbox, label):
    """用先验尺寸约束框大小（最小×0.3，最大×3），防止仍然异常。"""
    prior = CLASS_PRIORS.get(label, np.array([1.0, 1.0, 1.0]))
    for i, p in enumerate(prior):
        lo, hi = p * 0.25, p * 3.5
        bbox[3 + i] = float(np.clip(bbox[3 + i], lo, hi))
    return bbox


def expand_bbox_by_polygon(bbox, poly_mask, fx, fy):
    """
    用 2D 多边形投影到 z_med 处的尺寸下限扩展 bbox。
    解决近处深度缺失时框只覆盖局部（如只有轮胎）的问题：
    即使 depth 点稀疏，框的 dx/dy 至少要覆盖整个多边形投影范围。
    底面 y 锚定不变，向上扩。
    """
    z_med = float(bbox[2])
    rows, cols = np.where(poly_mask)
    if len(cols) == 0:
        return bbox
    proj_dx = (int(cols.max()) - int(cols.min())) / fx * z_med
    proj_dy = (int(rows.max()) - int(rows.min())) / fy * z_med
    if proj_dx > bbox[3]:
        bbox[3] = float(proj_dx)
    if proj_dy > bbox[4]:
        y_bottom = bbox[1] + bbox[4] / 2   # 底面 y 不变
        bbox[4]  = float(proj_dy)
        bbox[1]  = y_bottom - bbox[4] / 2
    return bbox


def _box3d_iou(b1, b2):
    """轴对齐 3D IoU，b = [cx,cy,cz,dx,dy,dz]。"""
    inter = 1.0
    for i in range(3):
        lo = max(b1[i] - b1[i+3]/2, b2[i] - b2[i+3]/2)
        hi = min(b1[i] + b1[i+3]/2, b2[i] + b2[i+3]/2)
        if hi <= lo:
            return 0.0
        inter *= hi - lo
    v1 = b1[3] * b1[4] * b1[5]
    v2 = b2[3] * b2[4] * b2[5]
    return inter / (v1 + v2 - inter + 1e-9)


def nms_instances(instances_info, iou_thr=0.25):
    """
    同类别 3D 框 IoU NMS：重叠度超过阈值时保留体积更大的框。
    处理同一物体被多个 polygon 重复标注的情况。
    """
    n = len(instances_info)
    suppressed = [False] * n
    # 按体积降序，优先保留大框
    order = sorted(range(n),
                   key=lambda i: (instances_info[i]['bbox_3d'][3]
                                  * instances_info[i]['bbox_3d'][4]
                                  * instances_info[i]['bbox_3d'][5]),
                   reverse=True)
    for idx, i in enumerate(order):
        if suppressed[i]:
            continue
        bi = instances_info[i]['bbox_3d']
        for j in order[idx+1:]:
            if suppressed[j]:
                continue
            if instances_info[j]['label'] != instances_info[i]['label']:
                continue
            if _box3d_iou(bi, instances_info[j]['bbox_3d']) > iou_thr:
                suppressed[j] = True
    return [inst for i, inst in enumerate(instances_info) if not suppressed[i]]


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

def parse_annotation(ann_path):
    """
    解析新版标注 JSON（markData 格式），返回归一化后的 shapes 列表：
      [{'label': <英文类名>, 'points': [[x, y], ...]}, ...]
    仅保留 type=='polygon' 且标签在 LABEL_MAP 中的实例。
    """
    with open(ann_path) as f:
        raw = json.load(f)
    mark = raw['markData']
    shapes = []
    for annotation in mark.get('annotations', []):
        for item in annotation.get('data', []):
            if item.get('type') != 'polygon':
                continue
            en_label = LABEL_MAP.get(item.get('label', ''))
            if en_label is None:
                continue
            pts = [[p['x'], p['y']] for p in item.get('relativePos', [])]
            if len(pts) < 3:
                continue
            shapes.append({'label': en_label, 'points': pts})
    return shapes, int(mark.get('width', 640)), int(mark.get('height', 480))


def process_frame(rgb_path, depth_path, ann_path, K,
                  scene_id, bins_dir, vis_dir,
                  erode_px=2, depth_proc=None,
                  interactive=False, frame_info=''):
    """
    depth_proc: (processor, model) 或 None（不使用 DA2）
    """
    fx, fy, cx_k, cy_k = K['fx'], K['fy'], K['cx'], K['cy']
    H, W = K['height'], K['width']

    rgb      = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth_mm = np.load(depth_path).astype(np.float32)
    shapes, ann_W, ann_H = parse_annotation(ann_path)
    H, W = ann_H, ann_W   # 以标注文件中的图像尺寸为准

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

    for shape in shapes:
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

        # 2D 多边形投影扩框：确保框覆盖完整物体而非仅有深度的局部区域
        bbox = expand_bbox_by_polygon(bbox, poly_mask, fx, fy)

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

    # 去除同类重叠框（同一物体被多个 polygon 重复标注）
    before_nms = len(instances_info)
    instances_info = nms_instances(instances_info)
    if len(instances_info) < before_nms:
        print(f"  [NMS] {before_nms} → {len(instances_info)} 框")

    inst_mask = inst_img[valid]
    sem_mask  = sem_img[valid]

    # 语义着色：有标签的点用类别色，背景保持原始 RGB
    orig_colors = np.clip(points[:, 3:6] / 255., 0., 1.)
    seg_colors  = orig_colors.copy()
    for cls_id, name in enumerate(CLASS_NAMES):
        mask = sem_mask == cls_id
        if mask.any():
            seg_colors[mask] = CLASS_COLORS_RGB.get(name, np.array([.8, .8, .8]))

    fname = f'{scene_id}.bin'
    points.tofile(os.path.join(bins_dir['points'],         fname))
    inst_mask.tofile(os.path.join(bins_dir['instance_mask'], fname))
    sem_mask.tofile(os.path.join(bins_dir['semantic_mask'],  fname))
    sp_mask.tofile(os.path.join(bins_dir['super_points'],    fname))

    if vis_dir:
        _vis_2d(rgb_path, shapes, xyz_img, valid, instances_info, K,
                os.path.join(vis_dir, f'{scene_id}_2d.jpg'), erode_px)
        if depth_proc is not None:
            _vis_depth_compare(depth_mm, depth_use,
                               os.path.join(vis_dir, f'{scene_id}_depth.jpg'))
        _vis_3d_open3d(points, instances_info,
                       os.path.join(vis_dir, f'{scene_id}_3d.png'))

    if interactive:
        _vis_interactive_combined(rgb, depth_use, shapes,
                                  points, instances_info, seg_colors,
                                  scene_id, frame_info)

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


def _build_3d_geometries(points, instances_info, seg_colors=None):
    """构建点云 + 3D 框的 Open3D 几何体列表（离线/在线共用）。
    seg_colors: (N,3) float [0,1]，若提供则额外返回一个语义着色点云。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.clip(points[:, 3:6] / 255., 0, 1))
    pcd_vis = pcd.voxel_down_sample(0.02)

    p = np.asarray(pcd_vis.points)
    xmin, ymin, zmin = p.min(0)
    xmax, ymax, zmax = p.max(0)
    ctr   = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    xext, yext, zext = xmax-xmin, ymax-ymin, zmax-zmin

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=min(min(xext, yext, zext) * 0.3, 0.3), origin=ctr.tolist())

    pcd_seg_vis = None
    if seg_colors is not None:
        pcd_seg = o3d.geometry.PointCloud()
        pcd_seg.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd_seg.colors = o3d.utility.Vector3dVector(np.clip(seg_colors, 0., 1.))
        pcd_seg_vis = pcd_seg.voxel_down_sample(0.02)

    geoms = [pcd_vis, axes]
    for info in instances_info:
        color = CLASS_COLORS_RGB.get(info['label'], np.array([.8, .8, .8]))
        geoms.append(bbox_lineset(info['bbox_3d'], color))

    return geoms, ctr, (xext, yext, zext), zmin, pcd_seg_vis


def _vis_3d_open3d(points, instances_info, out_path, render_w=1280, render_h=960):
    geoms, ctr, (xext, yext, zext), zmin, _ = _build_3d_geometries(points, instances_info)
    pcd_vis = geoms[0]

    render = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
    render.scene.set_background([0.10, 0.10, 0.10, 1.0])

    mat_p = o3d.visualization.rendering.MaterialRecord()
    mat_p.shader = "defaultUnlit"; mat_p.point_size = 2.0
    render.scene.add_geometry("pcd", pcd_vis, mat_p)

    mat_l = o3d.visualization.rendering.MaterialRecord()
    mat_l.shader = "unlitLine"; mat_l.line_width = 3.0
    for idx, g in enumerate(geoms[2:]):
        render.scene.add_geometry(f"b{idx}", g, mat_l)

    mat_a = o3d.visualization.rendering.MaterialRecord()
    mat_a.shader = "defaultLit"
    render.scene.add_geometry("axes", geoms[1], mat_a)

    pull = max(zext * 0.8, 1.5)
    eye  = np.array([ctr[0] + xext*0.2, ctr[1] - yext*0.6, zmin - pull])
    render.scene.camera.look_at(ctr.tolist(), eye.tolist(), [0., -1., 0.])
    render.scene.camera.set_projection(
        55., render_w / render_h, 0.05, 200.,
        o3d.visualization.rendering.Camera.FovType.Vertical)

    o3d.io.write_image(out_path, render.render_to_image())
    print(f"  3D  → {out_path}")


def _build_2d_panel(rgb, depth_use, shapes, panel_h=540):
    """
    三格对比图（横向拼接，统一高度）：
      原始 RGB  |  深度伪彩  |  分割 Mask 叠加
    """
    def resize_h(img, h):
        s = h / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * s)), h))

    def add_title(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (220, 220, 220), 1, cv2.LINE_AA)
        return out

    # ── 原始 RGB ──────────────────────────────────────────────────────────────
    orig_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── 深度伪彩 ──────────────────────────────────────────────────────────────
    d = np.clip(depth_use, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    d = ((d - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)

    # ── 分割 Mask 叠加 ────────────────────────────────────────────────────────
    H, W = rgb.shape[:2]
    mask_bgr = orig_bgr.copy()
    overlay  = mask_bgr.copy()
    for shape in shapes:
        color = CLASS_COLORS_RGB.get(shape['label'], np.array([.8, .8, .8]))
        c_bgr = (color[::-1] * 255).astype(int).tolist()
        pts   = np.array(shape['points'], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], c_bgr)
        cv2.polylines(mask_bgr, [pts], True, c_bgr, 2, cv2.LINE_AA)
    mask_bgr = cv2.addWeighted(overlay, 0.45, mask_bgr, 0.55, 0)
    for shape in shapes:
        pts = np.array(shape['points'], dtype=np.float32)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        color  = CLASS_COLORS_RGB.get(shape['label'], np.array([.8, .8, .8]))
        c_bgr  = (color[::-1] * 255).astype(int).tolist()
        cv2.putText(mask_bgr, shape['label'], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(mask_bgr, shape['label'], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_bgr,    1, cv2.LINE_AA)

    panels = [
        add_title(resize_h(orig_bgr,   panel_h), 'Original'),
        add_title(resize_h(depth_bgr,  panel_h), 'Depth (fused)'),
        add_title(resize_h(mask_bgr,   panel_h), 'Seg Mask'),
    ]
    return np.hstack(panels)


def _vis_interactive_combined(rgb, depth_use, shapes,
                               points, instances_info, seg_colors,
                               scene_id, frame_info):
    """
    同时展示：
      • OpenCV 窗口 — 原始图 | 深度伪彩 | 分割 Mask（三格横排）
      • Open3D 窗口 — 点云 + 3D 检测框（可交互旋转/缩放）
    关闭 Open3D 窗口或按 Q / ESC 后继续下一帧。
    """
    # ── 2D 面板 ───────────────────────────────────────────────────────────────
    panel = _build_2d_panel(rgb, depth_use, shapes)
    win2d = f'2D  [{frame_info}]'
    cv2.namedWindow(win2d, cv2.WINDOW_NORMAL)
    cv2.imshow(win2d, panel)
    cv2.waitKey(1)

    # ── 3D 窗口 ───────────────────────────────────────────────────────────────
    geoms, ctr, (xext, yext, zext), zmin, pcd_seg_vis = \
        _build_3d_geometries(points, instances_info, seg_colors=seg_colors)

    labels_str = '  |  '.join(
        f"{i['label']} {i['bbox_3d'][3]:.2f}×{i['bbox_3d'][4]:.2f}×{i['bbox_3d'][5]:.2f}m"
        for i in instances_info) or '（无实例）'
    win3d = f'3D  [{frame_info}]  {labels_str}'

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=win3d, width=1280, height=960)
    for g in geoms:
        vis.add_geometry(g)

    # 语义着色点云：向右偏移一个场景宽度，与原始点云并排显示
    if pcd_seg_vis is not None:
        offset = np.array([xext * 1.15, 0., 0.])
        pts_off = np.asarray(pcd_seg_vis.points) + offset
        pcd_seg_shifted = o3d.geometry.PointCloud()
        pcd_seg_shifted.points = o3d.utility.Vector3dVector(pts_off)
        pcd_seg_shifted.colors = pcd_seg_vis.colors
        vis.add_geometry(pcd_seg_shifted)
        # 3D 框也偏移一份
        for info in instances_info:
            b = info['bbox_3d'][:]
            b[0] += float(offset[0])
            color = CLASS_COLORS_RGB.get(info['label'], np.array([.8, .8, .8]))
            vis.add_geometry(bbox_lineset(b, color))

    vc = vis.get_view_control()
    pull = max(zext * 0.8, 1.5)
    eye  = ctr + np.array([xext * 0.2, -yext * 0.6, -(zext * 0.5 + pull)])
    vc.set_lookat(ctr.tolist())
    vc.set_front((eye - ctr).tolist())
    vc.set_up([0., -1., 0.])
    vc.set_zoom(0.5)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.10])
    opt.point_size = 2.0

    print(f"  [交互] 左=原始RGB点云  右=语义分割着色点云  关闭或按 Q/ESC 继续下一帧 ...")
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    vis.destroy_window()
    cv2.destroyWindow(win2d)


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

def _build_stem_index(directory, extensions):
    """返回 {stem: abs_path}，取第一个匹配扩展名的文件。"""
    index = {}
    for ext in extensions:
        for p in glob.glob(os.path.join(directory, f'*{ext}')):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem not in index:
                index[stem] = p
    return index


def _resolve_dirs(args):
    """
    支持两种输入模式：
      1. --data-dir <root>  → 自动推导子目录 images/label/depth 及 camera_intrinsics.json
      2. 传统模式：--rgb-dir / --ann-dir / --depth-dir / --intrinsics 单独指定
    """
    if args.data_dir:
        root = args.data_dir
        rgb_dir    = os.path.join(root, 'images')
        ann_dir    = os.path.join(root, 'label')
        depth_dir  = os.path.join(root, 'depth')
        intrinsics = os.path.join(root, 'camera_intrinsics.json')
        for p, name in [(rgb_dir, 'images'), (ann_dir, 'label'), (depth_dir, 'depth')]:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"子目录不存在: {p}")
        if not os.path.isfile(intrinsics):
            raise FileNotFoundError(f"相机内参文件不存在: {intrinsics}")
    else:
        for name, val in [('--rgb-dir', args.rgb_dir),
                          ('--ann-dir', args.ann_dir),
                          ('--depth-dir', args.depth_dir),
                          ('--intrinsics', args.intrinsics)]:
            if val is None:
                raise ValueError(f"未指定 --data-dir 时必须提供 {name}")
        rgb_dir, ann_dir, depth_dir, intrinsics = (
            args.rgb_dir, args.ann_dir, args.depth_dir, args.intrinsics)
    return rgb_dir, ann_dir, depth_dir, intrinsics


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)

    # ── 简化模式 ──────────────────────────────────────────────────────────────
    parser.add_argument('--data-dir', default=None,
                        help='数据集根目录，内含 images/ label/ depth/ camera_intrinsics.json')

    # ── 传统模式（单独指定各路径） ────────────────────────────────────────────
    parser.add_argument('--rgb-dir',    default=None, help='RGB 图像目录（.jpg/.jpeg/.png）')
    parser.add_argument('--ann-dir',    default=None, help='标注 JSON 目录（.json）')
    parser.add_argument('--depth-dir',  default=None, help='深度图目录（.npy）')
    parser.add_argument('--intrinsics', default=None, help='相机内参 JSON 文件路径')

    # ── 公共选项 ──────────────────────────────────────────────────────────────
    parser.add_argument('--out-dir',    default='data/custom')
    parser.add_argument('--vis-dir',    default='results/custom_vis')
    parser.add_argument('--val-ratio',  type=float, default=0.2)
    parser.add_argument('--erode-px',   type=int,   default=2)
    parser.add_argument('--interactive', action='store_true',
                        help='每帧处理完后弹出 Open3D 交互窗口，关闭后继续下一帧')
    parser.add_argument('--no-da2',     action='store_true',
                        help='跳过 Depth Anything V2，只用传感器深度')
    parser.add_argument('--device',       default='cuda:0')
    parser.add_argument('--label-config', default=_DEFAULT_LABEL_CONFIG,
                        help='标签配置 YAML 文件路径')
    args = parser.parse_args()

    load_label_config(args.label_config)

    rgb_dir, ann_dir, depth_dir, intrinsics_path = _resolve_dirs(args)
    print(f"[数据目录] rgb={rgb_dir}  ann={ann_dir}  depth={depth_dir}")

    with open(intrinsics_path) as f:
        K = json.load(f)
    K['height'] = int(K.get('height', 480))
    K['width']  = int(K.get('width',  640))

    bins_dir = {}
    for sub in ['points', 'instance_mask', 'semantic_mask', 'super_points']:
        p = os.path.join(args.out_dir, sub)
        os.makedirs(p, exist_ok=True)
        bins_dir[sub] = p
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    depth_proc = None if args.no_da2 else load_depth_model(args.device)

    ann_files = sorted(glob.glob(os.path.join(ann_dir, '*.json')))
    if not ann_files:
        raise FileNotFoundError(f"*.json not found in {ann_dir}")

    rgb_index   = _build_stem_index(rgb_dir,   ['.jpg', '.jpeg', '.png'])
    depth_index = _build_stem_index(depth_dir, ['.npy'])
    # 去掉首个前缀词后按 suffix 索引，兼容 rgb_* / depth_* 前缀不同的文件名
    depth_suffix_index = {s.split('_', 1)[-1]: p for s, p in depth_index.items()}

    all_infos = []
    for idx, ann_path in enumerate(ann_files):
        stem = os.path.splitext(os.path.basename(ann_path))[0]

        rgb_path   = rgb_index.get(stem)
        depth_path = (depth_index.get(stem)
                      or depth_suffix_index.get(stem.split('_', 1)[-1]))
        if rgb_path is None or depth_path is None:
            missing = []
            if rgb_path   is None: missing.append('jpg')
            if depth_path is None: missing.append('npy')
            print(f"[skip] {stem}: 缺 {', '.join(missing)}")
            continue

        scene_id   = f'scene_{idx:04d}'
        frame_info = f'{idx+1}/{len(ann_files)}  {stem}'
        print(f"\n[{frame_info}] {scene_id}")
        info = process_frame(rgb_path, depth_path, ann_path, K,
                             scene_id, bins_dir, args.vis_dir,
                             erode_px=args.erode_px,
                             depth_proc=depth_proc,
                             interactive=args.interactive,
                             frame_info=frame_info)
        all_infos.append(info)

    n_val = max(1, int(len(all_infos) * args.val_ratio))
    write_pkl(all_infos[:-n_val], os.path.join(args.out_dir, 'custom_infos_train.pkl'))
    write_pkl(all_infos[-n_val:], os.path.join(args.out_dir, 'custom_infos_val.pkl'))
    print(f"\n完成！train={len(all_infos)-n_val}  val={n_val}")


if __name__ == '__main__':
    main()