from os import path as osp
from typing import Optional, Sequence
import numpy as np
import warnings

from mmdet3d.datasets.scannet_dataset import ScanNetSegDataset
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class ScanNetSegDataset_(ScanNetSegDataset):
    """We just add super_pts_path."""

    def get_scene_idxs(self, *args, **kwargs):
        """Compute scene_idxs for data sampling."""
        return np.arange(len(self)).astype(np.int32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_path'] = osp.join(
            self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

        info = super().parse_data_info(info)

        return info

@DATASETS.register_module()
class ScanNetDetDataset(ScanNetSegDataset_):
    """Dataset with loading gt_bboxes_3d, gt_labels_3d and
    axis-align matrix for evaluating SPFormer/OneFormer with
    IndoorMetric. We just copy some functions from Det3DDataset
    and comment some lines in them.
    """
    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `axis_align_matrix'.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        # info['super_pts_path'] = osp.join(
        #     self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

        info = super().parse_data_info(info)

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
    
    def _det3d_parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        In `Custom3DDataset`, we simply concatenate all the field
        in `instances` to `np.ndarray`, you can do the specific
        process in subclass. You have to convert `gt_bboxes_3d`
        to different coordinates according to the task.

        Args:
            info (dict): Info dict.

        Returns:
            dict or None: Processed `ann_info`.
        """
        # add s or gt prefix for most keys after concat
        # we only process 3d annotations here, the corresponding
        # 2d annotation process is in the `LoadAnnotations3D`
        # in `transforms`
        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }
        instances = info['instances']
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                # if 'label' in ann_name and ann_name != 'attr_label':
                #     temp_anns = [
                #         self.label_mapping[item] for item in temp_anns
                #     ]
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info['instances'] = info['instances']

            # for label in ann_info['gt_labels_3d']:
            #     if label != -1:
            #         cat_name = self.metainfo['classes'][label]
            #         self.num_ins_per_cat[cat_name] += 1

        return ann_info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """
        ann_info = self._det3d_parse_ann_info(info)
        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)
        # to target box structure

        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5))  # .convert_to(self.box_mode_3d)

        return ann_info


@DATASETS.register_module()
class CustomRGBDDetDataset(ScanNetDetDataset):
    """任意类别的 RGB-D 检测数据集，不受 ScanNet 词表限制。

    语义 mask 中类别 ID 应直接对应 metainfo['classes'] 的下标（0-based），
    背景点用 -1 标记。
    使用方法：在 config 中指定 metainfo=dict(classes=[...]) 即可。
    """

    # 提供一个空壳 METAINFO，避免继承 ScanNet 的类别限制
    # 真正的 classes/palette 在运行时由 metainfo=dict(classes=...) 注入
    METAINFO: dict = dict(
        classes=(),
        palette=(),
        seg_valid_class_ids=(),
        seg_all_class_ids=())

    def get_label_mapping(self,
                          new_classes: Optional[Sequence] = None) -> tuple:
        """对自定义类别返回恒等映射，不做词表校验。"""
        if new_classes is None:
            new_classes = self.metainfo.get('classes', [])
        n = len(new_classes)
        label_mapping  = {i: i for i in range(n)}
        label2cat      = {i: name for i, name in enumerate(new_classes)}
        valid_class_ids = tuple(range(n))
        return label_mapping, label2cat, valid_class_ids

    def _update_palette(self, new_classes, palette):
        """为自定义类别自动生成或透传调色板，不查 ScanNet 词表。"""
        # 如果外部已提供足够颜色就直接用
        if palette is not None and len(palette) >= len(new_classes):
            return list(palette[:len(new_classes)])
        # 自动生成循环颜色
        default_colors = [
            (106, 90,  205), (255, 165,  0),  (50,  205, 50),
            (30, 144,  255), (220,  20,  60), (255, 215,  0),
            (0,  128, 128),  (148,   0, 211), (255, 127,  80),
            (135, 206, 235),
        ]
        return [default_colors[i % len(default_colors)]
                for i in range(len(new_classes))]