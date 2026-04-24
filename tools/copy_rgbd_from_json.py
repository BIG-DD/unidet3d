import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Union
import json

def move_all_npy_to_dir(
    src_root: Union[str, Path],
    dst_dir: Union[str, Path],
    *,
    preserve_structure: bool = False,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    在 ``src_root`` 下递归查找所有 ``.npy`` 文件，并**移动**到 ``dst_dir``。

    - ``preserve_structure=False``（默认）：全部平铺在 ``dst_dir`` 下；若不同子目录
      存在同名 ``.npy``，自动加 ``_1``、``_2`` 等后缀避免覆盖。
    - ``preserve_structure=True``：保留相对 ``src_root`` 的子目录结构。

    Args:
        src_root: 可含多层子目录的根路径。
        dst_dir: 目标文件夹（不存在则创建）。
        preserve_structure: 是否保留相对路径结构。
        dry_run: 为 True 时只打印将要执行的操作，不真正移动。

    Returns:
        ``(成功移动数, 失败数)``
    """
    src_root = Path(src_root).expanduser().resolve()
    dst_dir = Path(dst_dir).expanduser().resolve()

    if not src_root.is_dir():
        raise NotADirectoryError(f"源路径不是目录: {src_root}")

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(src_root.rglob("*.jpg"))
    flat_name_count: defaultdict[str, int] = defaultdict(int)

    def unique_flat_name(basename: str) -> str:
        flat_name_count[basename] += 1
        n = flat_name_count[basename]
        if n == 1:
            return basename
        stem, suf = os.path.splitext(basename)
        return f"{stem}_{n - 1}{suf}"

    moved, failed = 0, 0
    for src_file in npy_files:
        try:
            if preserve_structure:
                rel = src_file.relative_to(src_root)
                dest = dst_dir / rel
                if not dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                dest = dst_dir / unique_flat_name(src_file.name)

            if dest.resolve() == src_file.resolve():
                continue

            if dry_run:
                print(f"[dry-run] move {src_file} -> {dest}")
            else:
                shutil.move(os.fspath(src_file), os.fspath(dest))
            moved += 1
        except OSError as e:
            print(f"[错误] 无法移动 {src_file}: {e}")
            failed += 1

    return moved, failed

def copy_rgbd_from_json():
    json_dir = r"E:\work\code\python\unidet3d\jpg_list.json"
    # annotations_dir = r"D:\security\version-4\annotation"
    # annotations_list = os.listdir(annotations_dir)
    # jpg_dir = r"D:\security\version-4\images"
    npy_dir = r"E:\work\data\changan_dataset\depth"
    output_dir = r"E:\work\data\changan_dataset\out_depth"
    with open(json_dir, "r") as f:
        json_data = json.load(f)
        for annotation in json_data:
            annotation_split_name = annotation.split("d2c_")[-1].split(".")[0]
            # jpg_path = os.path.join(jpg_dir, f"rgb_{annotation_split_name}.jpg")
            npy_path = os.path.join(npy_dir, f"depth_d2c_{annotation_split_name}.npy")

            shutil.copy(npy_path, os.path.join(output_dir, f"depth_d2c_{annotation_split_name}.npy"))

def copy_rgbd_from_dir():
    annotations_dir = r"D:\security\annotation\images"
    annotations_list = os.listdir(annotations_dir)
    jpg_dir = r"E:\work\data\changan_dataset\jpg"
    npy_dir = r"E:\work\data\changan_dataset\depth"
    output_dir = r"D:\security\new_data\images"
    for annotation in annotations_list:
        annotation_split_name = annotation.split("d2c_")[-1].split(".")[0]
        jpg_path = os.path.join(jpg_dir, f"rgb_d2c_{annotation_split_name}.jpg")
        npy_path = os.path.join(npy_dir, f"depth_d2c_{annotation_split_name}.npy")
        if os.path.exists(jpg_path) and os.path.exists(npy_path):
            shutil.copy(jpg_path, os.path.join(output_dir, f"rgb_d2c_{annotation_split_name}.jpg"))
            shutil.copy(npy_path, os.path.join(output_dir, f"depth_d2c_{annotation_split_name}.npy"))

if __name__ == "__main__":
    copy_rgbd_from_dir()