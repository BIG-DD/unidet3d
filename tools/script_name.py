import os
import random
import shutil
from pathlib import Path

def sample_dataset(src_dir, output_root, num_samples=10):
    # 1. 初始化路径
    src_path = Path(src_dir)
    img_out = Path(output_root) / "images"
    depth_out = Path(output_root) / "depth"

    # 创建输出目录
    img_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    # 2. 搜索所有的 .jpg 文件并解析出对应的 ID
    # 匹配规则: rgb_***.jpg
    all_jpgs = list(src_path.glob("rgb_*.jpg"))
    
    pairs = []
    for jpg_file in all_jpgs:
        # 提取文件名中间的 *** 部分
        # 例如: rgb_car_001.jpg -> car_001
        file_id = jpg_file.stem.replace("rgb_", "")
        
        # 构造对应的 npy 文件名: depth_***.npy
        npy_file = src_path / f"depth_{file_id}.npy"
        
        # 只有当 npy 文件确实存在时，才加入候选列表
        if npy_file.exists():
            pairs.append((jpg_file, npy_file))

    # 3. 检查是否有足够的样本
    if not pairs:
        print("错误：没有找到匹配的 rgb_*.jpg 和 depth_*.npy 文件对。")
        return

    actual_samples = min(len(pairs), num_samples)
    if len(pairs) < num_samples:
        print(f"警告：仅找到 {len(pairs)} 对匹配文件，将全部移动。")

    # 4. 随机抽取
    selected_pairs = random.sample(pairs, actual_samples)

    # 5. 执行复制操作
    print(f"正在抽取 {actual_samples} 对文件到目标目录...")
    for img_src, npy_src in selected_pairs:
        shutil.copy2(img_src, img_out / img_src.name)
        shutil.copy2(npy_src, depth_out / npy_src.name)
        print(f"已复制: ID 为 [{img_src.stem.replace('rgb_', '')}] 的文件对")

    print(f"\n完成！")
    print(f"图片目录: {img_out}")
    print(f"深度目录: {depth_out}")
# --- 配置参数 ---
source_folder = "/home/rossi/dataset/private/1"  # 存放原始文件的文件夹
target_folder = "/home/rossi/dataset/private/sampled_data" # 存放结果的根目录

if __name__ == "__main__":
    sample_dataset(source_folder, target_folder, 10)
