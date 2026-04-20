# Docker 使用说明

## 1. 安装 Docker 与 GPU 支持（若未安装）

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER   # 当前用户加入 docker 组，避免每次 sudo
# 重新登录或执行 newgrp docker
```

**NVIDIA GPU 支持（需先安装 NVIDIA 驱动与 Docker）：**
- 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 安装后可用 `docker run --gpus all ...` 在容器内使用 GPU

## 2. 构建镜像

在项目根目录执行（建议启用 BuildKit，避免 deprecated 提示）：

```bash
DOCKER_BUILDKIT=1 docker build -t unidet3d:latest .
```

首次构建会较久（需拉取基础镜像并安装 MinkowskiEngine、mmcv 等）。

## 3. 运行 `tools/test.py`

**方式一：使用脚本（推荐）**

```bash
# 使用默认 config 和 checkpoint（需先下载权重到 work_dirs/...）
./run_docker_test.sh

# 指定 config 和 checkpoint
./run_docker_test.sh configs/unidet3d_1xb8_scannet.py work_dirs/unidet3d_1xb8_scannet/latest.pth

# 带额外参数（如可视化）
./run_docker_test.sh configs/unidet3d_1xb8_scannet.py work_dirs/.../epoch_1024.pth --show-dir work_dirs/vis
```

**方式二：交互进入容器后手动执行**

```bash
docker run --rm -it --gpus all \
  -v "$(pwd):/workspace" \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  unidet3d:latest bash
# 在容器内：
python tools/test.py configs/unidet3d_1xb8_scannet.py work_dirs/.../epoch_1024.pth
```

**方式三：使用 docker-compose**

```bash
docker compose build
docker compose run --rm unidet3d bash
# 在容器内运行 test.py 同上
```

## 4. 数据与权重

- **数据**：按 README 将预处理后的数据放到 `data/` 下对应子目录（如 `data/scannet/`），或挂载到容器的 `/workspace/data`。
- **Checkpoint**：从 [Releases](https://github.com/filapro/unidet3d/releases) 下载 `unidet3d.pth` 等，放到 `work_dirs/` 下对应配置的目录，或挂载宿主机目录到 `/workspace/work_dirs`。

## 5. 常见问题

- **`docker: 未找到命令`**：未安装 Docker，见第 1 步。
- **`permission denied` / `connect: permission denied`（访问 docker.sock）**：
  - **临时解决**：用 sudo 运行，例如 `sudo docker build -t unidet3d:latest .`
  - **长期解决**：把当前用户加入 `docker` 组，之后无需 sudo：
    ```bash
    sudo usermod -aG docker $USER
    ```
    然后**重新登录**当前用户（或执行 `newgrp docker` 使当前 shell 生效），再运行 `docker build ...`。
- **Legacy builder deprecated 提示**：使用 BuildKit 构建即可：`DOCKER_BUILDKIT=1 docker build -t unidet3d:latest .`
- **`context deadline exceeded` / 拉取镜像超时（registry-1.docker.io）**：多为访问 Docker Hub 慢或被限，可配置**镜像加速**：
  1. 编辑（若无则创建）`/etc/docker/daemon.json`，内容示例（可只保留 `registry-mirrors`）：
     ```json
     {
       "registry-mirrors": [
         "https://docker.1ms.run",
         "https://docker.xuanyuan.me"
       ],
       "insecure-registries": []
     }
     ```
  2. 重启 Docker：`sudo systemctl restart docker`
  3. 再执行：`DOCKER_BUILDKIT=1 docker build -t unidet3d:latest .`
  - 若使用云主机，可改用对应厂商的镜像地址（如阿里云、腾讯云容器镜像服务里的“镜像加速器”）。
- **GPU 不可用**：安装 NVIDIA Container Toolkit 并确保 `docker run` 使用 `--gpus all`。
- **找不到 checkpoint**：确保第二个参数指向的 `.pth` 文件存在；可先只跑 `python tools/test.py --help` 确认脚本正常。
