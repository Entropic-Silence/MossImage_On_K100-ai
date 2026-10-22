# MOSS-Image @ 海光 DCU 技术手册

> 适用范围：SCNet 镜像（超算互联网，异构加速卡 AI / K100_AI）与非 docker 安装（见仓库根目录 [README](../README.md)）
> 文档版本：2026-10-22。本手册以 SCNet 镜像的目录（`/root/MossImage`、`/root/private_data/MossImage`）为例；
> `moss` 命令及 `scripts/` 下的启动脚本只在 SCNet 镜像中提供；在其他相同基础环境中使用时，按仓库 README 应用补丁与启动参数即可。

---

## 目录

1. [MOSS-Image 模型简介](#1-moss-image-模型简介)
2. [镜像概览](#2-镜像概览)
3. [使用指南](#3-使用指南)
4. [实现方案](#4-实现方案)
5. [修改清单](#5-修改清单)
6. [问题与修复](#6-问题与修复)
7. [性能参考](#7-性能参考)
8. [附录](#8-附录)

---

## 1. MOSS-Image 模型简介

MOSS-Image 是 OpenMOSS 团队于 2026 年 10 月 16 日开源的文生图模型（Apache-2.0），技术报告见 arXiv 2610.23374。主打**中英文文字渲染**和**版面（布局）控制**。

### 1.1 结构

| 组件 | 说明 |
|---|---|
| 扩散主干（DiT） | 约 5B 参数，单流结构：42 个 Transformer 块，宽 3072，24 头 × 128 维；图像 token 与文本 token 拼接后一起计算，所有块共享一套 AdaLN 调制；文本先经过 2 层 refiner |
| 文本编码器 | Qwen3.5-9B（取聊天模板化提示词的最后一层隐状态），提示词最长约 4K token |
| VAE | FLUX.2 VAE（ComfyUI 模板默认使用 small decoder 版本） |
| 采样 | Flow Matching；官方推荐 Euler、50 步、CFG 3、shift 5；CFG 负分支跳过第 22 层（Skip Layer Guidance） |

### 1.2 两个模型

| 模型 | 定位 | 适用 |
|---|---|---|
| **MOSS-Image-SFT** | 在 Base 上做了监督微调，版面更规整、文字更清晰 | 日常出图首选 |
| **MOSS-Image-Base** | 预训练权重，未做后训练和 CFG 蒸馏 | LoRA / 全量微调的底模 |

### 1.3 输入形式

- 自然语言描述
- 标签序列（booru 风格）
- **布局 JSON**：`summary.caption` 描述整图，`items` 列出每个元素的框（`bbox_format: yxyx`）、描述和要渲染的文字。ComfyUI 中用 *Create Bounding Boxes* 画框，*MOSS-Image Layout Prompt* 节点自动生成该 JSON。

推荐分辨率 1~2MP（宽高为 16 的倍数，0.25~4MP 均可）。社区已知问题：SFT 在平滑区域（天空、雾、纯色背景）会出现细竖条纹，社区 DeBand LoRA 可修复（本镜像已内置对应工作流）。

---

## 2. 镜像概览

### 2.1 组件与版本

| 类别 | 内容 |
|---|---|
| 基础镜像 | jupyterlab-pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11 |
| 加速卡 | K100_AI（gfx928，64GB） |
| PyTorch | 2.9.0+das.opt1.dtk2604（海光 DAS 版） |
| DAS 组件（新增） | torchvision 0.24.0、triton 3.5.1、flash_attn 2.8.3（DAS1.8，cp311） |
| 推理 | ComfyUI 0.42.0（commit 37ff76b4）+ ComfyUI-Manager 3.42 |
| 训练 | ai-toolkit 0.13.34（commit 04733873）+ Web 界面（Next.js） |
| 其他 | Node.js 22.23.3、为 gfx928 编译的 bitsandbytes 0.50.2 |

### 2.2 目录布局

```
/root/MossImage/                      ← 系统盘，随镜像保存（约 4.1GB）
├── ComfyUI/                          ComfyUI 源码 + ComfyUI-Manager + dcu_compat 节点
├── ai-toolkit/                       ai-toolkit 源码（含 DCU 补丁）+ 已构建的 Web 界面
├── envs/comfyui  envs/aitoolkit      两个独立 venv（继承系统的 DAS torch）
├── node/                             Node.js
├── scripts/                          moss 命令、启动脚本、下载器、自检、环境变量
├── workflows/                        DCU 版工作流
├── configs/                          命令行训练示例配置
├── assets/                           训练数据库模板、内置分词器缓存
└── README.md  install.sh  patches/   文档、复现脚本、补丁

/root/private_data/MossImage/         ← 个人持久化目录，不进镜像，重启/换镜像都保留
├── models/comfyui/                   权重（diffusion_models / text_encoders / vae / loras）
├── comfyui/                          ComfyUI 输出、输入、用户设置与工作流
├── aitk/                             训练数据集、训练输出（LoRA）、任务数据库
├── hf_home/  cache/                  HF 缓存、triton/MIOpen 编译缓存
└── logs/                             日志
```

推理与训练**共用同一份权重**（Comfy-Org 重打包布局），不重复下载。

### 2.3 架构

```
            浏览器（SCNet“访问自定义服务” 或 SSH 隧道）
                 │ :8188                         │ :8675
        ┌────────▼─────────┐            ┌────────▼──────────┐
        │ ComfyUI          │            │ ai-toolkit WebUI  │
        │ + dcu_compat 节点 │            │ (Node.js 任务队列) │
        └────────┬─────────┘            └────────┬──────────┘
                 │ envs/comfyui                  │ 调用 run.py（envs/aitoolkit）
                 │                               │ + dcu_patches（精确注意力反向等）
        ┌────────▼───────────────────────────────▼──────────┐
        │  系统 Python 3.11：DAS torch 2.9 / torchvision /   │
        │  triton / flash_attn（PIP_CONSTRAINT 锁定）         │
        └────────────────────────┬───────────────────────────┘
                                 │
                 /root/private_data/MossImage/models/comfyui（共用权重）
```

---

## 3. 使用指南

### 3.1 创建实例

在 SCNet 镜像库中选择本镜像创建 Notebook，加速卡选“异构加速卡 AI”，打开 JupyterLab 终端。终端会显示提示：

```
================ MOSS-Image @ 海光 DCU ================
 推理: ComfyUI（端口 8188）   训练: ai-toolkit WebUI（端口 8675）   权重: 未下载
 输入 moss 打开菜单；常用: moss download sft | moss comfyui | moss train | moss check
```

### 3.2 下载权重（首次）

镜像内不含权重，首次运行需下载到个人目录（之后永久保留）：

```bash
moss download sft              # SFT 推理所需：DiT + 文本编码器 + VAE，约 28GB
moss download base             # Base 底模（训练 LoRA 用），额外约 10GB
moss download community-loras  # 社区工作流用的 4 个 LoRA，约 4.5GB
moss download all              # 以上全部（约 43GB）
moss download comfy-quant      # 可选：官方模板默认的 int8/w4a8 量化版
```

下载器特性：多线程分片、断点续传（中断后重新执行即可）、sha256 校验、hf-mirror 与 ModelScope 自动切换。手动下载地址见 README 第 3 节。

### 3.3 常用命令

| 命令 | 作用 |
|---|---|
| `moss` | 交互菜单（状态、启动、下载、自检） |
| `moss comfyui [start\|stop\|restart\|log]` | ComfyUI（默认端口 8188） |
| `moss train [start\|stop\|restart\|log]` | 训练 Web 界面（默认端口 8675） |
| `moss download <集合>` | 下载权重 |
| `moss status` | 服务与权重状态 |
| `moss check` | 环境自检（加速卡、DAS 组件、两个环境、权重） |

### 3.4 在浏览器打开界面

- **方式一（平台内置）**：Notebook 列表 →【访问自定义服务】→ 填端口 8188（ComfyUI）或 8675（训练界面）→ 打开返回的链接。平台同一时间只开放一个自定义端口，切换界面时改端口即可。
- **方式二（SSH 隧道，可同时使用两个界面）**：
  ```bash
  ssh -p <SSH端口> -L 8188:127.0.0.1:8188 -L 8675:127.0.0.1:8675 root@<SSH地址>
  ```
  然后本地访问 `http://127.0.0.1:8188` 和 `http://127.0.0.1:8675`。

训练界面可设访问口令：`AI_TOOLKIT_AUTH=你的口令 moss train restart`。

### 3.5 推理：ComfyUI

启动后在左侧“工作流”面板的 `MOSS-Image/` 目录中打开：

| 工作流 | 用途 |
|---|---|
| MOSS-Image_SFT_文生图_DCU | 日常出图（官方模板，bf16 权重） |
| MOSS-Image_Base_文生图_DCU | Base 模型出图 / 验证 LoRA |
| MOSS-Image_SFT_布局控制_DCU | 画框指定每个元素的位置、描述和文字 |
| MOSS-Image_Base_布局控制_DCU | 同上，Base 模型 |
| 社区_DeBand去条纹与加速预设_DCU | 去竖条纹 + 加速 LoRA，顶部下拉框切换 Quality / Quality+ / Balanced / Fast 预设 |

使用要点：

- **Prompt 增强**（工作流节点上的 `enable_prompt_upsampler`）：用 Qwen3.5-9B 把简短中文需求改写成详细英文描述，效果明显，但 DCU 上要多花 30~60 秒；提示词已经很详细时可关闭。
- 推荐参数：1~2MP，25~50 步，CFG 2.5~3，采样器 er_sde / euler。
- 加载自训 LoRA：在 UNET Loader 后加 **LoraLoaderModelOnly** 节点；LoRA 文件放到 `/root/private_data/MossImage/models/comfyui/loras/`。
- 官方原版模板（int8/w4a8 量化）可从“模板”菜单搜索 MOSS-Image 打开，能正常运行，但比 DCU 版慢约 40%。

### 3.6 训练：ai-toolkit Web 界面

1. `moss download base` 后执行 `moss train`，打开 8675 端口页面。建议先 `moss comfyui stop` 释放显存。
2. **Datasets** → 新建数据集，上传图片（jpg/png）和同名 `.txt` 标注；也可以直接把文件夹放到 `/root/private_data/MossImage/aitk/datasets/<名称>/`。
3. **New Job** → Model Architecture 选 **MOSS-Image Base**（推荐）或 **MOSS-Image SFT**。
4. DCU 推荐设置：

   | 项目 | 推荐值 | 说明 |
   |---|---|---|
   | Quantize（transformer / text encoder） | 关闭 | 64GB 显存无需量化，bf16 更快更准；保持默认量化也能训练，约慢 20% |
   | Low VRAM | 关闭 | |
   | Cache Text Embeddings | 开启 | 编码完成后卸载 Qwen3.5-9B |
   | Optimizer | adamw8bit 或 adamw | 已为 gfx928 编译 bitsandbytes |
   | LoRA rank / 学习率 / 步数 | 32 / 1e-4 / 1000~3000 | |
   | Gradient Checkpointing | 开启 | 约 14GB；1024 分辨率下关闭可提速约 26%（峰值约 56GB） |

5. 点击 Start，在任务页面查看日志、loss 和采样图；LoRA 输出在 `/root/private_data/MossImage/aitk/output/<任务名>/`。

命令行训练（不开界面）：

```bash
cd /root/MossImage/ai-toolkit
MODELS_PATH=/root/private_data/MossImage/models/comfyui \
  ./venv/bin/python run.py /root/MossImage/configs/moss_image_base_lora_dcu.yaml
```

> YAML 中学习率请写成小数（`lr: 0.0001`），PyYAML 会把 `1e-4` 解析成字符串，8bit 优化器会因此报错。

### 3.7 安装自定义节点 / 额外 Python 包

可以用 ComfyUI-Manager 安装自定义节点。环境变量 `PIP_CONSTRAINT` 已锁定 DAS 版 torch、torchvision、triton、flash_attn 和 `numpy<2`：如果某个插件要求替换这些包，安装会报依赖冲突并失败，而不会破坏环境。注意自定义节点装在系统盘，实例重启后需要重装（或重新保存镜像）。

---

## 4. 实现方案

### 4.1 环境：继承 DAS torch 的独立 venv

DAS 版 torch 与 PyPI 官方 CUDA 版 ABI 不兼容，任何依赖 torch 的包如果顺带拉取官方 torch 都会破坏环境。做法：

1. 从海光 DAS 源安装与 torch 2.9.0 / Python 3.11 匹配的 torchvision、triton、flash_attn 到系统 Python；
2. ComfyUI 与 ai-toolkit 各建一个 `--system-site-packages` 的 venv：共享系统的 DAS torch，其余依赖各装各的（ai-toolkit 需要 transformers 5.5.3 等特定版本，与 ComfyUI 互不干扰）；
3. 所有安装都带约束文件 `scripts/constraints-das.txt`，并通过 `PIP_CONSTRAINT`/`UV_CONSTRAINT` 对运行期安装同样生效。

### 4.2 权重共享与持久化

- ai-toolkit 的 MOSS-Image 实现直接读取 Comfy-Org 布局的权重（DiT 为 q/k/v 融合格式，与 ComfyUI 一致），因此推理和训练共用 `models/comfyui`，并且训练出的 LoRA 键名就是 ComfyUI 格式。
- ComfyUI 通过 `--extra-model-paths-config`（启动时按 `$MOSS_MODELS` 生成）指向持久化目录；`--output-directory`、`--user-directory` 等也全部放到持久化目录。
- ai-toolkit 的 SQLite 任务数据库以符号链接指向持久化目录，首次启动从镜像内模板复制并预置数据集/输出目录；分词器等 HF 缓存预置在镜像中，首次使用时复制到持久化目录，离线也能训练。

### 4.3 启动与运维脚本

| 脚本 | 作用 |
|---|---|
| `scripts/env.sh` | 公共环境：DTK、平台代理、HF 镜像、CPU 线程数（按 cgroup 配额）、编译缓存目录、pip 约束、`ulimit -c 0` |
| `scripts/moss` | 统一入口（菜单 + 子命令） |
| `scripts/start_comfyui.sh` | 生成模型路径配置、同步工作流、带 DCU 优化参数启动 ComfyUI |
| `scripts/start_aitoolkit.sh` | 准备数据库与目录、迁移数据库结构、启动 Web 界面 |
| `scripts/moss_download.py` | 权重下载器（分片续传、校验、多源切换） |
| `scripts/check_env.py` | 环境自检 |
| `scripts/prepare_image.sh` | 保存镜像前清理缓存/core 文件，检查无权重、无凭据 |
| `scripts/build_bnb_dcu.sh` | 为 gfx928 编译 bitsandbytes |
| `install.sh` | 从基础镜像一键复现全部安装（版本已固定） |

### 4.4 工作流转换

`tools/make_dcu_workflows.py` 从官方模板生成 DCU 版本：把子图内加载器、子图实例参数和缺失模型下载信息中的 int8/w4a8 文件名统一替换为 bf16，并插入一个中文说明节点。`tools/validate_workflows.py` 校验每个节点类型都存在、所选模型文件都在下拉列表中。

### 4.5 镜像打包

SCNet 的“保存镜像”会把系统盘改动提交为一层（单层上限 15GB，镜像总上限 50GB），`/root/private_data` 不进入镜像。因此：

- 代码、环境、Web 界面构建产物放系统盘：增量约 6.5GB（`/root/MossImage` 4.1GB + DAS flash_attn/triton/torchvision 约 2.4GB），加基础镜像约 29GB；
- 权重（约 43GB）不放入镜像，由 `moss download` 下载到使用者自己的持久化目录；
- 保存前运行 `prepare_image.sh` 清理 pip/npm 缓存、测试输出和 core 文件，并检查系统盘无大权重文件、无代理凭据。

---

## 5. 修改清单

对上游代码的改动全部保存在 `patches/`，均只在检测到海光 DCU 时生效，其他平台行为不变。

### 5.1 ai-toolkit

| 文件 | 类型 | 内容 |
|---|---|---|
| `toolkit/dcu_sdpa.py` | 新增 | DCU 精确注意力反向（见 6.1） |
| `toolkit/dcu_patches.py` | 新增 | 补丁入口：安装精确注意力反向；关闭 convrot 量化的 triton 内核；进程退出保护 |
| `run.py` | 修改 3 行 | 启动时调用 `dcu_patches.install()`（`patches/ai-toolkit/run.py.patch`） |

### 5.2 ComfyUI

| 文件 / 配置 | 类型 | 内容 |
|---|---|---|
| `custom_nodes/dcu_compat.py` | 新增 | 不注册节点，只在加载时把量化相关函数从 comfy-kitchen 的 triton 后端移除（见 6.3） |
| 启动参数 | 配置 | `--use-pytorch-cross-attention --enable-triton-backend`，输入/输出/用户/模型目录指向持久化目录 |
| 源码 | 未修改 | |

### 5.3 运行环境

| 项目 | 内容 |
|---|---|
| DAS 组件 | 安装 torchvision 0.24.0、triton 3.5.1、flash_attn 2.8.3（DAS1.8 cp311） |
| torchaudio | 安装官方 2.9.0 的 Python 部分，移除无法加载的 C++ 扩展（纯 Python 模式） |
| bitsandbytes | 用 DTK clang 为 gfx928 重新编译 HIP 后端 |
| 依赖约束 | `constraints-das.txt`：锁定 DAS 包、`numpy<2`、`opencv<4.12` |
| 工作流 | 5 个 DCU 版工作流（4 个官方 + 1 个社区） |

---

## 6. 问题与修复

### 6.1 训练 loss 从第 3 步起恒为 NaN（最关键）

**现象**：LoRA 训练前两步 loss 正常，之后一直是 NaN，LoRA 权重被污染。

**定位过程**：

1. 在模型各层挂前向/反向钩子，发现第一个非有限梯度出现在第 22 层注意力的 q 梯度（k、v 正常），随后传遍前面所有层；
2. 关闭 flash、改用 math 后端后训练正常，确认是 **DAS flash 注意力的反向** 问题；
3. 随机数据测试一切正常，于是导出一次真实出错调用的 q/k/v/梯度离线分析：MOSS-Image 的注意力 logit 高达约 570，softmax 近乎 one-hot。与 fp32 真值相比，flash 的 dq 余弦相似度只有 0.006，dk 为 0.975。

**原因**：flash 反向用 `D = rowsum(dO · O)` 计算 `dS = P · (dP − D)`。softmax 近 one-hot 时 `dP − D` 发生灾难性抵消，而 O 只以 bf16（7 位尾数）保存，舍入误差淹没了真实梯度；q/k/v 为非连续内存布局时误差进一步放大。

**修复**（`dcu_sdpa.py`，仅训练时生效，推理零开销）：

- 前向照常走 flash（已验证正确）；
- 反向：利用注意力对 V 的平移不变性先减去 `mean(O)` 缩小数值范围 → 转 fp16（尾数多 3 位）→ 梯度按 2 的幂放大避免下溢 → 转连续内存 → 调用 flash 反向 → 缩放回来；
- 输入超出 fp16 安全范围或结果非有限时，该次调用自动回退 math 后端。

**效果**：一个训练步内全部注意力调用的 dq/dk/dv 与 fp32 真值余弦 ≥ 0.99997；每次调用约 16ms（math 约 75ms），训练整体比 math 后端快约 2 倍（4.1 s/步 vs 8.9 s/步）。

### 6.2 ComfyUI 注意力慢

K100（gfx928）不在 ComfyUI 的 AMD 白名单中，默认退回 sub-quadratic 注意力。加 `--use-pytorch-cross-attention` 后 SDPA 调用 DAS flash_attn，注意力算子快约 15 倍；推理只有前向，不受 6.1 影响。

### 6.3 加载 int8 / w4a8 模型时 ComfyUI 段错误

comfy-kitchen 的 HIP 后端只支持官方 ROCm，因此启用 triton 后端（`--enable-triton-backend`）来加速 RMSNorm/AdaLN/RoPE 融合算子（整体快约 10%）。但其中的 int8/fp8/w4a8 量化 triton 内核在 DAS triton + DTK 上编译失败（LLVM 不认识 `getelementptr inbounds nuw`），进程直接段错误。官方模板默认的量化权重会触发这个问题。

**修复**：`dcu_compat.py` 只保留已验证的 bf16 融合算子，量化相关的 10 个函数回落到 eager 实现。官方模板因此可正常出图（2.3 s/步）；DCU 版工作流改用 bf16 权重（1.36 s/步）。

### 6.4 训练界面默认量化（convrot8）进程崩溃

原因与 6.3 相同：ai-toolkit 的 convrot int8 激活量化 triton 内核编译失败，训练进程 SIGSEGV。`dcu_patches.py` 在 DCU 上关闭这些内核，改走 ai-toolkit 自带的纯 torch 实现；int8 矩阵乘用 `torch._int_mm`，在 DCU 上结果精确（约 150 TOPS）。

### 6.4b 开启 Prompt 增强时 ComfyUI 段错误（triton IR 兼容，根因修复）

6.3、6.4 的崩溃其实是同一个根因：DAS triton 3.5.1 生成的 LLVM IR 带有 `getelementptr inbounds nuw` 等新语法，DTK 26.04 的编译器不认识（`addSource: error: expected type`），任何新编译的 triton 内核都可能直接段错误。实际触发过的还有 Prompt 增强（Qwen3.5 逐 token 生成）用到的 RoPE 内核。

**修复**：`dcu_compat.py`（ComfyUI）与 `dcu_patches.py`（ai-toolkit）在 triton 把 IR 交给 DTK 之前去掉 GEP 的 `nuw`/`nusw`、`trunc` 的 `nuw`/`nsw`、`icmp samesign` 等标记（只是优化提示，不改变计算结果）。对照测试：不加修复时进程段错误（退出码 139），加上后编译成功，结果与 eager 实现一致（`tools/test_triton_ir_fix.py`）。

此外，ComfyUI 启动脚本加了监护循环：进程异常退出时 3 秒后自动重启，避免服务挂掉后平台“访问自定义服务”检测不到服务。

### 6.5 训练结束时段错误并生成巨大 core 文件

训练完成、LoRA 已保存后，进程在 libc `exit()` 卸载 HIP 运行时时段错误，退出码 139，并在工作目录（系统盘）写出 2~12GB 的 core 文件。**修复**：Python 清理完成后用原退出码 `os._exit` 结束进程（异常退出码不变）；`env.sh` 设 `ulimit -c 0`；`prepare_image.sh` 清理残留 core 文件。

### 6.6 其他问题

| 问题 | 修复 |
|---|---|
| 容器无直连外网 | 自动加载平台提供的代理配置；HF 走 hf-mirror，PyPI 走阿里云，npm 走 npmmirror |
| 官方 PyPI 的 torchvision/triton/flash_attn 与 DAS torch 不兼容 | 改用 DAS1.8 的 cp311 版本，并用约束文件锁定 |
| hf-mirror 下 Xet 协议返回 401，ai-toolkit 下载分词器失败 | `HF_HUB_DISABLE_XET=1`，并在镜像中内置分词器 |
| torchaudio 无 cp311 DAS 版；ai-toolkit 顶层导入它；ComfyUI-Manager 缺它时关闭 torch 保护 | 两个环境都使用官方 torchaudio 的纯 Python 模式 |
| bitsandbytes 官方 ROCm 包不含 gfx928（invalid device function） | 为 gfx928 重新编译；AdamW8bit 与 AdamW 更新方向余弦 0.9999 |
| 下载代理偶发长时间卡住 | 下载器加单分片时限、hf-mirror/ModelScope 轮换重试 |
| opencv ≥4.12 需要 numpy≥2，pip 反复回溯 | 约束 `opencv<4.12`、`numpy<2` |
| Next.js 构建需下载 Google 字体 | 构建时加载代理；镜像内为构建产物，运行时无需联网 |
| 容器配额 7 核但 nproc=64，线程超订 | 按 cgroup 配额设置 `OMP_NUM_THREADS` |
| ComfyUI DynamicVRAM 仅支持 ROCm ≥ 7.14 | DCU 上自动使用传统显存管理，64GB 下无影响 |

---

## 7. 性能参考

K100_AI 单卡实测：

| 场景 | 设置 | 速度 | 显存 |
|---|---|---|---|
| SFT/Base 文生图（DCU 版工作流） | 1024²，25 步，CFG 2.5，er_sde，bf16 | 约 35 秒/张（1.36 s/步） | 约 30GB |
| 官方原版模板（int8 + w4a8） | 同上 | 约 60 秒/张（2.3 s/步） | 更低 |
| 社区 DeBand Fast 预设 | 10 步 + DMD LoRA | 采样约 15 秒 | — |
| Prompt 增强 | Qwen3.5-9B | 约 10 token/s | — |
| LoRA 训练（推荐配置） | Base，1024，rank 32，bf16，梯度检查点，adamw8bit | 约 4.1 s/步 | 训练中约 14GB |
| LoRA 训练，关闭梯度检查点 | 同上 | 约 3.0 s/步 | 峰值约 56GB |
| LoRA 训练，界面默认（convrot8 + low_vram） | 同上 | 约 5.1 s/步 | 更低 |

验证案例：用 10 张风格图训练 300 步（约 20 分钟）后，Base + LoRA 可稳定复现该风格，LoRA 在 ComfyUI 中直接加载，所有键正常匹配。

---

## 8. 附录

### 8.1 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MOSS_DATA` | `/root/private_data/MossImage` | 持久化数据根目录 |
| `MOSS_MODELS` | `$MOSS_DATA/models` | 权重目录 |
| `COMFY_PORT` / `AITK_PORT` | 8188 / 8675 | 服务端口 |
| `AI_TOOLKIT_AUTH` | 空 | 训练界面访问口令 |
| `MOSS_DL_WORKERS` | 8 | 下载线程数 |
| `MOSS_DCU_SAFE_SDPA` | 1 | 设为 0 关闭精确注意力反向（不建议） |
| `MOSS_DCU_SDPA_MODE` | fp16 | 设为 `math` 时反向全部用 math 后端（最稳最慢） |
| `MOSS_DCU_SDPA_CHECK` | 0 | 设为 1 时每次反向都与 fp32 对比并打印（调试用） |
| `MOSS_DCU_CONVROT_TRITON` | 0 | 设为 1 恢复 convrot 的 triton 内核（DCU 上会崩溃） |
| `MOSS_DCU_TRITON_QUANT` | 0 | 设为 1 恢复 comfy-kitchen 量化 triton 内核（DCU 上会崩溃） |
| `MOSS_DCU_TRITON_IR_FIX` | 1 | 设为 0 关闭 triton LLVM IR 兼容处理（不建议，部分内核会编译崩溃） |
| `MOSS_DCU_FAST_EXIT` | 1 | 设为 0 关闭训练进程退出保护 |

### 8.2 常见问题

- **打开页面后提示模型缺失**：执行 `moss status` 确认权重已下载；下载中断可重新执行 `moss download`。
- **训练时显存不足**：先 `moss comfyui stop`；开启梯度检查点；或在界面中开启量化与 Low VRAM。
- **装了自定义节点后环境异常**：`moss check` 自检；若因依赖冲突安装失败，属于约束保护的正常行为。
- **查看日志**：`moss comfyui log`、`moss train log`，或 `/root/private_data/MossImage/logs/`。

### 8.3 在相同基础环境中复现

依赖版本、补丁应用方法与启动参数见仓库根目录 [README](../README.md)。

### 8.4 参考链接

- 官方仓库：https://github.com/OpenMOSS/MOSS-Image
- 模型：https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT 、https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
- ComfyUI 重打包权重：https://huggingface.co/Comfy-Org/MOSS-Image
- ComfyUI 教程：https://docs.comfy.org/tutorials/image/moss-image/moss-image
- ai-toolkit：https://github.com/ostris/ai-toolkit
- 社区 DeBand：https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand
- 社区 8 步 DMD LoRA：https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd
