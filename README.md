# MossImage_On_K100-ai

让 OpenMOSS **MOSS-Image**（Base / SFT）在海光 **K100_AI** 加速卡上完成推理（ComfyUI）与 LoRA 训练（ai-toolkit）的核心改动。

仓库只包含改动本身：补丁、DCU 版工作流、配置与依赖版本清单。ComfyUI、ai-toolkit 源码和模型权重需按下文版本自行获取。

## 适用环境

以下为作者验证所用的基础环境，环境一致即可直接使用本仓库：

| 项目 | 版本 |
|---|---|
| 加速卡 | K100_AI（gfx928，64GB） |
| 操作系统 | Ubuntu 22.04 |
| DTK | 26.04 |
| Python | 3.11.9 |
| PyTorch | 2.9.0+das.opt1.dtk2604（基础环境自带） |
| numpy | 1.25.0（基础环境自带） |

对应超算互联网基础镜像 `jupyterlab-pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11`。

## 依赖版本

### 基础 Python 中追加的海光 DAS 组件

来自光合开发者社区 DAS1.8（cp311），见 `requirements/base-das.txt`：

| 包 | 版本 |
|---|---|
| torchvision | 0.24.0+das.opt1.dtk2604.torch290 |
| triton | 3.5.1+das.opt1.dtk2604.torch290 |
| flash_attn | 2.8.3+das.opt1.dtk2604.torch290 |
| einops / pytest | 0.8.2 / 9.1.1（flash_attn 导入时需要） |

PyPI 上的同名包与 DAS 版 torch 不兼容，不能替代。

### 上游项目

| 项目 | 版本 |
|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | commit `37ff76b4`（0.42.0，frontend 1.57.1） |
| [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager) | commit `a74929c6`（3.42） |
| [ai-toolkit](https://github.com/ostris/ai-toolkit) | commit `04733873`（0.13.34） |
| [diffusers](https://github.com/huggingface/diffusers) | commit `c943837`（ai-toolkit 指定） |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | 0.50.2（可选，需按 gfx928 编译） |
| Node.js | 22.23.3（ai-toolkit Web 界面） |

### Python 包

ComfyUI 与 ai-toolkit 各用一个 venv（`--system-site-packages`，继承基础环境的 DAS 组件），完整版本见：

- `requirements/comfyui.txt`（50 个包）
- `requirements/ai-toolkit.txt`（82 个包）

主要版本：

| 环境 | 关键包 |
|---|---|
| ComfyUI | comfy-kitchen 0.2.39、comfy-aimdo 0.5.6、comfyui-frontend-package 1.57.1、comfyui-workflow-templates 0.11.93、av 18.1.0、numpy 1.26.4 |
| ai-toolkit | transformers 5.5.3、huggingface_hub 1.23.0、peft 0.18.1、accelerate 1.15.0、torchao 0.10.0、optimum-quanto 0.2.4、lycoris-lora 1.8.3、bitsandbytes 0.50.2、opencv-python 4.11.0.86、numpy 1.26.4 |

安装时注意：

- 始终带上约束文件：`pip install -c requirements/constraints-das.txt ...`。它锁定 DAS 组件及 `numpy<2`、`opencv<4.12`，防止被替换成官方 CUDA 版。
- **torchaudio**：两个环境都安装 `torchaudio==2.9.0 --no-deps`，再删除 `site-packages/torchaudio/lib/*.so`，以纯 Python 模式运行。没有对应的 DAS 版本，官方版的 C++ 扩展又无法加载。
- **bitsandbytes**（可选，用于 8bit 优化器）：先从 PyPI 安装 0.50.2，再用 `scripts/build_bnb_dcu.sh <源码目录> <venv 的 python>` 编译 gfx928 动态库并替换。不编译时训练请选 `adamw` 优化器。

## 仓库内容

```
patches/
  comfyui/custom_nodes/dcu_compat.py      放入 ComfyUI/custom_nodes/
  ai-toolkit/toolkit/dcu_sdpa.py          放入 ai-toolkit/toolkit/
  ai-toolkit/toolkit/dcu_patches.py       放入 ai-toolkit/toolkit/
  ai-toolkit/run.py.patch                 在 ai-toolkit 目录执行 patch -p1 < run.py.patch
workflows/                                DCU 版工作流（upstream/ 为官方与社区原版）
configs/
  moss_image_base_lora_dcu.yaml           命令行训练示例
  comfyui_extra_model_paths.yaml          ComfyUI 与 ai-toolkit 共用权重目录的示例
requirements/                             依赖版本清单与约束文件
scripts/
  build_bnb_dcu.sh                        为 DCU 编译 bitsandbytes
  moss_download.py                        权重下载（断点续传、sha256 校验、多源切换）
tools/make_dcu_workflows.py               由官方模板生成 DCU 版工作流
tools/test_triton_ir_fix.py               验证 triton IR 修复（不加修复时段错误，加上后与 eager 结果一致）
docs/MossImage_DCU技术手册.md             问题分析与实现细节
```

所有补丁只在检测到海光 DAS torch / DTK 时生效，其他平台不受影响。

## 使用

### ComfyUI 推理

1. 复制 `patches/comfyui/custom_nodes/dcu_compat.py` 到 `ComfyUI/custom_nodes/`。
2. 启动时加上两个参数：

   ```bash
   python main.py --listen 0.0.0.0 --port 8188 \
       --use-pytorch-cross-attention --enable-triton-backend \
       --extra-model-paths-config configs/comfyui_extra_model_paths.yaml
   ```

   | 参数 | 作用 |
   |---|---|
   | `--use-pytorch-cross-attention` | gfx928 不在 ComfyUI 的 AMD 白名单内，默认会用慢速注意力；开启后走 DAS flash_attn，注意力约快 15 倍 |
   | `--enable-triton-backend` | 融合算子（RMSNorm/AdaLN/RoPE）走 DAS triton，整体约快 10%；**必须配合 `dcu_compat.py`**，否则 Prompt 增强、int8/w4a8 模型等会在编译 triton 内核时段错误 |

3. 把 `workflows/*.json` 拖入 ComfyUI 使用：

   | 工作流 | 说明 |
   |---|---|
   | MOSS-Image_SFT_文生图_DCU | 官方 SFT 模板，改用 bf16 权重 |
   | MOSS-Image_Base_文生图_DCU | 官方 Base 模板 |
   | MOSS-Image_SFT_布局控制_DCU | 画框指定元素位置与文字 |
   | MOSS-Image_Base_布局控制_DCU | 同上，Base 模型 |
   | 社区_DeBand去条纹与加速预设_DCU | MATLOWAI DeBand + 8 步 / 4 步加速 LoRA |

### ai-toolkit 训练

1. 复制两个补丁文件到 `ai-toolkit/toolkit/`，并应用 `run.py.patch`。
2. Web 界面：在 `ai-toolkit/ui` 中 `npm ci && npm run build && npm run start`（端口 8675），启动前设置 `MODELS_PATH` 指向权重目录。新建任务选 **MOSS-Image Base**。64GB 显存下建议：

   | 设置 | 建议 |
   |---|---|
   | Quantize（transformer / text encoder） | 关闭（bf16 更快更准；保持默认量化也可训练） |
   | Low VRAM | 关闭 |
   | Cache Text Embeddings | 开启 |
   | Optimizer | `adamw8bit`（已编译 bitsandbytes）或 `adamw` |
   | rank / 学习率 / 步数 | 32 / 1e-4 / 1000~3000 |
   | Gradient Checkpointing | 开启（约 14GB）；关闭约快 26%，峰值约 56GB |

3. 命令行：修改 `configs/moss_image_base_lora_dcu.yaml` 中的路径后执行

   ```bash
   cd ai-toolkit && MODELS_PATH=<权重目录> ./venv/bin/python run.py <配置文件>
   ```

   YAML 中学习率写成小数（`0.0001`），写成 `1e-4` 会被解析为字符串。

训练得到的 LoRA 可直接在 ComfyUI 中用 LoraLoaderModelOnly 加载。

### 权重

ComfyUI 与 ai-toolkit 共用 [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) 的重打包权重：

| 文件 | 目录 | 大小 |
|---|---|---|
| moss_image_sft_bf16.safetensors | diffusion_models | 9.5GB |
| moss_image_base_bf16.safetensors | diffusion_models | 9.5GB |
| qwen3.5_9b_bf16.safetensors | text_encoders | 18GB |
| flux2-vae_small.safetensors / flux2-vae.safetensors | vae | 0.2 / 0.3GB |

也可以用下载脚本（默认走 hf-mirror，失败分片自动切换 ModelScope）：

```bash
python3 scripts/moss_download.py --set sft --dest <目录>     # 文件落在 <目录>/comfyui/...
python3 scripts/moss_download.py --list                      # 查看 base / all / community-loras 等集合
```

## 解决的问题

| 现象 | 处理 |
|---|---|
| 训练 loss 从第 3 步起恒为 NaN | DAS flash 注意力反向在大 logit 下 dq 错误。`dcu_sdpa.py` 前向仍用 flash，反向改为 V 平移 + fp16 + 梯度缩放后计算，异常时回退 math；与 fp32 结果余弦 ≥ 0.99997 |
| 开启 Prompt 增强或加载 int8/w4a8 模型时 ComfyUI 段错误 | DAS triton 生成的 LLVM IR 含 DTK 不识别的 `getelementptr inbounds nuw` 等标记，编译内核时崩溃。`dcu_compat.py` / `dcu_patches.py` 在交给 DTK 前去掉这些标记（仅优化提示，结果不变）；量化函数另外改用 PyTorch 实现 |
| ai-toolkit 默认的 convrot8 量化训练崩溃 | 同类原因；`dcu_patches.py` 修复 IR，并让 convrot 量化走 ai-toolkit 自带的 torch 实现 |
| 训练结束时段错误并生成数 GB core 文件 | `dcu_patches.py` 在清理完成后按原退出码结束进程 |
| 8bit 优化器报 invalid device function | 官方包不含 gfx928；用 `build_bnb_dcu.sh` 编译 |
| ComfyUI 注意力慢 | 启动参数 `--use-pytorch-cross-attention` |

详细分析见 [docs/MossImage_DCU技术手册.md](docs/MossImage_DCU技术手册.md)。

## 性能（K100_AI 单卡）

| 场景 | 速度 |
|---|---|
| SFT 文生图 1024×1024，25 步，bf16 | 约 35 秒/张 |
| 官方量化模板（int8 + w4a8） | 约 60 秒/张 |
| Base LoRA 训练，1024，rank 32，梯度检查点 | 约 4.1 秒/步，显存约 14GB |

## 许可与致谢

本仓库以 [Apache-2.0](LICENSE) 发布。上游项目遵循各自许可：MOSS-Image（Apache-2.0）、ComfyUI（GPL-3.0）、ai-toolkit（MIT）。工作流改编自 Comfy-Org/workflow_templates 与 MATLOWAI/MOSS-Image-SFT-DeBand；加速 LoRA 来自 lodestones 与 darask0。
