"""海光 DCU 兼容（MossImage 适配新增，ComfyUI 加载自定义节点时执行，不注册任何节点）。

问题：启动参数 --enable-triton-backend 让 comfy-kitchen 的融合算子走 DAS triton，
RMSNorm/AdaLN/RoPE（bf16）内核工作正常、整体快约 10%；但 int8/fp8/nvfp4/w4a8 等量化相关的
triton 内核在 DAS triton 3.5.1 + DTK 26.04 上编译失败（LLVM 不认识 `getelementptr inbounds nuw`），
ComfyUI 进程直接段错误 —— 官方模板默认的 moss_image_*_int8_convrot / qwen3.5_9b_w4a8_gptq 都会触发。

处理：在 DCU 上把量化相关函数从 triton 后端的能力列表中移除，这些函数自动回落到 eager（纯 PyTorch）实现；
非量化的融合算子继续用 triton。设置 MOSS_DCU_TRITON_QUANT=1 可关闭本补丁。
"""
import logging
import os

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 已验证在 DCU triton 上可用的非量化融合算子；其余（量化相关）一律回落 eager
_SAFE_PREFIXES = ("adaln", "rms_adaln", "apply_rope", "rms_rope")


def _is_dcu():
    return (torch.cuda.is_available() and torch.version.hip is not None
            and ("+das" in torch.__version__ or os.path.isdir("/opt/dtk")))


def _patch():
    if os.environ.get("MOSS_DCU_TRITON_QUANT") == "1" or not _is_dcu():
        return
    try:
        import comfy_kitchen as ck
    except ImportError:
        return
    reg = ck.registry
    caps = reg._capabilities.get("triton")
    if not caps or "triton" in reg._disabled:
        return
    removed = sorted(f for f in caps if not f.startswith(_SAFE_PREFIXES))
    reg._capabilities["triton"] = {f for f in caps if f.startswith(_SAFE_PREFIXES)}
    logging.info("[MossImage DCU] comfy-kitchen triton 后端仅保留融合算子 %s；量化相关 %d 个函数改用 eager",
                 sorted(reg._capabilities["triton"]), len(removed))


_patch()
