"""海光 DCU 兼容（MossImage 适配新增，ComfyUI 加载自定义节点时执行，不注册任何节点）。

1. triton 编译崩溃（根因修复）
   DAS triton 3.5.1 生成的 LLVM IR 带有 `getelementptr inbounds nuw` 等新语法，DTK 26.04 的编译器
   不认识（报 `addSource: error: expected type`），进程直接段错误。实测会触发的有：
   Prompt 增强（Qwen3.5 生成）用到的 RoPE 内核、int8/w4a8 量化内核等。
   这里在 triton 把 IR 交给 DTK 之前去掉这些标记（nuw/nusw/samesign 等只是优化提示，去掉不改变计算结果）。

2. comfy-kitchen 量化函数走 eager
   启动参数 --enable-triton-backend 让 RMSNorm/AdaLN/RoPE 等融合算子走 triton（整体快约 10%）；
   int8/fp8/nvfp4/w4a8 等量化相关的 triton 内核未在 DCU 上做数值验证，仍回落到 eager（纯 PyTorch）实现。

环境变量：MOSS_DCU_TRITON_IR_FIX=0 关闭 1；MOSS_DCU_TRITON_QUANT=1 关闭 2。
"""
import logging
import os
import re

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 已验证在 DCU triton 上可用的非量化融合算子；其余（量化相关）一律回落 eager
_SAFE_PREFIXES = ("adaln", "rms_adaln", "apply_rope", "rms_rope")

_GEP_FLAGS = re.compile(r"\bgetelementptr((?: inbounds| nusw| nuw)+)")
_TRUNC_FLAGS = re.compile(r"\btrunc (?:nuw nsw |nsw nuw |nuw |nsw )")


def _is_dcu():
    return (torch.cuda.is_available() and torch.version.hip is not None
            and ("+das" in torch.__version__ or os.path.isdir("/opt/dtk")))


def _sanitize_llvm_ir(src: str) -> str:
    """去掉 DTK 26.04 的 LLVM 不识别的指令标记（仅优化提示，语义不变）。"""
    src = _GEP_FLAGS.sub(lambda m: "getelementptr inbounds" if "inbounds" in m.group(1) else "getelementptr", src)
    src = _TRUNC_FLAGS.sub("trunc ", src)
    return src.replace("icmp samesign ", "icmp ")


def _patch_triton_ir():
    if os.environ.get("MOSS_DCU_TRITON_IR_FIX") == "0":
        return
    try:
        from triton.backends.amd import compiler as amdc
    except Exception:
        return
    if getattr(amdc.HIPBackend.make_amdgcn, "_moss_patched", False):
        return
    orig = amdc.HIPBackend.make_amdgcn

    def make_amdgcn(src, metadata, options):
        return orig(_sanitize_llvm_ir(src), metadata, options)

    make_amdgcn._moss_patched = True
    amdc.HIPBackend.make_amdgcn = staticmethod(make_amdgcn)
    logging.info("[MossImage DCU] 已启用 triton LLVM IR 兼容处理（修复 DTK 编译内核时的段错误）")


def _restrict_quant_to_eager():
    if os.environ.get("MOSS_DCU_TRITON_QUANT") == "1":
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


if _is_dcu():
    _patch_triton_ir()
    _restrict_quant_to_eager()
