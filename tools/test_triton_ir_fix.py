#!/usr/bin/env python3
"""验证 DAS triton 的 LLVM IR 兼容处理：去掉 DTK 不识别的 GEP nuw/nusw 等标记后，
原先编译崩溃的 comfy-kitchen triton 内核（apply_rope_split_half 等）能否编译、结果是否与 eager 一致。"""
import os, re, sys, torch
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_ir_fix_test")

_GEP = re.compile(r"\bgetelementptr((?: inbounds| nusw| nuw)+)")
def sanitize(src):
    src = _GEP.sub(lambda m: "getelementptr inbounds" if "inbounds" in m.group(1) else "getelementptr", src)
    src = re.sub(r"\btrunc (?:nuw nsw |nsw nuw |nuw |nsw )", "trunc ", src)
    return re.sub(r"\bicmp samesign ", "icmp ", src)

if os.environ.get("NO_FIX") != "1":
    from triton.backends.amd import compiler as amdc
    _orig = amdc.HIPBackend.make_amdgcn
    amdc.HIPBackend.make_amdgcn = staticmethod(lambda src, metadata, options: _orig(sanitize(src), metadata, options))
    print("[test] IR 兼容处理已启用")

import comfy_kitchen as ck
torch.manual_seed(0)
dev = "cuda"
# Qwen3.5 文本编码器的部分 RoPE：预填充(多 token) 与 逐 token 生成(1 token) 两种形状
for name, (B, H, L, D) in {"prefill": (1, 16, 300, 64), "decode": (1, 16, 1, 64), "odd": (2, 8, 37, 64)}.items():
    for dt in (torch.float16, torch.bfloat16):
        q = torch.randn(B, H, L, D, device=dev, dtype=dt); k = torch.randn(B, H // 2, L, D, device=dev, dtype=dt)
        freqs = torch.randn(1, 1, L, D // 2, 2, 2, device=dev, dtype=torch.float32)
        with ck.registry.use_backend("eager"):
            rq, rk = ck.apply_rope_split_half(q, k, freqs)
        with ck.registry.use_backend("triton"):
            oq, ok = ck.apply_rope_split_half(q, k, freqs)
        torch.cuda.synchronize()
        err = max((oq.float() - rq.float()).abs().max().item(), (ok.float() - rk.float()).abs().max().item())
        print(f"apply_rope_split_half {name:8s} {str(dt):15s} triton vs eager max err = {err:.3e}")
print("DONE")
