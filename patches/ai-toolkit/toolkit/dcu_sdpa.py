"""海光 DCU 训练用的精确 SDPA 反向（MossImage 适配新增）。

问题（K100_AI + DAS torch 2.9.0 + DAS flash_attn 2.8.3，bf16）：
  SDPA flash 后端的**反向**在 MOSS-Image 这类“注意力 logit 很大（实测达 ~570，softmax 近乎 one-hot）”的
  真实数据上算出的 dq 严重错误（与 fp32 真值的余弦相似度仅 0.006~0.4，个别调用直接 NaN），dk 也有偏差；
  dv 与前向输出正确。训练表现为：第 22 层 dq 出现 NaN -> 传遍所有层 -> LoRA 第 1 步即被污染 -> loss 恒为 NaN。
  随机数据测不出来；math 后端结果正确但训练慢约 8 倍（1.07 -> 8.9 s/it）。

原因（两点）：
  1. flash 反向内部用 D = rowsum(dO * O) 计算 dS = P * (dP - D)。softmax 近 one-hot 时 dP - D 发生灾难性抵消，
     而 O 只以 bf16（7 位尾数）保存，舍入误差淹没了真实梯度。
  2. q/k/v 为非连续视图（模型里是 (B,L,H,D) 的转置）时，DAS flash 反向的 dq 误差进一步放大。

方案（训练时、需要梯度、无 mask、bf16/fp16 输入时生效）：
  - 前向：原样走 flash（已验证正确且快）
  - 反向：利用注意力对 V 的平移不变性（softmax 每行和为 1），先减去常向量 c = mean_L(O)，缩小 |O|；
    转成 fp16（尾数多 3 位，舍入误差降为 1/8），梯度按 2 的幂放大避免 fp16 下溢，转连续内存后调用 flash 反向，
    再缩放回来。实测最差调用 dq/dk/dv 与 fp32 真值余弦 0.9995/0.99996/1.0，耗时约 16ms（math 约 75ms）。
  - 兜底：输入超出 fp16 安全范围或结果出现 NaN/Inf 时，该次调用改用 math 后端重算。
推理（不需要梯度）时直接调用原始 SDPA，零开销。

环境变量：
  MOSS_DCU_SAFE_SDPA=0   关闭本补丁（恢复原始 flash 反向）
  MOSS_DCU_SDPA_MODE=math  反向全部改用 math（最稳，最慢）
  MOSS_DCU_SDPA_CHECK=1  每次调用都与 math 反向对比并打印余弦相似度（调试用，很慢）
"""
import math
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_orig_sdpa = F.scaled_dot_product_attention
_stats = {"calls": 0, "fallbacks": 0}
_FP16_SAFE = 30000.0


def _is_dcu() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return "+das" in torch.__version__ or os.path.isdir("/opt/dtk") or os.path.isdir("/opt/hyhal")


def _finite(*ts):
    return all(t is None or bool(torch.isfinite(t).all()) for t in ts)


def _grads(q, k, v, g, kw, backends):
    with torch.enable_grad():
        qq, kk, vv = (t.detach().requires_grad_() for t in (q, k, v))
        with sdpa_kernel(backends):
            o = _orig_sdpa(qq, kk, vv, **kw)
        return list(torch.autograd.grad(o, (qq, kk, vv), g))


def _math_bwd(q, k, v, g, kw):
    return _grads(q, k, v, g, kw, [SDPBackend.MATH])


def _fp16_flash_bwd(q, k, v, o, g, kw):
    """返回 (dq, dk, dv)，无法安全计算时返回 None。"""
    c = o.float().mean(-2, keepdim=True)                       # V 平移常量（不参与求导）
    vc = v.float() - c
    amax = torch.stack([q.abs().amax().float(), k.abs().amax().float(), vc.abs().amax()]).max().item()
    if not math.isfinite(amax) or amax > _FP16_SAFE:
        return None
    gmax = g.abs().amax().float().item()
    if gmax == 0.0:
        return [torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)]
    if not math.isfinite(gmax):
        return None
    s = 2.0 ** math.floor(math.log2(1.0 / gmax))               # 2 的幂缩放，无额外舍入
    qh, kh, vh = (t.to(torch.float16).contiguous() for t in (q, k, vc))
    gh = (g.float() * s).to(torch.float16).contiguous()
    dq, dk, dv = _grads(qh, kh, vh, gh, kw, [SDPBackend.FLASH_ATTENTION])
    return [(x.float() / s).to(t.dtype) for x, t in zip((dq, dk, dv), (q, k, v))]


class _DcuSDPA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal, scale, enable_gqa):
        kw = dict(is_causal=is_causal, scale=scale)
        if enable_gqa:
            kw["enable_gqa"] = True
        ctx.kw = kw
        with torch.no_grad():
            o = _orig_sdpa(q, k, v, **kw)
        ctx.save_for_backward(q, k, v, o)
        return o

    @staticmethod
    def backward(ctx, g):
        q, k, v, o = ctx.saved_tensors
        kw = ctx.kw
        _stats["calls"] += 1
        res = None
        if os.environ.get("MOSS_DCU_SDPA_MODE", "fp16") != "math":
            try:
                res = _fp16_flash_bwd(q, k, v, o, g, kw)
            except RuntimeError as e:  # 例如该形状 flash 不支持
                if _stats["fallbacks"] < 3:
                    print(f"[dcu_sdpa] fp16 flash 反向失败，改用 math: {str(e)[:120]}", flush=True)
                res = None
            if res is not None and not _finite(*res):
                res = None
            if res is None:
                _stats["fallbacks"] += 1
                if _stats["fallbacks"] <= 3:
                    print(f"[dcu_sdpa] 回退 math 反向 (shape={tuple(q.shape)})", flush=True)
        if res is None:
            res = _math_bwd(q, k, v, g, kw)
        if os.environ.get("MOSS_DCU_SDPA_CHECK") == "1":
            ref = _math_bwd(q.float(), k.float(), v.float(), g.float(), kw)
            cos = [F.cosine_similarity(a.float().flatten(), b.flatten(), 0).item() for a, b in zip(res, ref)]
            print(f"[dcu_sdpa] call {_stats['calls']} shape={tuple(q.shape)} cos(vs fp32 math) dq/dk/dv="
                  + " ".join(f"{x:.5f}" for x in cos), flush=True)
        return res[0], res[1], res[2], None, None, None


def safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if (torch.is_grad_enabled() and attn_mask is None and dropout_p == 0.0
            and query.dtype in (torch.bfloat16, torch.float16) and query.is_cuda
            and (query.requires_grad or key.requires_grad or value.requires_grad)):
        return _DcuSDPA.apply(query, key, value, is_causal, scale, enable_gqa)
    kw = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    if enable_gqa:
        kw["enable_gqa"] = True
    return _orig_sdpa(query, key, value, **kw)


def install():
    if os.environ.get("MOSS_DCU_SAFE_SDPA", "1") == "0" or not _is_dcu():
        return False
    if F.scaled_dot_product_attention is not safe_sdpa:
        F.scaled_dot_product_attention = safe_sdpa
        torch.nn.functional.scaled_dot_product_attention = safe_sdpa
        print("[MossImage] 已启用 DCU 精确 SDPA 反向（修复 DAS flash 反向在大 logit 下的 dq 错误/NaN）", flush=True)
    return True
