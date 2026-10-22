"""ai-toolkit 在海光 DCU 上的兼容补丁入口（MossImage 适配新增，由 run.py 在启动时调用）。

1. 注意力反向：DAS flash 反向在大 logit 下 dq 错误/NaN -> 见 toolkit/dcu_sdpa.py
2. ConvRot 量化（UI 中 MOSS-Image 默认的 convrot8）：其 triton 融合内核（int8 激活量化等）在
   DAS triton 3.5.1 + DTK 26.04 上编译失败（LLVM 不认识 `getelementptr inbounds nuw`），进程直接 SIGSEGV。
   这里在 DCU 上关闭这些 triton 内核，改走 ai-toolkit 自带的纯 torch 实现（int8 矩阵乘仍用 torch._int_mm，
   DCU 上结果精确，约 150 TOPS），作者说明训练质量不受影响。
   设置 MOSS_DCU_CONVROT_TRITON=1 可恢复使用 triton 内核。
3. 进程退出：训练完成、LoRA 已保存后，进程在 libc exit() 卸载共享库（HIP 运行时析构）时段错误，
   退出码 139 并生成数 GB 的 core 文件。这里在 Python 正常清理（atexit）全部完成后用 os._exit 以原退出码
   直接结束进程，跳过出错的 C 层析构；未捕获异常 / sys.exit(n) 的退出码保持不变。
   设置 MOSS_DCU_FAST_EXIT=0 可关闭。
4. triton 编译崩溃（根因修复）：DAS triton 生成的 LLVM IR 带有 `getelementptr inbounds nuw` 等 DTK 26.04
   不认识的语法，任何新编译的 triton 内核都可能直接段错误。这里在交给 DTK 之前去掉这些标记
   （仅为优化提示，不改变计算结果）。设置 MOSS_DCU_TRITON_IR_FIX=0 可关闭。
"""
import atexit
import os
import re
import sys

from toolkit.dcu_sdpa import _is_dcu
from toolkit.dcu_sdpa import install as _install_sdpa


def _install_fast_exit():
    state = {"code": 0}
    orig_exit, orig_hook = sys.exit, sys.excepthook

    def _exit(code=0):
        state["code"] = code if isinstance(code, int) else (0 if code is None else 1)
        orig_exit(code)

    def _hook(t, v, tb):
        state["code"] = 1
        orig_hook(t, v, tb)

    def _final():  # 最先注册 => 最后执行（atexit 为后进先出）
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(state["code"])

    sys.exit, sys.excepthook = _exit, _hook
    atexit.register(_final)


_GEP_FLAGS = re.compile(r"\bgetelementptr((?: inbounds| nusw| nuw)+)")
_TRUNC_FLAGS = re.compile(r"\btrunc (?:nuw nsw |nsw nuw |nuw |nsw )")


def _sanitize_llvm_ir(src):
    src = _GEP_FLAGS.sub(lambda m: "getelementptr inbounds" if "inbounds" in m.group(1) else "getelementptr", src)
    src = _TRUNC_FLAGS.sub("trunc ", src)
    return src.replace("icmp samesign ", "icmp ")


def _install_triton_ir_fix():
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


def install():
    if not _is_dcu():
        return False
    if os.environ.get("MOSS_DCU_TRITON_IR_FIX", "1") != "0":
        _install_triton_ir_fix()
    if os.environ.get("MOSS_DCU_FAST_EXIT", "1") != "0":
        _install_fast_exit()
    _install_sdpa()
    if os.environ.get("MOSS_DCU_CONVROT_TRITON") != "1":
        try:
            import toolkit.util.convrot_quant as cq
            cq._triton_ok = False
        except Exception as e:  # 模块结构变化时不影响其他功能
            print(f"[MossImage] 跳过 convrot triton 补丁: {e}", flush=True)
    return True
