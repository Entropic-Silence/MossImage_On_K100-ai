#!/usr/bin/env python3
"""MOSS-Image 权重下载器（多线程分片 + 断点续传 + sha256 校验）。

源: hf-mirror.com（默认）/ huggingface.co / modelscope.cn
用法:
  python moss_download.py --set sft            # SFT 推理/训练所需（DiT + 文本编码器 + VAE，约 28GB）
  python moss_download.py --set base           # Base（约 28GB，与 sft 共用文本编码器/VAE）
  python moss_download.py --set all            # sft + base + 社区 LoRA（约 43GB）
  python moss_download.py --list               # 查看全部集合
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.request, urllib.parse

# ---------------------------------------------------------------- 下载集合定义
# (repo, 仓库内路径, 本地相对路径)
COMFY = "Comfy-Org/MOSS-Image"
SETS = {
    # ComfyUI 原生（官方 Comfy-Org 重打包），bf16 —— DCU 上推荐
    "comfy-bf16": [
        (COMFY, "diffusion_models/moss_image_sft_bf16.safetensors", "comfyui/diffusion_models/moss_image_sft_bf16.safetensors"),
        (COMFY, "diffusion_models/moss_image_base_bf16.safetensors", "comfyui/diffusion_models/moss_image_base_bf16.safetensors"),
        (COMFY, "text_encoders/qwen3.5_9b_bf16.safetensors", "comfyui/text_encoders/qwen3.5_9b_bf16.safetensors"),
        (COMFY, "vae/flux2-vae_small.safetensors", "comfyui/vae/flux2-vae_small.safetensors"),
        (COMFY, "vae/flux2-vae.safetensors", "comfyui/vae/flux2-vae.safetensors"),
    ],
    # 官方模板默认的量化文件（int8 convrot / w4a8 gptq）
    "comfy-quant": [
        (COMFY, "diffusion_models/moss_image_sft_int8_convrot.safetensors", "comfyui/diffusion_models/moss_image_sft_int8_convrot.safetensors"),
        (COMFY, "diffusion_models/moss_image_base_int8_convrot.safetensors", "comfyui/diffusion_models/moss_image_base_int8_convrot.safetensors"),
        (COMFY, "text_encoders/qwen3.5_9b_int8_convrot.safetensors", "comfyui/text_encoders/qwen3.5_9b_int8_convrot.safetensors"),
        (COMFY, "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors", "comfyui/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors"),
        (COMFY, "vae/flux2-vae_small.safetensors", "comfyui/vae/flux2-vae_small.safetensors"),
    ],
}
_SHARED = [x for x in SETS["comfy-bf16"] if not x[1].startswith("diffusion_models/")]
SETS["sft"] = [SETS["comfy-bf16"][0]] + _SHARED
SETS["base"] = [SETS["comfy-bf16"][1]] + _SHARED
# 社区热门 LoRA（MATLOWAI DeBand 预设工作流所需；文件名按其工作流要求重命名）
SETS["community-loras"] = [
    ("MATLOWAI/MOSS-Image-SFT-DeBand", "moss_image_sft_banding_fix_L37-41_bf16.safetensors", "comfyui/loras/moss_image_sft_banding_fix_L37-41_bf16.safetensors"),
    ("MATLOWAI/MOSS-Image-SFT-DeBand", "moss_image_sft_banding_fix_L32-36_extra_bf16.safetensors", "comfyui/loras/moss_image_sft_banding_fix_L32-36_extra_bf16.safetensors"),
    ("lodestones/MOSS-Image-SFT-8-steps-dmd", "lora_latest.safetensors", "comfyui/loras/lodestones_moss_sft_8step_dmd_lora_latest.safetensors"),
    ("darask0/MOSS-Image_4-step_adapter_lora", "lora.safetensors", "comfyui/loras/darask0_moss_sft_4step_dmad_lora.safetensors"),
]
SETS["all"] = SETS["comfy-bf16"] + SETS["community-loras"]
# diffusers 格式（官方 diffusers 管线 / DiffSynth 等使用；ComfyUI 与 ai-toolkit 不需要）：整仓下载，排除 assets
for _v in ("Base", "SFT"):
    SETS["diffusers-" + _v.lower()] = [("OpenMOSS-Team/MOSS-Image-" + _v, "*", "diffusers/MOSS-Image-" + _v)]

MS_REPO = {"Comfy-Org/MOSS-Image": "Comfy-Org/MOSS-Image",
           "OpenMOSS-Team/MOSS-Image-Base": "OpenMOSS/MOSS-Image-Base",
           "OpenMOSS-Team/MOSS-Image-SFT": "OpenMOSS/MOSS-Image-SFT"}

CHUNK = 64 << 20  # 修改会使已有 .chunks 续传记录失效


def http_get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "moss-dl/1.0", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def list_repo(src, repo):
    """返回 [(path, size, sha256)]"""
    if src == "modelscope" and repo not in MS_REPO:
        src = "hf-mirror"  # 该仓库在 ModelScope 无镜像
    if src == "modelscope":
        r = MS_REPO[repo]
        d = json.load(http_get(f"https://www.modelscope.cn/api/v1/models/{r}/repo/files?Recursive=true"))
        return [(f["Path"], f["Size"], f.get("Sha256") or None) for f in d["Data"]["Files"] if f["Type"] == "blob"]
    host = "https://hf-mirror.com" if src == "hf-mirror" else "https://huggingface.co"
    d = json.load(http_get(f"{host}/api/models/{repo}/tree/main?recursive=true"))
    return [(f["path"], f["size"], (f.get("lfs") or {}).get("oid")) for f in d if f["type"] == "file"]


def file_url(src, repo, path):
    q = urllib.parse.quote(path)
    if src == "modelscope" and repo not in MS_REPO:
        src = "hf-mirror"
    if src == "modelscope":
        return f"https://www.modelscope.cn/models/{MS_REPO[repo]}/resolve/master/{q}"
    host = "https://hf-mirror.com" if src == "hf-mirror" else "https://huggingface.co"
    return f"{host}/{repo}/resolve/main/{q}"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(urls, out, size, sha, workers):
    """urls: 同一文件的多个源（hf-mirror / modelscope），分片失败时轮换源重试"""
    if os.path.isfile(out) and os.path.getsize(out) == size:
        if not sha or os.path.exists(out + ".ok"):
            print(f"[skip] {out}"); return True
        if sha256_file(out) == sha:
            open(out + ".ok", "w").close(); print(f"[ok-existing] {out}"); return True
    os.makedirs(os.path.dirname(out), exist_ok=True)
    part = out + ".part"
    if not os.path.exists(part) or os.path.getsize(part) != size:
        with open(part, "ab") as f:
            f.truncate(size)
    state = out + ".chunks"  # 已完成分片记录
    done = set()
    if os.path.exists(state):
        done = {int(x) for x in open(state).read().split()}
    chunks = [i for i in range((size + CHUNK - 1) // CHUNK) if i not in done]
    lock = threading.Lock()
    prog = {"n": len(done) * CHUNK, "t": time.time()}
    err = []

    def worker():
        while True:
            with lock:
                if not chunks or len(err) > 20: return
                i = chunks.pop(0)
            s, e = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
            for attempt in range(30):
                try:
                    url = urls[attempt % len(urls)]
                    r = http_get(url, {"Range": f"bytes={s}-{e}"}, timeout=60)
                    if r.status != 206 and not (s == 0 and e == size - 1):
                        raise IOError(f"range not honored: HTTP {r.status}")
                    # 分块读取 + 单分片总时限，防止代理慢速“吊死”连接
                    buf, deadline = [], time.time() + 600
                    while True:
                        piece = r.read(1 << 20)
                        if not piece: break
                        buf.append(piece)
                        if time.time() > deadline:
                            raise IOError("chunk deadline exceeded")
                    data = b"".join(buf)
                    if len(data) != e - s + 1:
                        raise IOError(f"short read {len(data)}")
                    with open(part, "r+b") as f:
                        f.seek(s); f.write(data)
                    with lock:
                        with open(state, "a") as f: f.write(f"{i}\n")
                        prog["n"] += len(data)
                    break
                except Exception as ex:  # noqa
                    time.sleep(min(20, 2 + attempt))
            else:
                with lock: err.append(i)

    ths = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in ths: t.start()
    while any(t.is_alive() for t in ths):
        time.sleep(15)
        el = time.time() - prog["t"]
        print(f"  {os.path.basename(out)} {prog['n']/size*100:5.1f}% {prog['n']/max(el,1)/1e6:6.1f} MB/s", flush=True)
    if err:
        print(f"[fail] {out}: {len(err)} chunks failed, 重新运行可续传"); return False
    if sha:
        got = sha256_file(part)
        if got != sha:
            print(f"[badsha] {out} {got} != {sha}"); os.remove(state); return False
    os.replace(part, out)
    if os.path.exists(state): os.remove(state)
    open(out + ".ok", "w").close()
    print(f"[done] {out}")
    return True


def small_get(url, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for a in range(10):
        try:
            data = http_get(url).read()
            open(out, "wb").write(data); return True
        except Exception:
            time.sleep(3)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", help=",".join(SETS))
    ap.add_argument("--dest", default=os.environ.get("MOSS_MODELS", "/root/private_data/MossImage/models"))
    ap.add_argument("--source", default=os.environ.get("MOSS_SOURCE", "hf-mirror"), choices=["hf-mirror", "huggingface", "modelscope"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.set:
        for k, v in SETS.items():
            print(k); [print("   ", r, p) for r, p, _ in v]
        return
    ok = True
    listing = {}
    for name in [s for x in a.set for s in x.split(",")]:
        for repo, path, local in SETS[name]:
            if repo not in listing:
                listing[repo] = {p: (sz, sh) for p, sz, sh in list_repo(a.source, repo)}
            files = listing[repo]
            targets = [(p, os.path.join(local, p)) for p in files if not p.startswith("assets/")] if path == "*" else [(path, local)]
            for p, lp in targets:
                sz, sh = files[p]
                out = os.path.join(a.dest, lp)
                urls = [file_url(a.source, repo, p)]
                alt = "modelscope" if a.source != "modelscope" else "hf-mirror"
                if repo in MS_REPO:
                    urls.append(file_url(alt, repo, p))
                if sz < (8 << 20):
                    if not (os.path.isfile(out) and os.path.getsize(out) == sz):
                        ok &= small_get(urls[0], out)
                    continue
                ok &= download(urls, out, sz, sh, a.workers)
    print("ALL_DONE" if ok else "SOME_FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
