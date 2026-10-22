#!/usr/bin/env python3
"""由官方/社区工作流生成海光 DCU 版本（bf16 权重 + 中文说明节点）。

官方模板默认用 int8 convrot DiT + w4a8 GPTQ 文本编码器（为 NVIDIA 小显存优化）。
DCU 上这些量化格式没有原生内核（comfy-kitchen 的 HIP 后端只支持官方 ROCm，回退到 eager 反量化），
K100 有 64GB 显存，直接用 bf16 权重更快、更准。

用法: python make_dcu_workflows.py <官方模板目录> <社区工作流目录> <输出目录>
"""
import copy, json, os, sys

HF = "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main"
SWAP = {
    "moss_image_sft_int8_convrot.safetensors": ("moss_image_sft_bf16.safetensors", "diffusion_models"),
    "moss_image_base_int8_convrot.safetensors": ("moss_image_base_bf16.safetensors", "diffusion_models"),
    "qwen3.5_9b_w4a8_gptq.safetensors": ("qwen3.5_9b_bf16.safetensors", "text_encoders"),
    "qwen3.5_9b_int8_convrot.safetensors": ("qwen3.5_9b_bf16.safetensors", "text_encoders"),
}

NOTE = """## 海光 DCU (K100) 版说明

本工作流由官方模板转换：**模型改为 bf16 权重**（DCU 上比 int8/w4a8 量化版更快、更准，64GB 显存足够）。

| 节点 | 文件 | 目录 |
|---|---|---|
| UNET Loader | {unet} | models/diffusion_models |
| CLIPLoader (moss_image) | qwen3.5_9b_bf16.safetensors | models/text_encoders |
| VAE Loader | {vae} | models/vae |

- 权重下载：终端执行 `moss download`（权重存放在 /root/private_data/MossImage/models，重启不丢）
- 速度参考（K100_AI，1024×1024，25 步，CFG 2.5）：约 35 秒/张；首次加载模型约 1~2 分钟
- **Prompt 增强**（enable_prompt_upsampler）用 Qwen3.5-9B 改写提示词，DCU 上约 10 token/s，会多花 30~60 秒；
  提示词已经很详细时可关闭
- 推荐分辨率 1~2MP（宽高为 16 的倍数）；步数 25~50，CFG 2.5~3
{extra}"""


def swap_models(obj):
    """递归替换所有出现的量化模型文件名（widgets_values / properties.models）。"""
    if isinstance(obj, dict):
        if obj.get("name") in SWAP and "url" in obj:
            new, d = SWAP[obj["name"]]
            obj["name"], obj["url"], obj["directory"] = new, f"{HF}/{d}/{new}", d
        for k, v in obj.items():
            if isinstance(v, str) and v in SWAP:
                obj[k] = SWAP[v][0]
            else:
                swap_models(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v in SWAP:
                obj[i] = SWAP[v][0]
            else:
                swap_models(v)


def add_note(d, text, pos=(-520, -40), size=(480, 560)):
    nid = max([n["id"] for n in d["nodes"]] + [d.get("last_node_id", 0)]) + 1
    d["nodes"].append({
        "id": nid, "type": "MarkdownNote", "pos": list(pos), "size": list(size), "flags": {}, "order": 0,
        "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": [text],
        "color": "#432", "bgcolor": "#653"})
    if "last_node_id" in d:
        d["last_node_id"] = nid


def model_names(d):
    s = json.dumps(d)
    unet = next((n for n in ("moss_image_sft_bf16.safetensors", "moss_image_base_bf16.safetensors") if n in s), "")
    vae = "flux2-vae_small.safetensors" if "flux2-vae_small" in s else "flux2-vae.safetensors"
    return unet, vae


def convert(src, dst, extra=""):
    d = json.load(open(src))
    swap_models(d)
    unet, vae = model_names(d)
    xs = [n["pos"][0] for n in d["nodes"] if isinstance(n.get("pos"), list)]
    ys = [n["pos"][1] for n in d["nodes"] if isinstance(n.get("pos"), list)]
    add_note(d, NOTE.format(unet=unet, vae=vae, extra=extra), pos=(min(xs) - 540, min(ys)))
    json.dump(d, open(dst, "w"), ensure_ascii=False, indent=1)
    left = [k for k in SWAP if k in json.dumps(d).replace("\\u", "")
            and f'"{k}"' in json.dumps(d)]
    print(f"{os.path.basename(dst)}: unet={unet} vae={vae} 残留量化文件名={left}")


def main():
    off, com, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    jobs = [
        ("image_moss_image_t2i.json", "MOSS-Image_SFT_文生图_DCU.json", ""),
        ("image_moss_image_base_t2i.json", "MOSS-Image_Base_文生图_DCU.json",
         "- Base 为预训练模型（未做 SFT/蒸馏），适合作为 LoRA 训练底模；直接出图质量略低于 SFT\n"),
        ("image_moss_image_t2i_bboxes.json", "MOSS-Image_SFT_布局控制_DCU.json",
         "- 在 Create Bounding Boxes 节点里画框并填写每个框的内容/文字，MOSS-Image Layout Prompt 会转成模型读取的布局 JSON\n"),
        ("image_moss_image_base_t2i_bboxes.json", "MOSS-Image_Base_布局控制_DCU.json",
         "- 在 Create Bounding Boxes 节点里画框并填写每个框的内容/文字\n"),
    ]
    for s, d, extra in jobs:
        convert(os.path.join(off, s), os.path.join(out, d), extra)
    convert(os.path.join(com, "moss_banding_fix_presets.json"), os.path.join(out, "社区_DeBand去条纹与加速预设_DCU.json"),
            "- 社区热门工作流：MATLOWAI/MOSS-Image-SFT-DeBand（去除 SFT 的竖条纹）+ lodestones 8 步 DMD / darask0 4 步加速 LoRA\n"
            "- 所需 LoRA：终端执行 `moss download community-loras`\n"
            "- 顶部 Preset 下拉框切换：Quality(25步) / Quality+ / Balanced(14步) / Fast(10步) / Fast alt / Off\n")


if __name__ == "__main__":
    main()
