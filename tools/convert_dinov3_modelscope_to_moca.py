#!/usr/bin/env python3
"""Convert ModelScope/HF DINOv3 ViT-L/16 weights to MoCA3D local format.

ModelScope mirrors the HuggingFace Transformers checkpoint with keys such as
``embeddings.patch_embeddings.weight`` and ``layer.0.attention.q_proj.weight``.
MoCA3D vendors the facebookresearch DINOv3 implementation, which expects keys
such as ``patch_embed.proj.weight`` and fused ``blocks.0.attn.qkv.weight``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Convert DINOv3 ViT-L/16 ModelScope weights for MoCA3D")
    parser.add_argument(
        "--input",
        default="third_party/MoCA3D/checkpoints/modelscope_dinov3_vitl16/model.safetensors",
        help="ModelScope/HF model.safetensors file.",
    )
    parser.add_argument(
        "--output",
        default="third_party/MoCA3D/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="Output .pth file expected by MoCA3D.",
    )
    return parser.parse_args()


def layer_indices(state: dict[str, torch.Tensor]) -> list[int]:
    indices = set()
    for key in state:
        if key.startswith("layer."):
            parts = key.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
    return sorted(indices)


def convert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}

    out["cls_token"] = state["embeddings.cls_token"]
    out["storage_tokens"] = state["embeddings.register_tokens"]
    out["mask_token"] = state["embeddings.mask_token"].squeeze(1)
    out["patch_embed.proj.weight"] = state["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = state["embeddings.patch_embeddings.bias"]

    for idx in layer_indices(state):
        src = f"layer.{idx}"
        dst = f"blocks.{idx}"

        out[f"{dst}.norm1.weight"] = state[f"{src}.norm1.weight"]
        out[f"{dst}.norm1.bias"] = state[f"{src}.norm1.bias"]
        out[f"{dst}.norm2.weight"] = state[f"{src}.norm2.weight"]
        out[f"{dst}.norm2.bias"] = state[f"{src}.norm2.bias"]

        out[f"{dst}.attn.qkv.weight"] = torch.cat(
            [
                state[f"{src}.attention.q_proj.weight"],
                state[f"{src}.attention.k_proj.weight"],
                state[f"{src}.attention.v_proj.weight"],
            ],
            dim=0,
        )
        q_bias = state[f"{src}.attention.q_proj.bias"]
        k_bias = state.get(f"{src}.attention.k_proj.bias", torch.zeros_like(q_bias))
        v_bias = state[f"{src}.attention.v_proj.bias"]
        out[f"{dst}.attn.qkv.bias"] = torch.cat(
            [
                q_bias,
                k_bias,
                v_bias,
            ],
            dim=0,
        )
        out[f"{dst}.attn.proj.weight"] = state[f"{src}.attention.o_proj.weight"]
        out[f"{dst}.attn.proj.bias"] = state[f"{src}.attention.o_proj.bias"]

        out[f"{dst}.ls1.gamma"] = state[f"{src}.layer_scale1.lambda1"]
        out[f"{dst}.ls2.gamma"] = state[f"{src}.layer_scale2.lambda1"]

        out[f"{dst}.mlp.fc1.weight"] = state[f"{src}.mlp.up_proj.weight"]
        out[f"{dst}.mlp.fc1.bias"] = state[f"{src}.mlp.up_proj.bias"]
        out[f"{dst}.mlp.fc2.weight"] = state[f"{src}.mlp.down_proj.weight"]
        out[f"{dst}.mlp.fc2.bias"] = state[f"{src}.mlp.down_proj.bias"]

    out["norm.weight"] = state["norm.weight"]
    out["norm.bias"] = state["norm.bias"]
    return out


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise FileNotFoundError(src)

    state = load_file(str(src), device="cpu")
    converted = convert_state(state)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, dst)
    print(f"input: {src}")
    print(f"input_keys: {len(state)}")
    print(f"output: {dst}")
    print(f"output_keys: {len(converted)}")
    print(f"output_size: {dst.stat().st_size}")


if __name__ == "__main__":
    main()
