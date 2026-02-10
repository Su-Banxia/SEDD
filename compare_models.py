#!/usr/bin/env python3
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from load_model import load_model
import sampling

BASE_MODEL   = "sedd-small"  # 基础模型路径或名称
FINETUNE_DIR = "exp_local/data/alpaca/train.jsonl/2026.02.10/111627"  # 微调后的模型保存目录
VALID_FILE   = "data/alpaca/valid.jsonl"  # 验证集文件路径
NUM_SAMPLES  = 10      # 从验证集中采样的样本数量
LENGTH       = 256     # 生成文本的最大token长度
STEPS        = 256     # 采样步数
BATCH_SIZE   = 1       # 每个前缀生成的样本数
# 注意：这个脚本用于比较基础模型和微调后模型的生成效果，生成结果会保存到comparison_results目录下


def load_sedd(path, device):
    """加载SEDD模型并返回(model, graph, noise)三元组"""
    model, graph, noise = load_model(path, device)
    model.eval()
    return model, graph, noise


def generate(model, graph, noise, tokenizer, prefix, device):
    """根据给定的前缀（prompt）生成文本"""
    prefix_ids = tokenizer.encode(prefix)
    input_locs = list(range(len(prefix_ids)))
    input_tensor = torch.tensor(prefix_ids, device=device).unsqueeze(0).expand(BATCH_SIZE, -1)

    def proj_fun(x):
        x[:, input_locs] = input_tensor
        return x

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (BATCH_SIZE, LENGTH),
        'analytic', STEPS, device=device, proj_fun=proj_fun,
    )

    with torch.no_grad():
        samples = proj_fun(sampling_fn(model))

    return tokenizer.batch_decode(samples)


def main():
    device = torch.device("cuda")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # 创建输出目录：comparison_results/YYYYMMDD_HHMMSS/
    out_root = Path("comparison_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_root / "base").mkdir(parents=True, exist_ok=True)
    (out_root / "finetuned").mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_root}")

    # 加载基础模型和微调后的模型
    print("加载基础模型...")
    base_model, base_graph, base_noise = load_sedd(BASE_MODEL, device)

    print("加载微调后的模型...")
    ft_model, ft_graph, ft_noise = load_sedd(FINETUNE_DIR, device)

    # 从验证集读取样本并提取前缀（prompt）
    items = []
    with open(VALID_FILE) as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            idx = text.find("Output:")
            if idx == -1:
                continue
            items.append({"prefix": text[: idx + len("Output:")], "gt": text})
            if len(items) >= NUM_SAMPLES:
                break

    # 生成文本并保存结果
    all_results = []
    for i, item in enumerate(items):
        prefix, gt = item["prefix"], item["gt"]
        print(f"\n{'='*60}\n[{i}] Prefix: {prefix[:80]}...")

        base_out = generate(base_model, base_graph, base_noise, tokenizer, prefix, device)
        ft_out   = generate(ft_model, ft_graph, ft_noise, tokenizer, prefix, device)

        # Truncate content after endoftext token
        base_out = [t.split("<|endoftext|>")[0] for t in base_out]
        ft_out   = [t.split("<|endoftext|>")[0] for t in ft_out]

        print(f"Base Model:\t {base_out[0][:120]}...")
        print(f"Fintuned Model:\t {ft_out[0][:120]}...")

        record = {"id": i, "prefix": prefix, "ground_truth": gt,
                  "base": base_out, "finetuned": ft_out}
        all_results.append(record)

        # 将每个样本的生成结果保存为文本文件
        for tag, texts in [("base", base_out), ("finetuned", ft_out)]:
            with open(out_root / tag / f"sample_{i}.txt", "w") as fp:
                fp.write(f"~~~~~~Prefix~~~~~~:\n{prefix}\n\n~~~~~~GT~~~~~~:\n{gt}\n\n~~~~~~Result~~~~~~:\n")
                for j, t in enumerate(texts):
                    fp.write(f"\n--- variant {j} ---\n{t}\n\n")

    # 保存所有结果为JSON文件
    with open(out_root / "results.json", "w") as fp:
        json.dump(all_results, fp, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\n完成。结果已保存到 {out_root}/")


if __name__ == "__main__":
    main()