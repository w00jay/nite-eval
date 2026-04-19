#!/usr/bin/env python3
"""Dump and diff GGUF metadata for two model files.

Compares: general.* and tokenizer.* metadata, chat template, tokenizer
vocabulary hash, per-tensor quant type breakdown. Useful for understanding
why two ostensibly-equivalent quants (e.g. Q4_K_S vs Q4_K_M, or two providers'
takes on the same base model) actually differ.

Usage:
    uv run --with gguf python scripts/gguf_meta_diff.py \\
        MODEL_A.gguf MODEL_B.gguf --labels label_a label_b
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from gguf import GGUFReader


def summarize(path: str):
    reader = GGUFReader(path)

    meta: dict[str, object] = {}
    for name, field in reader.fields.items():
        if name.startswith(("general.", "tokenizer.ggml.")) or "." in name and name.split(".", 1)[0] not in {
            "tokenizer",
        }:
            try:
                value = field.contents()
            except Exception as exc:  # pragma: no cover
                value = f"<err: {exc}>"
            meta[name] = value

    chat_template = None
    ct_field = reader.fields.get("tokenizer.chat_template")
    if ct_field is not None:
        chat_template = ct_field.contents()

    tok_hash = None
    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field is not None:
        h = hashlib.sha256()
        for tok in tokens_field.contents():
            if isinstance(tok, str):
                h.update(tok.encode("utf-8", errors="replace"))
            else:
                h.update(bytes(tok))
        tok_hash = h.hexdigest()

    quant_counts: Counter[str] = Counter()
    quant_bytes: Counter[str] = Counter()
    for tensor in reader.tensors:
        name = tensor.tensor_type.name
        quant_counts[name] += 1
        quant_bytes[name] += int(tensor.n_bytes)

    return {
        "meta": meta,
        "chat_template": chat_template,
        "tokenizer_hash": tok_hash,
        "quant_counts": quant_counts,
        "quant_bytes": quant_bytes,
    }


def fmt_value(value: object) -> str:
    s = repr(value)
    if len(s) > 120:
        s = s[:117] + "..."
    return s


def short_hash(value: str | bytes | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        value = value.encode("utf-8", errors="replace")
    return hashlib.sha256(value).hexdigest()[:16]


def print_summary(label: str, path: str, summary: dict) -> None:
    print(f"\n--- {label}: {Path(path).name} ---")
    meta = summary["meta"]
    print(f"  general.architecture: {meta.get('general.architecture', '<missing>')}")
    print(f"  general.name:         {meta.get('general.name', '<missing>')}")
    print(f"  general.file_type:    {meta.get('general.file_type', '<missing>')}")
    print(f"  general.quant_ver:    {meta.get('general.quantization_version', '<missing>')}")
    print(f"  tokenizer.model:      {meta.get('tokenizer.ggml.model', '<missing>')}")
    print(f"  tokenizer.pre:        {meta.get('tokenizer.ggml.pre', '<missing>')}")
    print(f"  tokenizer.bos:        {meta.get('tokenizer.ggml.bos_token_id', '<missing>')}")
    print(f"  tokenizer.eos:        {meta.get('tokenizer.ggml.eos_token_id', '<missing>')}")
    print(f"  vocab sha256(16):     {short_hash(summary['tokenizer_hash'])}")
    ct = summary["chat_template"]
    if isinstance(ct, str):
        print(f"  chat_template:        len={len(ct)} sha256(16)={short_hash(ct)}")
    else:
        print(f"  chat_template:        <missing>")

    arch = meta.get("general.architecture")
    if isinstance(arch, str):
        print(f"  --- {arch}.* metadata ---")
        for key in sorted(k for k in meta if k.startswith(f"{arch}.")):
            print(f"    {key}: {fmt_value(meta[key])}")

    print(f"  --- tensor quant breakdown ---")
    counts = summary["quant_counts"]
    bytes_by = summary["quant_bytes"]
    total_bytes = sum(bytes_by.values())
    for qtype in sorted(counts):
        mb = bytes_by[qtype] / (1024 ** 2)
        pct = 100 * bytes_by[qtype] / total_bytes if total_bytes else 0
        print(f"    {qtype:12s}  {counts[qtype]:5d} tensors  {mb:8.0f} MB  ({pct:5.1f}%)")
    print(f"    total       :  {sum(counts.values()):5d} tensors  {total_bytes/(1024**2):8.0f} MB")


def diff_summaries(label_a: str, sa: dict, label_b: str, sb: dict) -> int:
    print("\n=== DIFF ===")
    diffs = 0

    if sa["tokenizer_hash"] != sb["tokenizer_hash"]:
        print(
            f"!! tokenizer vocab differs: "
            f"{label_a}={short_hash(sa['tokenizer_hash'])} "
            f"{label_b}={short_hash(sb['tokenizer_hash'])}"
        )
        diffs += 1

    cta, ctb = sa["chat_template"], sb["chat_template"]
    if isinstance(cta, str) and isinstance(ctb, str):
        if cta != ctb:
            print(f"!! chat_template differs: {label_a}_len={len(cta)} {label_b}_len={len(ctb)}")
            diffs += 1
    elif cta != ctb:
        print(f"!! chat_template presence differs: {label_a}={cta!r} {label_b}={ctb!r}")
        diffs += 1

    keys = sorted(set(sa["meta"]) | set(sb["meta"]))
    for key in keys:
        if key == "general.file_type":
            # file_type is the headline difference; surface it explicitly below
            continue
        va = sa["meta"].get(key, "<missing>")
        vb = sb["meta"].get(key, "<missing>")
        if va != vb:
            print(f"!! {key}: {label_a}={fmt_value(va)}  {label_b}={fmt_value(vb)}")
            diffs += 1

    fta = sa["meta"].get("general.file_type", "?")
    ftb = sb["meta"].get("general.file_type", "?")
    print(f"   general.file_type: {label_a}={fta}  {label_b}={ftb}")

    print("   --- per-tensor quant counts ---")
    all_types = sorted(set(sa["quant_counts"]) | set(sb["quant_counts"]))
    for qtype in all_types:
        ca = sa["quant_counts"].get(qtype, 0)
        cb = sb["quant_counts"].get(qtype, 0)
        ba = sa["quant_bytes"].get(qtype, 0) / (1024 ** 2)
        bb = sb["quant_bytes"].get(qtype, 0) / (1024 ** 2)
        marker = "!!" if ca != cb else "  "
        print(f"   {marker} {qtype:12s}  {label_a}={ca:5d}t/{ba:7.0f}MB   {label_b}={cb:5d}t/{bb:7.0f}MB")
        if ca != cb:
            diffs += 1

    print()
    if diffs == 0:
        print("(no differences in interesting fields — quants only differ by tensor *bit-width*, not structure)")
    else:
        print(f"({diffs} differences found)")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs=2, help="Two GGUF files to compare")
    parser.add_argument("--labels", nargs=2, required=True, help="Short labels for each model")
    args = parser.parse_args()

    summaries = []
    for path, label in zip(args.models, args.labels, strict=True):
        s = summarize(path)
        print_summary(label, path, s)
        summaries.append((label, s))

    diff_summaries(summaries[0][0], summaries[0][1], summaries[1][0], summaries[1][1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
