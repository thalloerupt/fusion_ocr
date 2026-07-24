#!/usr/bin/env python3
"""从 PP-FormulaNet_plus-S.yml 中提取 UniMERNet byte-level BPE 词表，
按 token id 每行一个 token 写出，供 fusion_ocr 公式识别模块解码使用。

用法: python3 scripts/extract_unimernet_tokens.py
依赖: pyyaml（建议在虚拟环境中运行）
"""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YML = ROOT / "models/PP-FormulaNet_plus-S.yml"
OUT = ROOT / "models/unimernet_tokens.txt"


def main():
    with open(YML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tok = cfg["PostProcess"]["character_dict"]["fast_tokenizer_file"]
    vocab = tok["model"]["vocab"]  # {token: id}
    added = tok["added_tokens"]  # [{content, id, ...}]

    id2token = {i: t for t, i in vocab.items()}
    for t in added:
        id2token[t["id"]] = t["content"]

    max_id = max(id2token)
    assert len(id2token) == max_id + 1, "token id 不连续"

    with open(OUT, "w", encoding="utf-8") as f:
        for i in range(max_id + 1):
            f.write(id2token[i] + "\n")
    print(f"wrote {OUT} ({max_id + 1} tokens)")


if __name__ == "__main__":
    main()
