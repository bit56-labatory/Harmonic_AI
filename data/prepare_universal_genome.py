import argparse
import os
import random
import numpy as np
import tiktoken
from datasets import load_dataset

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC._special_tokens["<|endoftext|>"]
OUT_DIR = "data/universal_genome"


def fetch_pillar(ds_name: str, cfg: str, col: str, skip: int, count: int) -> list[str]:
    ds = (
        load_dataset(ds_name, cfg, split="train", streaming=True).skip(skip)
        if cfg
        else load_dataset(ds_name, split="train", streaming=True).skip(skip)
    )
    docs = []
    for row in ds:
        txt = str(row.get(col, "")).strip()
        if len(txt) >= 200:
            docs.append(txt[:8000])
            if len(docs) >= count:
                break
    return docs


def encode_to_bin(docs: list[str], path: str, chunk_size: int = 2000):
    random.shuffle(docs)
    total_toks = 0
    with open(path, "wb") as f:
        for i in range(0, len(docs), chunk_size):
            buf = []
            for d in docs[i : i + chunk_size]:
                buf.extend(ENC.encode_ordinary(d))
                buf.append(EOT)
            arr = np.array(buf, dtype=np.uint16)
            arr.tofile(f)
            total_toks += len(arr)
    print(f"Saved {total_toks:,} tokens ({len(docs):,} docs, {total_toks * 2 / 1e6:.1f} MB) -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_docs", type=int, default=36000, help="Docs to skip per pillar for fresh data")
    parser.add_argument("--docs_per_set", type=int, default=65000, help="Total training docs to pack (~50M tokens)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    val_path = os.path.join(OUT_DIR, "val.bin")
    train_path = os.path.join(OUT_DIR, "train.bin")

    pillars = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 0.40),
        ("HuggingFaceTB/cosmopedia", "web_samples_v2", "text", 0.25),
        ("codeparrot/codeparrot-clean", None, "content", 0.20),
        ("meta-math/MetaMathQA", None, "response", 0.15),
    ]

    if not os.path.exists(val_path):
        print("Building fixed textbook validation set (val.bin)...")
        val_docs = []
        for ds_name, cfg, col, _ in pillars:
            val_docs.extend(fetch_pillar(ds_name, cfg, col, skip=0, count=600))
        encode_to_bin(val_docs, val_path)
    else:
        print(f"Using existing fixed validation set -> {val_path}")

    print(f"Building fresh training buffer (skip_docs={args.skip_docs:,}, target_docs={args.docs_per_set:,})...")
    train_docs = []
    for ds_name, cfg, col, weight in pillars:
        target = int(args.docs_per_set * weight)
        print(f"  -> Fetching {target:,} fresh docs from {ds_name} (skip={args.skip_docs:,})...")
        train_docs.extend(fetch_pillar(ds_name, cfg, col, skip=args.skip_docs, count=target))
    encode_to_bin(train_docs, train_path)


if __name__ == "__main__":
    main()