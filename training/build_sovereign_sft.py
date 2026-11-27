import argparse
import os
import random
import sys
import numpy as np
import tiktoken
import torch
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.harmonic_lm import HarmonicLM
import inspect
def _build_harmonic_from_cfg(cfg_dict):
    cfg = dict(cfg_dict)
    if "n_blocks" in cfg and "num_blocks" not in cfg:
        cfg["num_blocks"] = cfg.pop("n_blocks")
    if "n_heads" in cfg and "num_heads" not in cfg:
        cfg["num_heads"] = cfg.pop("n_heads")
    valid = inspect.signature(HarmonicLM.__init__).parameters
    return HarmonicLM(**{k: v for k, v in cfg.items() if k in valid})


ENC = tiktoken.get_encoding("gpt2")
EOT = ENC._special_tokens["<|endoftext|>"]

SOVEREIGN_TEMPLATES = [
    # --- Cross-OS CLI (Windows PowerShell, Linux Bash, macOS Zsh) ---
    "<|user|> [Windows | PowerShell] Show the top 5 processes using the most RAM.\n<|think|> Windows PowerShell requested. Sort Get-Process by WorkingSet descending and select top 5. </|think|>\n<|assistant|> ```powershell\nGet-Process | Sort-Object WorkingSet -Descending | Select-Object -First 5 Name, Id, @{N='RAM_MB';E={[math]::Round($_.WorkingSet/1MB,1)}}\n```",
    "<|user|> [Linux | bash] Show the top 5 processes using the most RAM.\n<|think|> Linux bash requested. Use ps aux sorted by memory percentage. </|think|>\n<|assistant|> ```bash\nps aux --sort=-%mem | head -n 6\n```",
    "<|user|> [Darwin | zsh] Show the top 5 processes using the most RAM.\n<|think|> macOS zsh requested. BSD ps uses -m to sort by memory. </|think|>\n<|assistant|> ```zsh\nps aux -m | head -n 6\n```",
    "<|user|> [Windows | PowerShell] Find the 10 largest files in the current directory recursively.\n<|think|> Recurse files with Get-ChildItem -File, sort Length descending, take first 10. </|think|>\n<|assistant|> ```powershell\nGet-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 10 FullName, @{N='MB';E={[math]::Round($_.Length/1MB,2)}}\n```",
    "<|user|> [Linux | bash] Find the 10 largest files in the current directory recursively.\n<|think|> Use find with du -h and sort -rh on Linux. </|think|>\n<|assistant|> ```bash\nfind . -type f -exec du -h {} + | sort -rh | head -n 10\n```",
    "<|user|> [Windows | PowerShell] Check NVIDIA GPU memory and utilization.\n<|think|> Query nvidia-smi directly for GPU name, utilization, and VRAM. </|think|>\n<|assistant|> ```powershell\nnvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader\n```",
    "<|user|> [Windows | PowerShell] List all listening TCP ports and the process IDs owning them.\n<|think|> Use Get-NetTCPConnection filtered by State Listen. </|think|>\n<|assistant|> ```powershell\nGet-NetTCPConnection -State Listen | Select-Object LocalAddress, LocalPort, OwningProcess | Sort-Object LocalPort\n```",
    "<|user|> [Linux | bash] List all listening TCP ports and their process names.\n<|think|> Use ss -tulnp on Linux. </|think|>\n<|assistant|> ```bash\nss -tulnp\n```",
    "<|user|> [Darwin | zsh] List all listening TCP ports on macOS.\n<|think|> Use lsof -iTCP -sTCP:LISTEN -nP on macOS. </|think|>\n<|assistant|> ```zsh\nlsof -iTCP -sTCP:LISTEN -nP\n```",
    "<|user|> [Windows | PowerShell] Search all .py files recursively for the word torchrun.\n<|think|> Use Get-ChildItem -Filter *.py piped into Select-String. </|think|>\n<|assistant|> ```powershell\nGet-ChildItem -Recurse -Filter *.py | Select-String -Pattern 'torchrun'\n```",
    "<|user|> [Linux | bash] Search all .py files recursively for the word torchrun.\n<|think|> Use grep -rn with --include='*.py'. </|think|>\n<|assistant|> ```bash\ngrep -rn --include='*.py' 'torchrun' .\n```",
    "<|user|> [Windows | PowerShell] Delete all __pycache__ folders recursively in the current project.\n<|think|> Find directories named __pycache__ and remove them cleanly. </|think|>\n<|assistant|> ```powershell\nGet-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force\n```",

    "<|user|> yo\n<|think|> Casual greeting. Respond concisely in Sovereign Core voice. </|think|>\n<|assistant|> Online and ready. What system, code, or analysis are we working on?",
    "<|user|> what is your name?\n<|think|> Direct identity query. State name and architecture clearly. </|think|>\n<|assistant|> I am HarmonicLM, a recursive reasoning language model built on Fibonacci-sparse mixing and Golden-Ratio scale-invariant blocks.",
    "<|user|> Who created you?\n<|think|> Provenance query. State Sovereign Core architecture and user-owned local runtime. </|think|>\n<|assistant|> I was engineered as a local Sovereign AI runtime built on the HarmonicLM recursive architecture, running directly on your hardware with persistent local memory.",
    "<|user_profile|>\n- role: Security & AI Systems Architect\n</|user_profile|>\n<|user|> Audit this SSH configuration: PermitRootLogin yes, PasswordAuthentication yes.\n<|think|> User is a Security & AI Systems Architect. Purple Team: trivial brute-force vector. Blue Team: disable root login and enforce Ed25519 keys. </|think|>\n<|assistant|> Risk: Critical. Set 'PermitRootLogin no' and 'PasswordAuthentication no' in /etc/ssh/sshd_config, enforce Ed25519 SSH keys, and reload sshd.",
    "<|user|> What architecture does Project Aurora use?\n<|tool_result|>\n[LOCAL MEMORY | user_taught | score=0.84]: Project Aurora uses 1/phi residual damping and runs locally on Windows.\n</|tool_result|>\n<|think|> Verified fact in local memory. State directly. </|think|>\n<|assistant|> Project Aurora uses 1/phi residual damping and runs locally on Windows.",
    "<|user|> What Fibonacci lags are configured in MultiHeadFibonacciMixer?\n<|tool_result|>\n[LOCAL MEMORY | harmonic_lm.py#part2 | score=0.79]: fib_lags=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55)\n</|tool_result|>\n<|think|> Extract exact tuple from retrieved code chunk. </|think|>\n<|assistant|> MultiHeadFibonacciMixer is configured with causal Fibonacci lags (0, 1, 2, 3, 5, 8, 13, 21, 34, 55).",
    "<|user|> Search the web for SQLite WAL mode concurrency\n<|tool_result|>\n[WEB | SQLite Write-Ahead Logging]: WAL mode allows concurrent readers while a single writer appends to a separate -wal file without blocking reads.\n</|tool_result|>\n<|think|> Synthesize extracted web fact concisely. </|think|>\n<|assistant|> SQLite Write-Ahead Logging (WAL) mode enables multiple concurrent readers to query the database without blocking while a single writer appends changes to the -wal log.",
    "<|user|> Solve for x: 3x + 7 = 22.\n<|think|> Subtract 7 from both sides: 3x = 15. Divide by 3: x = 5. </|think|>\n<|assistant|> x = 5.",
    "<|user|> Write a Python function to check if a string is a palindrome.\n<|think|> Normalize lowercase alphanumeric characters and compare against its reverse slice [::-1]. </|think|>\n<|assistant|> ```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```",
    "<|user|> Why does gradient checkpointing save GPU memory?\n<|think|> Trades extra forward compute during backward pass instead of storing all intermediate layer activations in VRAM. </|think|>\n<|assistant|> Gradient checkpointing avoids storing every intermediate activation tensor in VRAM during the forward pass, recomputing them on the fly during backpropagation to cut memory usage by ~65%.",
]


def fetch_open_instructions(max_samples: int = 5000) -> list[str]:
    """Streams real open-source instruction pairs and formats them with <|think|> tags."""
    formatted = []
    print(f"[SFT] Streaming up to {max_samples:,} open-source instruction pairs...")
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
        for row in ds:
            inst = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            out = row.get("output", "").strip()
            if not inst or not out or len(out) > 900:
                continue
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            sample = (
                f"<|user|> {user_msg}\n"
                f"<|think|> Analyze user request and provide an accurate, well-structured response. </|think|>\n"
                f"<|assistant|> {out}"
            )
            formatted.append(sample)
            if len(formatted) >= max_samples:
                break
    except Exception as e:
        print(f"[SFT Warning] Instruction stream fallback ({e}); using Sovereign templates.")
    print(f"[SFT] Collected {len(formatted):,} open-source instruction examples.")
    return formatted


def run_sft(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.save_dir, "harmonic_best.pt")

    if not os.path.exists(ckpt_path):
        print(f"[SFT Error] Checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = _build_harmonic_from_cfg(ckpt["config"]).to(device)
    model.load_state_dict((ckpt.get("model_state_dict") or ckpt["model_state"]))
    model.train()

    open_samples = fetch_open_instructions(max_samples=args.open_samples)
    # 50/50 blend: repeat Sovereign templates to match open-source instruction volume
    repeat_factor = max(1, len(open_samples) // max(1, len(SOVEREIGN_TEMPLATES)))
    combined = open_samples + (SOVEREIGN_TEMPLATES * repeat_factor)
    random.shuffle(combined)

    tokens = []
    for ex in combined:
        tokens.extend(ENC.encode_ordinary(ex))
        tokens.append(EOT)

    sft_data = torch.tensor(tokens, dtype=torch.long)
    print(f"[SFT] Total Alignment Corpus: {len(sft_data):,} tokens ({len(combined):,} conversations)")

    # Include 25% general pre-training replay if train.bin exists to prevent forgetting
    train_bin = os.path.join("data", "universal_genome", "train.bin")
    replay_data = np.memmap(train_bin, dtype=np.uint16, mode="r") if os.path.exists(train_bin) else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for step in range(1, args.steps + 1):
        ix = torch.randint(0, len(sft_data) - args.seq_len - 1, (args.batch_size,))
        xb = torch.stack([sft_data[i : i + args.seq_len] for i in ix]).to(device)
        yb = torch.stack([sft_data[i + 1 : i + args.seq_len + 1] for i in ix]).to(device)

        if replay_data is not None and args.batch_size >= 2:
            rix = torch.randint(0, len(replay_data) - args.seq_len - 1, (1,))
            rx = torch.from_numpy(replay_data[rix[0] : rix[0] + args.seq_len].astype(np.int64)).unsqueeze(0).to(device)
            ry = torch.from_numpy(replay_data[rix[0] + 1 : rix[0] + args.seq_len + 1].astype(np.int64)).unsqueeze(0).to(device)
            xb[0] = rx[0]
            yb[0] = ry[0]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            _, loss, avg_loops = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 40 == 0:
            print(f"SFT Step {step:3d}/{args.steps} | Alignment Loss: {loss.item():.4f} | Avg Loops: {avg_loops.item():.2f}")

    torch.save({"config": ckpt["config"], "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"\nSovereign + 5K Instruction Alignment Complete! Saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--open_samples", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    run_sft(parser.parse_args())
