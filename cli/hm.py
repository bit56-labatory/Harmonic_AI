import argparse
import inspect
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.harmonic_lm import HarmonicLM


def detect_shell_env() -> dict:
    sys_name = platform.system()
    if sys_name == "Windows":
        shell_name = "PowerShell" if os.environ.get("PSModulePath") else "CMD"
    elif sys_name == "Darwin":
        shell_name = Path(os.environ.get("SHELL", "/bin/zsh")).name
    else:
        shell_name = Path(os.environ.get("SHELL", "/bin/bash")).name

    return {
        "os": sys_name,
        "release": platform.release(),
        "arch": platform.machine(),
        "shell": shell_name,
        "cwd": os.getcwd(),
    }


def extract_command(reply: str):
    match = re.search(r"```(?:powershell|pwsh|bash|zsh|sh|cmd)?\s*\n(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    for line in reply.splitlines():
        clean = line.strip()
        if clean.startswith("$ ") or clean.startswith("PS> "):
            return clean.split(" ", 1)[-1].strip()
    return None


def resolve_checkpoint(explicit_ckpt: str | None) -> Path:
    if explicit_ckpt:
        return Path(explicit_ckpt)
    candidates = [
        ROOT / "checkpoints" / "harmonic_foundation_242m.pt",
        ROOT / "checkpoints" / "harmonic_best.pt",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        print("[hm] No checkpoint found in checkpoints/. Drop harmonic_foundation_242m.pt or harmonic_best.pt there first.")
        sys.exit(1)
    # Pick the largest checkpoint (242M ~925MB over tiny smoke-test checkpoints)
    return max(existing, key=lambda p: p.stat().st_size)


def load_engine(ckpt_path: Path, device: torch.device):
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    raw_cfg = {
        "dim": cfg.get("dim", 1280),
        "embed_dim": cfg.get("embed_dim", 192),
        "num_blocks": cfg.get("num_blocks", cfg.get("n_blocks", 16)),
        "num_heads": cfg.get("num_heads", cfg.get("n_heads", 8)),
        "max_seq_len": cfg.get("max_seq_len", 512),
        "max_loops": cfg.get("max_loops", 4),
    }
    valid_params = inspect.signature(HarmonicLM.__init__).parameters
    model = HarmonicLM(**{k: v for k, v in raw_cfg.items() if k in valid_params}).to(device)
    state = ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt
    clean_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[hm] Loaded {ckpt_path.name} ({params_m:.1f}M params) on {device.type} in {time.time() - t0:.1f}s", flush=True)
    return model


@torch.no_grad()
def generate_reply(
    model,
    tok,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 60,
    temperature: float = 0.75,
    top_p: float = 0.9,
    rep_penalty: float = 1.18,
) -> str:
    ids = tok.encode_ordinary(prompt)
    x = torch.tensor([ids[-128:]], dtype=torch.long, device=device)
    generated = []
    printed_len = 0
    for _ in range(max_new_tokens):
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        next_logits = logits[0, -1].float()

        # Repetition penalty on recent tokens
        seen = set(x[0, -64:].tolist())
        for tok_id in seen:
            if next_logits[tok_id] > 0:
                next_logits[tok_id] /= rep_penalty
            else:
                next_logits[tok_id] *= rep_penalty

        next_logits = next_logits / max(temperature, 1e-5)
        probs = F.softmax(next_logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        next_id = int(sorted_idx[torch.multinomial(sorted_probs, 1)].item())

        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)[:, -256:]
        decoded = tok.decode(generated)
        if "<|endoftext|>" in decoded or "\nUser:" in decoded:
            break
        new_text = decoded[printed_len:]
        if new_text:
            print(new_text, end="", flush=True)
            printed_len = len(decoded)
    print("", flush=True)
    return tok.decode(generated).replace("<|endoftext|>", "").split("\nUser:")[0].strip()


def execute_shell_command(cmd: str, env_info: dict) -> int:
    if env_info["os"] == "Windows" and env_info["shell"] == "PowerShell":
        proc = subprocess.run(["powershell", "-NoProfile", "-Command", cmd])
    else:
        proc = subprocess.run(cmd, shell=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(prog="hm", description="HarmonicLM Cross-Platform Terminal Agent")
    parser.add_argument("query", nargs="*", help="Instruction or question for HarmonicLM")
    parser.add_argument("--ckpt", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--tokens", type=int, default=60, help="Max tokens to generate")
    parser.add_argument("--chat", action="store_true", help="Use User/Assistant chat format (for SFT/GRPO checkpoints)")
    parser.add_argument("--run", "-y", action="store_true", help="Execute extracted command without confirmation")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.ckpt)
    env_info = detect_shell_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_engine(ckpt_path, device)
    tok = tiktoken.get_encoding("gpt2")

    is_sft = args.chat or ("sft" in ckpt_path.name.lower()) or ("best" in ckpt_path.name.lower() and ckpt_path.stat().st_size > 400_000_000) or ("best" in ckpt_path.name.lower() and ckpt_path.stat().st_size > 400_000_000)

    def handle_turn(user_text: str):
        if is_sft:
            prompt = (
                f"<|user|> [{env_info['os']} | {env_info['shell']}] {user_text}\n<|think|>"
            )
        else:
            prompt = user_text

        print(f"\n[hm | {env_info['shell']}] {'' if is_sft else prompt}", end="", flush=True)
        reply = generate_reply(model, tok, prompt, device, max_new_tokens=args.tokens)
        print()

        cmd = extract_command(reply)
        if cmd:
            if args.run:
                execute_shell_command(cmd, env_info)
            else:
                ans = input(f"Execute `{cmd}`? [Y/n]: ").strip().lower()
                if ans in ("", "y", "yes"):
                    code = execute_shell_command(cmd, env_info)
                    print(f"[hm] Exit code: {code}")

    if args.query:
        handle_turn(" ".join(args.query))
    else:
        print(f"[hm] Interactive mode ({env_info['os']} / {env_info['shell']}) — type 'exit' to quit.")
        while True:
            try:
                q = input("hm> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            handle_turn(q)


if __name__ == "__main__":
    main()



