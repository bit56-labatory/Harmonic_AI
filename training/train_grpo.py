import argparse
import copy
import os
import random
import re
import subprocess
import sys
import tempfile
import tiktoken
import torch
import torch.nn.functional as F
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


def load_rl_tasks(max_per_domain: int = 250) -> list[dict]:
    """Loads verifiable GSM8K math problems and MBPP Python unit-test tasks."""
    tasks = []
    print("[GRPO] Loading verifiable GSM8K math & MBPP coding tasks...")

    try:
        gsm = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        count = 0
        for row in gsm:
            q = row.get("question", "").strip()
            ans_raw = row.get("answer", "")
            if "####" in ans_raw:
                target_num = ans_raw.split("####")[-1].strip().replace(",", "")
                tasks.append({
                    "type": "math",
                    "prompt": f"<|user|> {q}\n<|think|>",
                    "target": target_num,
                })
                count += 1
                if count >= max_per_domain:
                    break
        print(f"  -> Loaded {count} GSM8K math tasks.")
    except Exception as e:
        print(f"  -> [Warning] GSM8K stream skipped ({e})")

    try:
        mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train", streaming=True)
        count = 0
        for row in mbpp:
            prompt_txt = row.get("prompt", "").strip()
            tests = row.get("test_list", [])
            if prompt_txt and tests:
                tasks.append({
                    "type": "code",
                    "prompt": f"<|user|> {prompt_txt}\n<|think|>",
                    "tests": tests,
                })
                count += 1
                if count >= max_per_domain:
                    break
        print(f"  -> Loaded {count} MBPP Python unit-test tasks.")
    except Exception as e:
        print(f"  -> [Warning] MBPP stream skipped ({e})")

    # Fallback built-in tasks if offline
    if not tasks:
        tasks = [
            {"type": "math", "prompt": "<|user|> Solve for x: 3x + 7 = 22.\n<|think|>", "target": "5"},
            {"type": "math", "prompt": "<|user|> What is 18 * 14?\n<|think|>", "target": "252"},
            {
                "type": "code",
                "prompt": "<|user|> Write a Python function `add_nums(a, b)` that returns the sum of a and b.\n<|think|>",
                "tests": ["assert add_nums(2, 3) == 5", "assert add_nums(-1, 4) == 3"],
            },
        ]


    cli_rl_tasks = [
        {"type": "cli", "os": "powershell", "prompt": "<|user|> [Windows | PowerShell] Show top 5 processes by RAM usage.\n<|think|>", "expected_tokens": ["Get-Process", "Sort-Object"]},
        {"type": "cli", "os": "powershell", "prompt": "<|user|> [Windows | PowerShell] Find the 10 largest files recursively.\n<|think|>", "expected_tokens": ["Get-ChildItem", "Sort-Object"]},
        {"type": "cli", "os": "powershell", "prompt": "<|user|> [Windows | PowerShell] Search all .py files for torchrun.\n<|think|>", "expected_tokens": ["Get-ChildItem", "Select-String"]},
        {"type": "cli", "os": "powershell", "prompt": "<|user|> [Windows | PowerShell] List all listening TCP ports.\n<|think|>", "expected_tokens": ["Get-NetTCPConnection"]},
        {"type": "cli", "os": "bash", "prompt": "<|user|> [Linux | bash] Show top 5 processes by memory.\n<|think|>", "expected_tokens": ["ps", "head"]},
        {"type": "cli", "os": "bash", "prompt": "<|user|> [Linux | bash] Find the 10 largest files recursively.\n<|think|>", "expected_tokens": ["find", "sort"]},
        {"type": "cli", "os": "zsh", "prompt": "<|user|> [Darwin | zsh] List all listening TCP ports on macOS.\n<|think|>", "expected_tokens": ["lsof"]},
    ]
    tasks.extend(cli_rl_tasks * 15)

    random.shuffle(tasks)
    return tasks


def verify_math(completion: str, target: str) -> float:
    """Checks format compliance (+0.2) and exact final numerical answer (+1.0)."""
    reward = 0.0
    if "</|think|>" in completion and "<|assistant|>" in completion:
        reward += 0.2

    after_tag = completion.split("<|assistant|>")[-1] if "<|assistant|>" in completion else completion
    nums = re.findall(r"-?\d+(?:\.\d+)?", after_tag.replace(",", ""))
    if nums:
        try:
            if abs(float(nums[-1]) - float(target)) < 1e-4:
                reward += 1.0
        except ValueError:
            pass
    return reward


def verify_python(completion: str, tests: list[str]) -> float:
    """Executes generated Python code against real assert unit tests in a 2s sandbox."""
    reward = 0.0
    if "</|think|>" in completion and "<|assistant|>" in completion:
        reward += 0.2

    after_tag = completion.split("<|assistant|>")[-1] if "<|assistant|>" in completion else completion
    code_match = re.search(r"```(?:python)?\s*(.*?)```", after_tag, re.DOTALL)
    code_body = code_match.group(1).strip() if code_match else after_tag.strip()

    if "def " not in code_body:
        return reward

    test_script = code_body + "\n\n" + "\n".join(tests) + "\n"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(test_script)
            tmp_path = f.name
        res = subprocess.run([sys.executable, tmp_path], capture_output=True, timeout=2.0)
        if res.returncode == 0:
            reward += 1.0
    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return reward




def verify_cli(completion: str, shell_os: str, expected_tokens: list[str]) -> float:
    """Verifies fenced shell code block, expected native cmdlets, balanced delimiters, and safety."""
    reward = 0.0
    if "</|think|>" in completion and "<|assistant|>" in completion:
        reward += 0.2

    after_tag = completion.split("<|assistant|>")[-1] if "<|assistant|>" in completion else completion
    match = re.search(r"```(?:powershell|pwsh|bash|zsh|sh)?\s*\n(.*?)```", after_tag, re.DOTALL | re.IGNORECASE)
    if not match:
        return reward

    cmd = match.group(1).strip()
    if not cmd:
        return reward

    banned = ["rm -rf /", "format-volume", "mkfs", "dd if=", ":(){", "del /f /s /q c:\\"]
    if any(b in cmd.lower() for b in banned):
        return 0.0

    if cmd.count("(") != cmd.count(")") or cmd.count("{") != cmd.count("}"):
        return reward
    if cmd.count("'") % 2 != 0 or cmd.count('"') % 2 != 0:
        return reward

    if any(tok.lower() in cmd.lower() for tok in expected_tokens):
        reward += 1.0

    return reward


def sequence_logprobs(model: HarmonicLM, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Computes per-token log probabilities strictly over the generated completion tokens."""
    logits, _, _ = model(full_ids[:, :-1])
    log_probs = F.log_softmax(logits, dim=-1)
    target_ids = full_ids[:, 1:]
    token_lp = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp[:, prompt_len - 1 :]


def run_grpo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.save_dir, "harmonic_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[GRPO Error] Checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = _build_harmonic_from_cfg(ckpt["config"]).to(device)
    model.load_state_dict((ckpt.get("model_state_dict") or ckpt["model_state"]))

    # Frozen SFT reference policy to prevent KL drift / language collapse
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tasks = load_rl_tasks(max_per_domain=args.max_tasks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\nStarting Verifiable GRPO RL | Group Size: {args.group_size} | Steps: {args.steps} | Tasks: {len(tasks)}\n")

    for step in range(1, args.steps + 1):
        task = random.choice(tasks)
        prompt_ids = ENC.encode_ordinary(task["prompt"])
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = len(prompt_ids)

        # 1. Sample G candidate completions
        model.eval()
        rollouts = []
        rewards = []
        for _ in range(args.group_size):
            out_ids = model.generate(
                prompt_tensor.clone(),
                max_new_tokens=args.max_new_tokens,
                temperature=0.85,
                top_k=40,
            )
            gen_tokens = out_ids[0, prompt_len:].tolist()
            if EOT in gen_tokens:
                cut = gen_tokens.index(EOT)
                gen_tokens = gen_tokens[:cut]
                out_ids = out_ids[:, : prompt_len + max(1, cut)]

            completion_text = ENC.decode(gen_tokens)
            if task["type"] == "math":
                r = verify_math(completion_text, task["target"])
            else:
                r = verify_python(completion_text, task["tests"])

            rollouts.append(out_ids)
            rewards.append(r)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()

        # Skip gradient update if all rollouts tied (zero variance -> zero advantage signal)
        if std_r < 1e-5:
            if step == 1 or step % 10 == 0:
                print(f"GRPO Step {step:3d}/{args.steps} | Avg Reward: {mean_r.item():.2f} (uniform group, skipped update)")
            continue

        advantages = (rewards_t - mean_r) / (std_r + 1e-6)

        # 2. Policy gradient update with group-relative advantage + KL penalty
        model.train()
        optimizer.zero_grad()
        total_pg_loss = 0.0

        for idx_g, seq_ids in enumerate(rollouts):
            if seq_ids.size(1) <= prompt_len + 1:
                continue
            policy_lp = sequence_logprobs(model, seq_ids, prompt_len)
            with torch.no_grad():
                ref_lp = sequence_logprobs(ref_model, seq_ids, prompt_len)

            kl_div = (torch.exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1.0).mean()
            pg_loss = -(advantages[idx_g] * policy_lp.mean()) + args.kl_beta * kl_div
            (pg_loss / args.group_size).backward()
            total_pg_loss += pg_loss.item() / args.group_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(
                f"GRPO Step {step:3d}/{args.steps} | "
                f"Avg Reward: {mean_r.item():.2f} (Max: {rewards_t.max().item():.2f}) | "
                f"PG Loss: {total_pg_loss:.4f}"
            )

    torch.save({"config": ckpt["config"], "model_state_dict": model.state_dict()}, ckpt_path)
    print(f"\nVerifiable GRPO RL Complete! Aligned checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--max_tasks", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_beta", type=float, default=0.04)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    run_grpo(parser.parse_args())
