import argparse
import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from brain.harmonic_lm import HarmonicConfig, HarmonicLM


def get_batch(data_path: str, batch_size: int, seq_len: int, device: torch.device):
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + seq_len]).astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_val(raw_model, val_path: str, batch_size: int, seq_len: int, device: torch.device, eval_batches: int = 8):
    raw_model.eval()
    losses = []
    for _ in range(eval_batches):
        xb, yb = get_batch(val_path, batch_size, seq_len, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            _, loss, _ = raw_model(xb, yb)
        losses.append(loss.item())
    raw_model.train()
    return sum(losses) / len(losses)


def train(args):
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_master = rank == 0
    seed = args.seed if args.seed >= 0 else int(time.time() * 1000) % (2**31)
    torch.manual_seed(seed + rank * 9973)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank * 9973)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")

    cfg = HarmonicConfig(
        dim=args.dim,
        embed_dim=args.embed_dim,
        n_blocks=args.blocks,
        n_heads=args.heads,
        max_seq_len=args.seq_len,
        max_loops=args.max_loops,
        gradient_checkpointing=args.grad_ckpt,
    )

    raw_model = HarmonicLM(cfg).to(device)
    start_step = 1
    best_val_loss = float("inf")

    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        if is_master:
            print(f"[Init] Loaded foundation weights from {args.init_ckpt}")
    elif args.resume and os.path.exists(os.path.join(args.ckpt_dir, "harmonic_latest.pt")):
        ckpt = torch.load(os.path.join(args.ckpt_dir, "harmonic_latest.pt"), map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if is_master:
            print(f"[Resume] Resumed from step {start_step - 1} (Best Val Loss: {best_val_loss:.4f})")

    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False) if is_ddp else raw_model

    if HAS_BNB and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.1)
        if is_master:
            print("[Optimizer] Using 8-bit AdamW (bitsandbytes)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len * world_size

    if is_master:
        print(f"Training HarmonicLM v2 on {world_size}x {device.type} | Params: {raw_model.count_parameters():,} | Tokens/Step: {tokens_per_step:,}")

    t0 = time.time()
    for step in range(start_step, args.steps + 1):
        rel_step = step - start_step + 1
        total_rel = max(1, args.steps - start_step + 1)
        if args.warmup_steps > 0 and rel_step <= args.warmup_steps:
            cur_lr = args.lr * (0.1 + 0.9 * (rel_step / args.warmup_steps))
        elif args.decay_steps > 0 and rel_step > (total_rel - args.decay_steps):
            rem = max(0, total_rel - rel_step)
            cur_lr = args.lr * (0.05 + 0.95 * (rem / args.decay_steps))
        else:
            cur_lr = args.lr
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_loops = 0.0

        for micro in range(args.grad_accum):
            if is_ddp and micro < args.grad_accum - 1:
                with model.no_sync():
                    xb, yb = get_batch(train_path, args.batch_size, args.seq_len, device)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                        _, loss, avg_loops = model(xb, yb)
                        scaled_loss = loss / args.grad_accum
                    scaler.scale(scaled_loss).backward()
            else:
                xb, yb = get_batch(train_path, args.batch_size, args.seq_len, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    _, loss, avg_loops = model(xb, yb)
                    scaled_loss = loss / args.grad_accum
                scaler.scale(scaled_loss).backward()

            step_loss += loss.item() / args.grad_accum
            step_loops += float(avg_loops) / args.grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if is_master and (step == start_step or step % args.eval_interval == 0 or step == args.steps):
            val_loss = evaluate_val(raw_model, val_path, args.batch_size, args.seq_len, device)
            elapsed = time.time() - t0
            total_toks = step * tokens_per_step / 1e6
            print(
                f"Step {step:4d}/{args.steps} | Tokens: {total_toks:5.1f}M | "
                f"Train Loss: {step_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Avg Loops: {step_loops:.2f}/4 | Time: {elapsed:.1f}s"
            )
            state_dict = {
                "step": step,
                "model_state": raw_model.state_dict(),
                "config": cfg.__dict__,
                "val_loss": val_loss,
            }
            torch.save(state_dict, os.path.join(args.ckpt_dir, "harmonic_latest.pt"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state_dict, os.path.join(args.ckpt_dir, "harmonic_best.pt"))

    if is_master:
        print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.4f} -> Saved to {args.ckpt_dir}/harmonic_best.pt")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--max_loops", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init_ckpt", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data/universal_genome")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--decay_steps", type=int, default=0)
    train(parser.parse_args())
