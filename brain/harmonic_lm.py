import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PHI = (1.0 + math.sqrt(5.0)) / 2.0
INV_PHI = 1.0 / PHI


class ScaleInvariantNorm(nn.Module):
    """Parameter-free scale-invariant RMS normalization without additive bias."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class PhiRoPE(nn.Module):
    """Golden-Ratio Rotary Positional Embeddings with offset support for KV-cached decoding."""
    def __init__(self, head_dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (PHI ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_offset: int = 0):
        seq_len = q.size(2)
        total_len = pos_offset + seq_len
        if total_len > self.cos_cached.size(2):
            self._build_cache(total_len + 256)
        cos = self.cos_cached[:, :, pos_offset:total_len, :]
        sin = self.sin_cached[:, :, pos_offset:total_len, :]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)


class MultiHeadFibonacciMixer(nn.Module):
    """
    O(T log T) Causal Fibonacci-Sparse Attention with Cross-Loop v_1 Anchor & O(1) Sliding KV-Cache:
    - Training / Prefill: vectorizes all Fibonacci lags (0, 1, 2, 3, 5, 8, 13, 21, 34, 55) in parallel
    - Cached Single-Token Decoding: stores only the last 55 K/V steps for constant-time generation
    """
    def __init__(self, dim: int, num_heads: int = 8, fib_lags=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55)):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.fib_lags = fib_lags
        self.max_lag = max(fib_lags)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = PhiRoPE(self.head_dim)

    def forward(self, x: torch.Tensor, v1_anchor: torch.Tensor = None, kv_cache: dict = None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if v1_anchor is None:
            v1_anchor = v
        else:
            v = (1.0 - INV_PHI * 0.5) * v + (INV_PHI * 0.5) * v1_anchor

        pos_offset = kv_cache["pos"] if (kv_cache is not None and "pos" in kv_cache) else 0
        q, k = self.rope(q, k, pos_offset=pos_offset)

        # Fast O(1) single-token decode path using sliding 56-token Fibonacci window
        if kv_cache is not None and T == 1 and "k" in kv_cache:
            k_hist = torch.cat([kv_cache["k"], k], dim=2)[:, :, -(self.max_lag + 1):, :]
            v_hist = torch.cat([kv_cache["v"], v], dim=2)[:, :, -(self.max_lag + 1):, :]
            kv_cache["k"] = k_hist
            kv_cache["v"] = v_hist
            kv_cache["pos"] = pos_offset + 1

            hist_len = k_hist.size(2)
            k_lags, v_lags, valid_masks = [], [], []
            for lag in self.fib_lags:
                idx = hist_len - 1 - lag
                if idx >= 0:
                    k_lags.append(k_hist[:, :, idx : idx + 1, :])
                    v_lags.append(v_hist[:, :, idx : idx + 1, :])
                    valid_masks.append(torch.ones((1, 1, 1, 1), device=x.device, dtype=torch.bool))
                else:
                    k_lags.append(torch.zeros_like(k))
                    v_lags.append(torch.zeros_like(v))
                    valid_masks.append(torch.zeros((1, 1, 1, 1), device=x.device, dtype=torch.bool))

            k_stack = torch.stack(k_lags, dim=3)
            v_stack = torch.stack(v_lags, dim=3)
            mask_stack = torch.cat(valid_masks, dim=-1)

            scores = torch.einsum("bhtd,bhtld->bhtl", q, k_stack) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask_stack, -float("inf"))
            attn = F.softmax(scores, dim=-1)

            out = torch.einsum("bhtl,bhtld->bhtd", attn, v_stack)
            out = out.transpose(1, 2).contiguous().view(B, 1, D)
            return self.out_proj(out), v1_anchor

        # Prompt prefill / training path
        if kv_cache is not None:
            kv_cache["k"] = k[:, :, -(self.max_lag + 1):, :]
            kv_cache["v"] = v[:, :, -(self.max_lag + 1):, :]
            kv_cache["pos"] = pos_offset + T

        lags = torch.tensor(self.fib_lags, device=x.device)
        pos = torch.arange(T, device=x.device).unsqueeze(1) - lags.unsqueeze(0)
        valid = pos >= 0
        safe_pos = pos.clamp(min=0)
        k_stack = k[:, :, safe_pos, :]
        v_stack = v[:, :, safe_pos, :]
        mask_stack = valid.view(1, 1, T, 1, len(self.fib_lags))

        scores = torch.einsum("bhtd,bhtld->bhtl", q, k_stack) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask_stack.squeeze(3), -float("inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.einsum("bhtl,bhtld->bhtd", attn, v_stack)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), v1_anchor


class PhiRatioProcessor(nn.Module):
    """Golden-Ratio SwiGLU feed-forward processor (hidden_dim = int(dim * phi))."""
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = int(dim * PHI)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HarmonicRecurrentBlock(nn.Module):
    """Scale-Invariant Adaptive Recursive Block with per-loop KV-cache support."""
    def __init__(self, dim: int, num_heads: int = 8, max_loops: int = 4):
        super().__init__()
        self.max_loops = max_loops
        self.norm1 = ScaleInvariantNorm(dim)
        self.mixer = MultiHeadFibonacciMixer(dim, num_heads=num_heads)
        self.norm2 = ScaleInvariantNorm(dim)
        self.processor = PhiRatioProcessor(dim)
        self.halt_router = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, block_cache: list = None):
        B, T, D = x.shape
        cumulative_halt = torch.zeros((B, T, 1), device=x.device, dtype=x.dtype)
        remainders = torch.ones((B, T, 1), device=x.device, dtype=x.dtype)
        weighted_state = torch.zeros_like(x)
        ponder_cost = torch.zeros((B, T, 1), device=x.device, dtype=x.dtype)
        expected_loops = torch.zeros((B, T, 1), device=x.device, dtype=x.dtype)

        state = x
        v1_anchor = None

        for step in range(1, self.max_loops + 1):
            step_cache = block_cache[step - 1] if block_cache is not None else None
            mixed, v1_anchor = self.mixer(self.norm1(state), v1_anchor=v1_anchor, kv_cache=step_cache)
            state = state + INV_PHI * mixed
            state = state + INV_PHI * self.processor(self.norm2(state))

            if step < self.max_loops:
                p_halt = torch.sigmoid(self.halt_router(state))
            else:
                p_halt = torch.ones_like(cumulative_halt)

            still_running = (cumulative_halt < 0.99).to(x.dtype)
            effective_halt = torch.minimum(p_halt, remainders) * still_running

            weighted_state = weighted_state + effective_halt * state
            cumulative_halt = cumulative_halt + effective_halt
            remainders = remainders - effective_halt
            ponder_cost = ponder_cost + remainders
            expected_loops = expected_loops + still_running

        return weighted_state, ponder_cost.mean(), expected_loops.mean()


class HarmonicLM(nn.Module):
    """Complete Scale-Invariant Recursive Language Model (v2) with O(1) KV-Cached Generation."""
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 1280,
        embed_dim: int = 192,
        num_blocks: int = 16,
        num_heads: int = 8,
        max_loops: int = 4,
        ponder_weight: float = 0.01,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed_dim = embed_dim
        self.max_loops = max_loops
        self.ponder_weight = ponder_weight
        self.gradient_checkpointing = False

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=1.0 / math.sqrt(embed_dim))
        self.emb_up = nn.Linear(embed_dim, dim, bias=False)
        self.emb_down = nn.Linear(dim, embed_dim, bias=False)

        self.blocks = nn.ModuleList([
            HarmonicRecurrentBlock(dim=dim, num_heads=num_heads, max_loops=max_loops)
            for _ in range(num_blocks)
        ])
        self.final_norm = ScaleInvariantNorm(dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None, kv_caches: list = None):
        x = self.emb_up(self.token_emb(input_ids))
        total_ponder = 0.0
        total_loops = 0.0

        for i, block in enumerate(self.blocks):
            b_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and b_cache is None:
                x, ponder_cost, avg_loops = checkpoint(block, x, use_reentrant=False)
            else:
                x, ponder_cost, avg_loops = block(x, block_cache=b_cache)
            total_ponder = total_ponder + ponder_cost
            total_loops = total_loops + avg_loops

        x = self.final_norm(x)
        logits = self.lm_head(self.emb_down(x))
        avg_loops_all = total_loops / len(self.blocks)

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            loss = ce_loss + self.ponder_weight * (total_ponder / len(self.blocks))

        return logits, loss, avg_loops_all

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 60, temperature: float = 0.7, top_k: int = 25):
        self.eval()
        kv_caches = [[{} for _ in range(self.max_loops)] for _ in range(len(self.blocks))]

        # 1. Prefill full prompt once
        logits, _, _ = self(idx, kv_caches=kv_caches)

        # 2. Decode subsequent tokens one at a time in O(1) constant time
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            logits, _, _ = self(next_id, kv_caches=kv_caches)

        return idx