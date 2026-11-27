# HarmonicLM (`harmonic_ai`)

**HarmonicLM** is a 242.3-million-parameter recursive language model engineered from scratch in PyTorch. Unlike fixed-depth transformers that spend identical compute on every token, HarmonicLM combines **16 recurrent blocks with adaptive pondering loops**, **Multi-Head Fibonacci sparse mixing**, and **Golden-Ratio (1/φ) scale-invariant residual damping** to dynamically allocate extra reasoning depth to complex sequences.

---

## Architecture Highlights

| Specification | Configuration |
| :--- | :--- |
| **Parameters** | `242,303,424` (~242.3M) |
| **Hidden Dimension (`dim`)** | `1280` |
| **Compressed Embedding (`embed_dim`)** | `192` |
| **Recurrent Blocks (`num_blocks`)** | `16` |
| **Attention / Mixing Heads (`num_heads`)** | `8` (Fibonacci causal lags: `0, 1, 2, 3, 5, 8, 13, 21, 34, 55`) |
| **Max Pondering Loops (`max_loops`)** | `4` (adaptive per-token halting) |
| **Context Window (`max_seq_len`)** | `512` |
| **Tokenizer** | `tiktoken` (`gpt2`, 50,257 vocab) |

---

## Repository Structure

```text
harmonic_ai/
├── brain/
│   └── harmonic_lm.py              # HarmonicLM v2 architecture & recursive blocks
├── data/
│   └── prepare_universal_genome.py # Streaming dataset tokenization -> train.bin / val.bin
├── training/
│   ├── train_harmonic.py           # Distributed (DDP) multi-GPU pre-training loop
│   ├── build_sovereign_sft.py      # Structured <|think|> SFT dataset & fine-tuning
│   └── train_grpo.py               # Group Relative Policy Optimization (GRPO) RL
├── cli/
│   └── hm.py                       # Cross-platform sovereign terminal agent (`hm`)
└── checkpoints/                    # Local model checkpoints (.pt, git-ignored)
```

---

## Training Pipeline

1. **Streaming Corpus Preparation**
   ```bash
   PYTHONPATH=. python -u data/prepare_universal_genome.py --skip_docs 36000 --docs_per_set 65000
   ```
2. **Multi-GPU Foundation Pre-Training (DDP + Gradient Checkpointing)**
   ```bash
   PYTHONPATH=. torchrun --nproc_per_node=2 training/train_harmonic.py \
     --dim 1280 --embed_dim 192 --blocks 16 --heads 8 \
     --batch_size 8 --grad_accum 4 --seq_len 512 \
     --steps 1500 --eval_interval 100 --lr 2.5e-4 --grad_ckpt
   ```
3. **Sovereign Supervised Fine-Tuning (SFT)**
   ```bash
   PYTHONPATH=. python -u training/build_sovereign_sft.py --steps 200 --lr 5e-5
   ```
4. **Verifiable GRPO Reinforcement Learning**
   ```bash
   PYTHONPATH=. python -u training/train_grpo.py --steps 50 --group_size 4 --lr 1e-5
   ```

---

## Sovereign Terminal CLI (`hm`)

`cli/hm.py` runs HarmonicLM locally offline across **Windows PowerShell**, **Linux (`bash`)**, and **macOS (`zsh`)** with live token streaming and human-in-the-loop (`[Y/n]`) command confirmation.

### Shell Setup
* **Windows PowerShell (`$PROFILE`):**
  ```powershell
  function hm { & "C:\Users\emanu\harmonic_ai\venv\Scripts\python.exe" "C:\Users\emanu\harmonic_ai\cli\hm.py" @args }
  ```
* **Linux (`~/.bashrc`):**
  ```bash
  hm() { python3 "\(HOME/harmonic_ai/cli/hm.py" "\)@"; }
  ```
* **macOS (`~/.zshrc`):**
  ```zsh
  hm() { python3 "\(HOME/harmonic_ai/cli/hm.py" "\)@"; }
  ```

### Usage
```bash
# Single-turn query or command request
hm "Show the top 5 processes using the most RAM"

# Interactive terminal loop
hm
