import asyncio
import os
import torch
from brain.harmonic_lm import HarmonicLM
import inspect
def _load_harmonic_from_ckpt(ckpt):
    cfg = dict(ckpt["config"])
    if "n_blocks" in cfg and "num_blocks" not in cfg:
        cfg["num_blocks"] = cfg.pop("n_blocks")
    if "n_heads" in cfg and "num_heads" not in cfg:
        cfg["num_heads"] = cfg.pop("n_heads")
    valid = inspect.signature(HarmonicLM.__init__).parameters
    return HarmonicLM(**{k: v for k, v in cfg.items() if k in valid})

from runtime.search_and_memory import HybridMemoryStore, FastWebEngine
from runtime.nervous_system import NervousSystem


async def run_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join("checkpoints", "harmonic_best.pt")

    if not os.path.exists(ckpt_path):
        print("Checkpoint not found! Run 'python training\\train_harmonic.py' first.")
        return

    # 1. Load your trained HarmonicLM weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = _load_harmonic_from_ckpt(ckpt).to(device)
    model.load_state_dict((ckpt.get("model_state_dict") or ckpt["model_state"]))
    model.eval()

    # 2. Initialize Local Dense Database Memory, Web Engine, and Nervous System Loop
    memory = HybridMemoryStore("harmonic_memory.db")
    web = FastWebEngine()
    agent = NervousSystem(model, memory, web, device)

    print("\n========================================================")
    print("  HARMONIC-LM SOVEREIGN AGENT ONLINE")
    print("  Commands:")
    print("    /learn <fact or note>       -> Store permanently in local DB")
    print("    /ingest <file_path>         -> Chunk & index any local file into DB")
    print("    /profile <key> = <value>    -> Save persistent user preference/role")
    print("    exit                        -> Shut down cleanly")
    print("========================================================\n")

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[System] Shutting down cleanly.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            if user_input.startswith("/learn "):
                fact = user_input[7:].strip()
                memory.add_document(fact, source="user_taught")
                print(f"[Memory] Stored outside weights: '{fact}'\n")
                continue

            if user_input.startswith("/ingest "):
                filepath = user_input[8:].strip().strip('"')
                try:
                    chunks = memory.ingest_file(filepath)
                    print(f"[Memory] Ingested '{filepath}' into {chunks} searchable semantic chunks.\n")
                except Exception as e:
                    print(f"[Memory Error] {e}\n")
                continue

            if user_input.startswith("/profile ") and "=" in user_input:
                key, val = user_input[9:].split("=", 1)
                memory.set_profile(key, val)
                print(f"[Profile] Saved '{key.strip()}' -> '{val.strip()}'\n")
                continue

            reply = await agent.step(user_input)
            print(f"\nHarmonicLM > {reply}\n")
    finally:
        memory.conn.close()


if __name__ == "__main__":
    asyncio.run(run_agent())
