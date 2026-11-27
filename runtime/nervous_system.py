import re
import subprocess
import sys
import tiktoken
import torch
from brain.harmonic_lm import HarmonicLM
from runtime.search_and_memory import HybridMemoryStore, FastWebEngine


class FibonacciRollingMemory:
    """
    Active Conversation Memory:
    Keeps recent turns at full resolution and preserves milestone context
    at Fibonacci lookback offsets (1, 2, 3, 5, 8, 13) so long sessions never bloat.
    """
    def __init__(self, fib_offsets=(1, 2, 3, 5, 8, 13)):
        self.fib_offsets = fib_offsets
        self.history: list[str] = []

    def add_turn(self, role: str, text: str):
        self.history.append(f"<|{role}|> {text}")

    def build_context(self) -> str:
        n = len(self.history)
        if n <= 5:
            return "\n".join(self.history)

        selected_indices = set(range(max(0, n - 3), n))
        for offset in self.fib_offsets:
            idx = n - offset
            if 0 <= idx < n:
                selected_indices.add(idx)

        return "\n".join(self.history[i] for i in sorted(selected_indices))


class NervousSystem:
    """
    Tuned Field Agent Loop with Test-Time Scaling:
    1. OBSERVE: Assemble User Profile + Fibonacci Rolling Memory + Dense Semantic Routing
    2. THINK & EXECUTE: Best-of-N rollouts with live <|python|> sandbox execution
    3. VERIFY & ACT: Score candidates by execution validity & structure, return winner
    """
    def __init__(self, model: HarmonicLM, memory: HybridMemoryStore, web: FastWebEngine, device: torch.device):
        self.model = model
        self.memory = memory
        self.web = web
        self.device = device
        self.rolling_mem = FibonacciRollingMemory()
        self.enc = tiktoken.get_encoding("gpt2")

    def _run_python_sandbox(self, code: str, timeout: float = 3.0) -> tuple[str, bool]:
        """Executes model-written Python in an isolated subprocess with a strict timeout."""
        clean_code = code.strip()
        if not clean_code:
            return "[Empty code block]", False
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", clean_code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                out = proc.stdout.strip() or "[Executed successfully with no stdout]"
                return out[:600], True
            err = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            return "ERROR: " + " | ".join(err)[:600], False
        except subprocess.TimeoutExpired:
            return "ERROR: Execution timed out (3.0s limit)", False
        except Exception as exc:
            return f"ERROR: {exc}", False

    def _generate_with_sandbox(
        self,
        base_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: int = 25,
        max_exec_rounds: int = 2,
    ) -> tuple[str, dict]:
        """Generates a single rollout, pausing to execute any <|python|>...</|python|> blocks."""
        working_prompt = base_prompt
        stats = {"code_runs": 0, "code_successes": 0, "code_errors": 0}

        for _ in range(max_exec_rounds + 1):
            input_ids = self.enc.encode_ordinary(working_prompt)[-512:]
            idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            out_ids = self.model.generate(
                idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
            )
            chunk = self.enc.decode(out_ids[0, len(input_ids):].tolist())

            # Stop if model ended turn
            if "<|endoftext|>" in chunk or "<|user|>" in chunk:
                working_prompt += chunk.split("<|endoftext|>")[0].split("<|user|>")[0]
                break

            # Intercept closed <|python|>...</|python|> block
            py_match = re.search(r"<\|python\|>(.*?)</\|python\|>", chunk, flags=re.DOTALL)
            if py_match:
                prefix = chunk[: py_match.end()]
                code_body = py_match.group(1)
                result_text, ok = self._run_python_sandbox(code_body)
                stats["code_runs"] += 1
                if ok:
                    stats["code_successes"] += 1
                else:
                    stats["code_errors"] += 1

                working_prompt += f"{prefix}\n<|tool_result|>\n{result_text}\n</|tool_result|>\n"
                continue

            working_prompt += chunk
            break

        # Extract the newly generated portion after base_prompt
        generated_full = working_prompt[len(base_prompt):]
        return generated_full, stats

    def _score_candidate(self, generated_full: str, stats: dict) -> tuple[float, str]:
        """Scores a rollout for Best-of-N selection and extracts clean assistant reply."""
        score = 0.0

        # Reward verified Python execution; penalize runtime tracebacks
        score += 3.0 * stats["code_successes"]
        score -= 2.0 * stats["code_errors"]

        # Extract assistant message if <|assistant|> tag was emitted inside rollout
        if "<|assistant|>" in generated_full:
            think_part, reply = generated_full.rsplit("<|assistant|>", 1)
            if "</|think|>" in think_part:
                score += 1.5
        else:
            reply = generated_full

        reply = reply.split("<|endoftext|>")[0].split("<|user|>")[0].split("<|think|>")[0].strip()

        # Penalize empty/collapsed outputs; reward coherent length
        word_count = len(reply.split())
        if word_count == 0:
            score -= 5.0
        elif word_count < 3:
            score -= 1.0
        else:
            score += min(word_count / 20.0, 2.0)

        return score, reply

    async def _smart_retrieve(self, query: str) -> str:
        """Routes intelligently: Identity -> Reflex -> Dense Local DB -> Live Web."""
        q_lower = query.lower()
        force_web = any(k in q_lower for k in ("search", "web", "latest", "news", "today", "online", "look up"))

        is_identity = any(k in q_lower for k in ("your name", "who are you", "what are you", "your architecture"))
        if is_identity:
            print("[NervousSystem] Route: Sovereign Identity (0ms)")
            return ""

        is_chitchat = len(query.split()) <= 6 and not force_web and "?" not in query
        if is_chitchat:
            print("[NervousSystem] Route: Internal Reflex (0ms)")
            return ""

        local_hits = self.memory.query(query, top_k=2)
        strong_local = [h for h in local_hits if h["score"] >= 0.42]

        blocks = ["<|tool_result|>"]
        for hit in strong_local:
            blocks.append(f"[LOCAL MEMORY | {hit['source']} | score={hit['score']:.2f}]: {hit['content']}")

        if strong_local and not force_web:
            print(f"[NervousSystem] Route: Dense Local Hit (score={strong_local[0]['score']:.2f}, skipped web)")
            blocks.append("</|tool_result|>")
            return "\n".join(blocks)

        print("[NervousSystem] Route: Local Miss -> Dispatching Live Web Search...")
        web_hits = await self.web.search_and_rerank(query, num_results=3)
        for hit in web_hits:
            blocks.append(f"[WEB | {hit['title']} ({hit['url']})]: {hit['full_extract']}")

        blocks.append("</|tool_result|>")
        return "\n".join(blocks) if len(blocks) > 2 else ""

    async def step(self, user_prompt: str, max_new_tokens: int = 80, num_candidates: int = 3) -> str:
        # 1. OBSERVE
        self.rolling_mem.add_turn("user", user_prompt)
        retrieved_block = await self._smart_retrieve(user_prompt)
        active_context = self.rolling_mem.build_context()

        prompt_parts = []
        profile_block = self.memory.get_profile_block()
        if profile_block:
            prompt_parts.append(profile_block)
        prompt_parts.append(active_context)
        if retrieved_block:
            prompt_parts.append(retrieved_block)

        # Allow the model to emit <|python|> inside <|think|> before <|assistant|>
        prompt_parts.append("<|think|>")
        full_prompt = "\n".join(prompt_parts)

        # Measure recursive loop depth once on the base prompt
        base_ids = self.enc.encode_ordinary(full_prompt)[-512:]
        base_idx = torch.tensor([base_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, _, avg_loops = self.model(base_idx)

        # Use N=1 for trivial chitchat, Best-of-N for reasoning/coding queries
        is_trivial = len(user_prompt.split()) <= 6 and "?" not in user_prompt
        n_rollouts = 1 if is_trivial else max(1, num_candidates)

        # 2. THINK & VERIFY (Best-of-N + Live Python Sandbox)
        best_score = float("-inf")
        best_reply = ""
        best_stats = {"code_runs": 0, "code_successes": 0, "code_errors": 0}
        scores_log = []

        for i in range(n_rollouts):
            temp = 0.4 if i == 0 else 0.75  # Candidate 0 is low-temp anchor; others explore
            gen_full, stats = self._generate_with_sandbox(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=25,
            )
            score, reply = self._score_candidate(gen_full, stats)
            scores_log.append(round(score, 2))

            if score > best_score:
                best_score = score
                best_reply = reply
                best_stats = stats

        if not best_reply:
            best_reply = "[Model halted early — needs more training steps]"

        print(
            f"[NervousSystem] Recursive Loops: {avg_loops.item():.2f}/{self.model.blocks[0].max_loops} | "
            f"Best-of-{n_rollouts} Scores: {scores_log} | "
            f"Python Execs: {best_stats['code_successes']}/{best_stats['code_runs']}"
        )

        # 3. Record into Fibonacci Rolling Memory
        self.rolling_mem.add_turn("assistant", best_reply)
        return best_reply

