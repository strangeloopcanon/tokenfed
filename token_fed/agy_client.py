"""Async CLI LLM execution harness (`agy`).

All LLM turns (AI TokenFed governor policy decisions, PTU sublease clearing,
agent-firm portfolio/bidding planning, code drafting, and verifier-guided repair)
run strictly through the local `agy` CLI with tools denied.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any

AGY_BIN = os.environ.get("AGY_BIN") or shutil.which("agy") or str(Path.home() / ".local" / "bin" / "agy")

MODEL_FAMILIES: dict[str, dict[str, str]] = {
    "flash": {
        "label": "Gemini 3.8 Flash",
        "lean": "gemini-3.8-flash-low",
        "planner": "gemini-3.8-flash-low",
        "repair": "gemini-3.8-flash-low",
        "deep": "gemini-3.8-flash-medium",
        "fed": "gemini-3.8-flash-medium",
    },
    "argon": {
        "label": "Gemini 4 Argon",
        "lean": "gemini-4-argon-low",
        "planner": "gemini-4-argon-low",
        "repair": "gemini-4-argon-low",
        "deep": "gemini-4-argon-medium",
        "fed": "gemini-4-argon-medium",
    },
}

_DENY_TOOLS_SETTINGS = json.dumps({
    "permissions": {
        "deny": [
            "command(*)",
            "write_file(*)",
            "edit_file(*)",
            "read_file(*)",
        ]
    }
})

_FORBIDDEN_KEY_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
)


def extract_code_block(text: str) -> str:
    """Extract Python source from <CODE>...</CODE> or markdown code fences."""
    m = re.search(r"<CODE>(.*?)</CODE>", text, re.S | re.I)
    if m:
        body = m.group(1).strip()
    else:
        fences = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
        body = fences[-1].strip() if fences else text.strip()
    body = re.sub(r"^```(?:python)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    return body.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from <JSON>...</JSON>, markdown fences, or outermost braces."""
    m = re.search(r"<JSON>(.*?)</JSON>", text, re.S | re.I)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1).strip())
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    candidates.extend(reversed(fences))
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON object from response: {text[:240]!r}")


class AgyRunner:
    """Bounded-concurrency async runner for the `agy` CLI."""

    def __init__(self, max_concurrency: int = 10, timeout_sec: float = 110.0) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        per_fam = max(6, (max_concurrency * 4) // 5)
        self._fam_sems = {
            "flash": asyncio.Semaphore(per_fam),
            "argon": asyncio.Semaphore(per_fam),
        }
        self._launch_lock = asyncio.Lock()
        self._last_launch = 0.0
        self._timeout_sec = timeout_sec
        self._sandbox_cwd = Path(__file__).resolve().parent.parent / ".agy_cwd"
        self._sandbox_cwd.mkdir(parents=True, exist_ok=True)
        self.total_calls = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0
        self.total_wall_seconds = 0.0
        self.downgraded_calls = 0
        self.retry_events = 0

    def _clean_env(self) -> dict[str, str]:
        env = dict(os.environ)
        for k in _FORBIDDEN_KEY_VARS:
            env.pop(k, None)
        return env

    async def complete(
        self,
        model: str,
        prompt: str,
        max_retries: int = 9,
    ) -> dict[str, Any]:
        """Run a single non-interactive `agy -p` turn with tools denied."""
        full_prompt = (
            "IMPORTANT SYSTEM CONSTRAINT: Do NOT invoke or call ANY tools, shell commands, "
            "or file reads/writes. Respond directly in plain text in a single turn.\n\n"
            + prompt
        )
        fam_sem = self._fam_sems["flash"] if "flash" in model else self._fam_sems["argon"]
        last_err = ""
        for attempt in range(max_retries):
            if attempt > 0:
                self.retry_events += 1
            is_downgrade = attempt >= 2 and ("-medium" in model or "-high" in model)
            active_model = (
                model.replace("-medium", "-low").replace("-high", "-low")
                if is_downgrade
                else model
            )
            async with self._sem, fam_sem:
                async with self._launch_lock:
                    gap = 0.10 - (time.monotonic() - self._last_launch)
                    if gap > 0:
                        await asyncio.sleep(gap)
                    self._last_launch = time.monotonic()
                t0 = time.monotonic()
                proc = await asyncio.create_subprocess_exec(
                    AGY_BIN,
                    "--model",
                    active_model,
                    "--settings",
                    _DENY_TOOLS_SETTINGS,
                    "--disable-slash-commands",
                    "--output-format",
                    "json",
                    "-p",
                    full_prompt,
                    cwd=str(self._sandbox_cwd),
                    env=self._clean_env(),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                try:
                    out_b, err_b = await asyncio.wait_for(
                        proc.communicate(), timeout=self._timeout_sec
                    )
                except TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        proc.kill()
                    await proc.communicate()
                    last_err = f"Timeout after {self._timeout_sec}s"
                    await asyncio.sleep(3.5 * (attempt + 1))
                    continue

                dt = time.monotonic() - t0
                if proc.returncode != 0:
                    last_err = err_b.decode(errors="replace")[:300]
                    await asyncio.sleep(0.8 * (attempt + 1))
                    continue

                raw = out_b.decode(errors="replace").strip()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    lines = [ln for ln in raw.splitlines() if ln.strip().startswith("{")]
                    if not lines:
                        last_err = f"Non-JSON stdout: {raw[:200]}"
                        await asyncio.sleep(0.8 * (attempt + 1))
                        continue
                    payload = json.loads(lines[-1])

                resp_text = payload.get("response", "") or ""
                usage = payload.get("usage", {}) or {}
                if not resp_text.strip() and attempt + 1 < max_retries:
                    last_err = "Empty response string"
                    await asyncio.sleep(0.8 * (attempt + 1))
                    continue

                out_tok = int(usage.get("output_tokens", 0) or 0)
                think_tok = int(usage.get("thinking_tokens", 0) or 0)
                self.total_calls += 1
                self.total_output_tokens += out_tok
                self.total_thinking_tokens += think_tok
                self.total_wall_seconds += dt
                if is_downgrade:
                    self.downgraded_calls += 1
                return {
                    "response": resp_text,
                    "model": active_model,
                    "downgraded": is_downgrade,
                    "duration_seconds": round(dt, 3),
                    "num_turns": int(payload.get("num_turns", 1) or 1),
                    "output_tokens": out_tok,
                    "thinking_tokens": think_tok,
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                }

        raise RuntimeError(f"agy call failed after {max_retries} attempts ({model}): {last_err}")
