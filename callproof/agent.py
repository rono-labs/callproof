"""A tool-using issue-localization agent for Gemma 4 served by llama.cpp's OpenAI-compatible server."""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from callproof.tools import Repo, ToolBox

SYSTEM_PROMPT = (
    "You localize the code that must change to resolve a GitHub issue in a Python repository (checked out at the "
    "base commit). Explore with the tools, using at most {budget} tool calls. When done, reply with ONLY a JSON "
    'list of up to 5 strings "path::qualname" (path relative to the repository root, qualname such as '
    '"Class.method" or "function"), best candidate first. Example: ["fastapi/routing.py::APIRoute.__init__"]. '
    "Always give 5 candidates, even when unsure: an educated guess scores better than an empty list. "
    "Graph tools name symbols as dotted ids (module.Class.method); those are also accepted as answers. "
    "Do not propose fixes or write code, and keep any reply outside tool calls to the JSON list."
)
_ANSWER_PATTERN = re.compile(r"[\w./-]+\.py::[\w.<>]+")
_TOOL_MARKERS = ("<|tool_call", "tool_call>", "call:")


@dataclass(frozen=True)
class AgentConfig:
    tool_set: str = "both"
    budget: int = 15
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    seed: int | None = None
    max_tokens: int = 400
    answer_tokens: int = 200
    thinking: bool = False
    max_issue_chars: int = 3000
    max_hint_chars: int = 800
    context_size: int = 24576
    request_timeout: float = 900.0


class LlamaServer:
    """A `llama-server` subprocess; use as a context manager so it is always stopped."""

    def __init__(self, binary: Path, model: Path, port: int, threads: int, context_size: int, log_path: Path):
        self.port = port
        self._command = [
            str(binary), "-m", str(model), "--jinja", "-t", str(threads), "-c", str(context_size),
            "--port", str(port), "--host", "127.0.0.1", "--parallel", "1",
        ]
        self._log_path = log_path
        self._process: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LlamaServer:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as log:
            self._process = subprocess.Popen(self._command, stdout=log, stderr=subprocess.STDOUT)
        return self

    def wait_ready(self, timeout: float = 300.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self._log_path}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        return
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                pass
            time.sleep(1)
        raise TimeoutError(f"llama-server on port {self.port} not ready after {timeout}s")

    def __exit__(self, *exc: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()


def parse_answer(text: str) -> list[str]:
    """Extract candidate locations from a model reply: a JSON list if present, else any `file.py::name` tokens."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            items = None
        if isinstance(items, list):
            return [
                item if isinstance(item, str) else f"{item.get('path')}::{item.get('qualname')}"
                for item in items
                if isinstance(item, (str, dict))
            ]
    return _ANSWER_PATTERN.findall(text)


class LocalizationAgent:
    def __init__(self, base_url: str, config: AgentConfig):
        self.base_url = base_url
        self.config = config

    def _chat(self, messages: list[dict], tools: list[dict] | None) -> dict:
        cfg = self.config
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "max_tokens": cfg.max_tokens if tools else cfg.answer_tokens,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": cfg.thinking},
        }
        if cfg.seed is not None:
            payload["seed"] = cfg.seed
        if tools:
            payload["tools"] = tools
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=cfg.request_timeout) as response:
            return json.load(response)

    def run(self, task: dict, toolbox: ToolBox, repo: Repo) -> dict:
        cfg = self.config
        issue = task["problem_statement"][: cfg.max_issue_chars]
        hints = (task.get("hints_text") or "")[: cfg.max_hint_chars]
        prompt = f"Issue:\n{issue}" + (f"\n\nHints:\n{hints}" if hints.strip() else "")
        if task.get("anchors"):
            prompt += "\n\nKeyword search over function names and code suggests these starting points:\n" + "\n".join(task["anchors"])
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(budget=cfg.budget)},
            {"role": "user", "content": prompt},
        ]
        turns: list[dict] = []
        calls: list[dict] = []
        started = time.time()
        answer_text, force_final = "", False
        while len(turns) < cfg.budget * 2 + 4:
            tools = None if force_final else toolbox.schemas
            turn_started = time.time()
            response = self._chat(messages, tools)
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            usage = response.get("usage", {})
            timings = response.get("timings", {})
            turns.append(
                {
                    "seconds": round(time.time() - turn_started, 2),
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": timings.get("cache_n"),
                    "prompt_ms": timings.get("prompt_ms"),
                    "predicted_ms": timings.get("predicted_ms"),
                    "reasoning_chars": len(message.get("reasoning_content") or ""),
                    "leaked_tool_syntax": bool(not tool_calls and any(m in content for m in _TOOL_MARKERS)),
                }
            )
            messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls") and v is not None})
            messages[-1].setdefault("content", "")
            near_limit = (usage.get("prompt_tokens") or 0) > cfg.context_size - cfg.max_tokens - 1500
            if tool_calls and not force_final:
                for call in tool_calls:
                    function = call.get("function", {})
                    name, arguments = function.get("name", ""), function.get("arguments")
                    if len(calls) >= cfg.budget:
                        result, valid = '{"status":"error","error_message":"Tool budget exhausted"}', False
                    else:
                        tick = time.time()
                        result, valid = toolbox.call(name, arguments)
                        calls.append(
                            {
                                "name": name,
                                "arguments": arguments,
                                "valid": valid,
                                "ok": valid and '"status": "ok"' in result[:20],
                                "result_chars": len(result),
                                "seconds": round(time.time() - tick, 3),
                            }
                        )
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
                if len(calls) >= cfg.budget or near_limit:
                    force_final = True
                    messages.append({"role": "user", "content": "No more tool calls. Reply now with the final JSON list of 5 candidates."})
                continue
            answer_text = content
            if parse_answer(content) or force_final:
                break
            force_final = True
            messages.append({"role": "user", "content": "Reply now with only the final JSON list of 5 path::qualname candidates."})
        predictions: list[str] = []
        for item in parse_answer(answer_text):
            normalized = repo.normalize(item)
            if normalized and normalized not in predictions:
                predictions.append(normalized)
        return {
            "predictions": predictions[:5],
            "answer_text": answer_text,
            "config": asdict(cfg),
            "tool_calls": calls,
            "turns": turns,
            "messages": messages,
            "wall_seconds": round(time.time() - started, 2),
        }
