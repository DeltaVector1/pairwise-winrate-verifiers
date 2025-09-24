from __future__ import annotations

from typing import Any, Dict, List

import json
import sys
from pathlib import Path

from datasets import Dataset

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers.think_parser import ThinkParser
from verifiers.rubrics.rubric import Rubric


def _import_polar_client(repo_root: Path | None = None):
    # Prefer local repo layout first: <repo>/POLAR_RFT/src
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    polar_src = repo_root / "POLAR_RFT" / "src"
    if polar_src.exists():
        sys.path.insert(0, str(polar_src))
        try:
            from polar.reward_func import POLARClient  # type: ignore

            return POLARClient
        except Exception as e:  # pragma: no cover
            raise ImportError(
                f"Failed to import POLARClient from {polar_src}: {e}"
            )
    # Fallback to installed package
    from polar.reward_func import POLARClient  # type: ignore

    return POLARClient


def _load_sharegpt_jsonl(path: str | Path) -> Dataset:
    path = str(path)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations", [])
            # Build initial prompt messages: include system if present, then first human
            messages: List[Dict[str, str]] = []
            system = next((x for x in conv if x.get("from") == "system"), None)
            if system and system.get("value"):
                messages.append({"role": "system", "content": str(system["value"])})
            # collect human/assistant in order
            user_msgs: List[str] = []
            assistant_refs: List[str] = []
            for turn in conv:
                if turn.get("from") == "human":
                    user_msgs.append(str(turn.get("value", "")))
                elif turn.get("from") == "gpt":
                    assistant_refs.append(str(turn.get("value", "")))
            if not user_msgs:
                continue
            # seed initial user
            messages.append({"role": "user", "content": user_msgs[0]})
            info: Dict[str, Any] = {
                "user_msgs": user_msgs,
                "assistant_refs": assistant_refs,
            }
            rows.append({
                "prompt": messages,
                "answer": "",  # not used; refs in info
                "task": "sharegpt-polar",
                "info": info,
            })
    return Dataset.from_list(rows)


class ShareGPTPOLAREnv(MultiTurnEnv):
    def __init__(
        self,
        dataset_path: str,
        polar_config: Dict[str, Any] | None = None,
        **kwargs,
    ):
        dataset = _load_sharegpt_jsonl(dataset_path)
        parser = ThinkParser()

        POLARClient = _import_polar_client()
        polar_config = polar_config or {}
        rm_path = polar_config.get("model_path", "internlm/POLAR-7B")
        rm_server_type = polar_config.get("server_type", "vllm")
        rm_address = polar_config.get("server_address", "http://127.0.0.1:8000")
        rm_max_length = polar_config.get("max_length", 16384)
        rm_max_response_length = polar_config.get("max_response_length", 4096)
        rm_response_cut_side = polar_config.get("response_cut_side", "right")
        debug_flag = bool(polar_config.get("debug", False))

        self.rm_client = POLARClient(
            path=rm_path,
            server_type=rm_server_type,
            server_address=rm_address,
            max_length=rm_max_length,
            max_response_length=rm_max_response_length,
            response_cut_side=rm_response_cut_side,
            debug=debug_flag,
        )

        rubric = Rubric(parser=parser)

        async def polar_multi_reward(prompt, completion, state, **_: Any) -> float:
            # Extract generated assistant messages
            assert isinstance(completion, list)
            gen_assistant = [m.get("content", "") for m in completion if m.get("role") == "assistant"]
            ref_assistant: List[str] = state.get("assistant_refs", [])
            user_msgs: List[str] = state.get("user_msgs", [])
            # Build batch for POLAR per assistant turn (align by index)
            batch = []
            max_i = min(len(gen_assistant), len(ref_assistant))
            # Build prompt context for each turn i using system + alternating prior turns
            # We can reconstruct from state["base_system"] and user_msgs
            base_system: str = state.get("base_system", "")
            for i in range(max_i):
                # context: optional system + all user/assistant up to user i
                msgs: List[str] = []
                if base_system:
                    msgs.append(base_system)
                for j in range(i + 1):
                    msgs.append(user_msgs[j] if j < len(user_msgs) else "")
                    if j < len(gen_assistant):
                        # Use generated assistant so far for context
                        msgs.append(gen_assistant[j])
                ctx = "\n".join(msgs)
                batch.append({
                    "prompt": ctx,
                    "reference": ref_assistant[i] if i < len(ref_assistant) else "",
                    "output": gen_assistant[i],
                    "wrapper": "sft",
                })
            if not batch:
                return 0.0
            scores = self.rm_client(batch) or []
            if not scores:
                return 0.0
            # Map [-1,1] -> [0,1]
            norm = [max(0.0, min(1.0, (float(s) + 1.0) / 2.0)) for s in scores]
            return float(sum(norm) / len(norm))

        rubric.add_reward_func(polar_multi_reward, weight=1.0)

        super().__init__(
            dataset=dataset,
            system_prompt=None,
            parser=parser,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )

    async def setup_state(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Extract info
        info: Dict[str, Any] = state.get("info", {})
        user_msgs: List[str] = list(info.get("user_msgs", []))
        assistant_refs: List[str] = list(info.get("assistant_refs", []))
        # Cache base system from first message if present
        base_system = ""
        for m in state.get("prompt", []):
            if m.get("role") == "system":
                base_system = str(m.get("content", ""))
                break
        state["user_msgs"] = user_msgs
        state["assistant_refs"] = assistant_refs
        state["base_system"] = base_system
        state["assistant_count"] = 0
        state["next_user_idx"] = 1  # we already seeded user_msgs[0]
        return state

    async def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> bool:
        # Complete after we produced as many assistant turns as references
        gen_assistant = [m for m in messages if m.get("role") == "assistant"]
        return len(gen_assistant) >= len(state.get("assistant_refs", []))

    async def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
        # After each assistant, append the next ground-truth user turn, if any
        next_user_idx: int = state.get("next_user_idx", 1)
        user_msgs: List[str] = state.get("user_msgs", [])
        if next_user_idx < len(user_msgs):
            nxt = user_msgs[next_user_idx]
            state["next_user_idx"] = next_user_idx + 1
            return ([{"role": "user", "content": nxt}], state)
        return ([], state)


def load_environment(**kwargs):
    return ShareGPTPOLAREnv(**kwargs)


