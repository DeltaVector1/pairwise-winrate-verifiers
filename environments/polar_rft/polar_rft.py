from __future__ import annotations

from typing import Any, Dict, List

import json
import sys
from pathlib import Path
import os

from datasets import Dataset

import verifiers as vf
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.parsers.think_parser import ThinkParser
from verifiers.rubrics.rubric import Rubric


def _import_polar_client(repo_root: Path | None = None):
    """
    Import POLARClient from the bundled POLAR_RFT directory.

    Falls back to raising ImportError with a helpful message if unavailable.
    """
    # Prefer local repo layout first: <repo>/POLAR_RFT/src
    if repo_root is None:
        # environments/polar_rft/ -> repo root two levels up
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
    # Fallback to installed package if available
    try:
        from polar.reward_func import POLARClient  # type: ignore

        return POLARClient
    except Exception:
        pass
    raise ImportError(
        "POLAR_RFT not found. Ensure POLAR_RFT/src is present in the repo or install the POLAR package."
    )


def _load_sharegpt_jsonl(path: str | Path) -> Dataset:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations", [])
            system_msg: str | None = None
            # Collect ordered user/assistant pairs
            pairs: List[tuple[str, str]] = []
            pending_user: str | None = None
            for turn in conv:
                origin = turn.get("from")
                value = str(turn.get("value", ""))
                if origin == "system" and system_msg is None:
                    system_msg = value
                elif origin == "human":
                    pending_user = value
                elif origin == "gpt" and pending_user is not None:
                    pairs.append((pending_user, value))
                    pending_user = None
            if not pairs:
                continue
            # Construct dataset rows per assistant turn
            for idx, (user_text, assistant_text) in enumerate(pairs):
                messages: List[Dict[str, str]] = []
                if system_msg:
                    messages.append({"role": "system", "content": system_msg})
                # include history up to previous assistant turns
                for prev_user, prev_assistant in pairs[:idx]:
                    messages.append({"role": "user", "content": prev_user})
                    messages.append({"role": "assistant", "content": prev_assistant})
                # current user prompt
                messages.append({"role": "user", "content": user_text})

                info = {
                    "conversation_id": obj.get("id"),
                    "turn_index": idx,
                }
                rows.append(
                    {
                        "prompt": messages,
                        "answer": assistant_text,
                        "task": "sharegpt-polar",
                        "info": info,
                    }
                )
    if not rows:
        raise ValueError(f"No usable conversations found in ShareGPT file: {path}")
    return Dataset.from_list(rows)


class POLARRFTEnv(SingleTurnEnv):
    """
    A SingleTurnEnv that scores completions with the POLAR reward model server.

    Supports classic QA datasets via `dataset_name` as well as ShareGPT-style
    JSONL via `dataset_path`.
    """

    def __init__(
        self,
        dataset_name: str = "gsm8k",
        split: str | None = None,
        n: int | None = None,
        seed: int = 0,
        dataset_path: str | None = None,
        polar_config: Dict[str, Any] | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        dataset: Dataset
        if dataset_path:
            dataset = _load_sharegpt_jsonl(dataset_path)
        else:
            dataset = vf.load_example_dataset(dataset_name, split=split, n=n, seed=seed)

        parser = ThinkParser()

        POLARClient = _import_polar_client()
        polar_config = polar_config or {}
        rm_path = polar_config.get("model_path", "internlm/POLAR-7B")
        rm_server_type = polar_config.get("server_type", "sglang")
        rm_address = polar_config.get("server_address", "127.0.0.1:30000")
        rm_max_length = polar_config.get("max_length", 16384)
        rm_max_response_length = polar_config.get("max_response_length", 4096)
        rm_response_cut_side = polar_config.get("response_cut_side", "right")

        debug_flag = bool(polar_config.get("debug", False))
        rm_client = POLARClient(
            path=rm_path,
            server_type=rm_server_type,
            server_address=rm_address,
            max_length=rm_max_length,
            max_response_length=rm_max_response_length,
            response_cut_side=rm_response_cut_side,
            debug=debug_flag,
        )

        rubric = Rubric(parser=parser)

        async def polar_reward(
            prompt, completion, answer, state, info, parser, **_: Any
        ) -> float:
            """
            Compute POLAR score for a single rollout using the reward server.
            Falls back to 0.0 on failure to keep training robust.
            """
            try:
                # Extract plain text from completion
                if isinstance(completion, str):
                    completion_text = completion
                else:
                    completion_text = str(completion[-1].get("content", "")) if completion else ""

                # POLAR client expects dict with prompt/reference/output
                data = [
                    {
                        "prompt": prompt,
                        "reference": answer,
                        "output": completion_text,
                        "wrapper": "sft",
                    }
                ]
                if debug_flag or os.getenv("POLAR_DEBUG", "").lower() in ("1", "true", "yes", "y"):
                    preview = completion_text[:160].replace("\n", " ")
                    print(f"[POLAR][call] server={rm_server_type} addr={rm_address} model={rm_path} sample='{preview}'")
                scores = rm_client(data)
                if scores and len(scores) > 0:
                    # scores typically in [-1, 1]; map to [0, 1]
                    s = float(scores[0])
                    return max(0.0, min(1.0, (s + 1.0) / 2.0))
            except Exception:
                return 0.0
            return 0.0

        rubric.add_reward_func(polar_reward, weight=1.0)
        rubric.add_reward_func(parser.get_format_reward_func(), weight=0.1)

        if system_prompt is None:
            system_prompt = (
                "Think step-by-step inside <think>...</think>. Then provide the final answer."
            )

        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            message_type="chat",
            **kwargs,
        )



def load_environment(**kwargs):
    return POLARRFTEnv(**kwargs)


