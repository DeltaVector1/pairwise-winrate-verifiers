import os
import re
from typing import Any, Dict
from openai import OpenAI
from datasets import load_dataset

import verifiers as vf
from verifiers import PairwiseWinrateRubric
from verifiers.rubrics.pairwise_winrate_rubric import ABJudgeOutput


SFT_SYSTEM_PROMPT = (
    "You will be given two samples contained between XML tags (<Sample-A></Sample-A> | <Sample-B></Sample-B>). "
    "You are tasked with judging both samples and determining which is better. "
    "Start with doing a small 1-3 sentence rationale contained within <rationale></rationale> XML tags justifying your final choice rating. "
    "Your final rating must be contained within Rating XML tags (<Rating></Rating>)."
)


def _extract_user_prompt(messages) -> str:
    try:
        if isinstance(messages, list):
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    return str(m.get("content", "")).strip()
    except Exception:
        pass
    return ""


def _format_ab_prompt(prompt_a, completion_a, prompt_b, completion_b, rubric_text: str | None = None) -> str:
    # Exact XML structure to match SFT prompting
    a_text = completion_a if isinstance(completion_a, str) else completion_a[-1].get("content", "")
    b_text = completion_b if isinstance(completion_b, str) else completion_b[-1].get("content", "")
    user_text = _extract_user_prompt(prompt_a) or _extract_user_prompt(prompt_b)
    base = (
        "<Query>\nDecide whether Sample-A or Sample-B is better\n</Query>\n\n"
        f"<Sample-A>\n<user-prompt>\n{user_text}\n</user-prompt>\n\n{a_text}\n</Sample-A>\n\n"
        f"<Sample-B>\n<user-prompt>\n{user_text}\n</user-prompt>\n\n{b_text}\n</Sample-B>\n"
    )
    if rubric_text:
        return base + f"\n<Rubric>\n{rubric_text}\n</Rubric>\n"
    return base


def load_environment(
    dataset_name: str,
    split: str = "train",
    question_key: str = "text",
    answer_key: str | None = None,
    group_key: str | None = None,
    judge_model: str = "your-ab-rm-model",
    judge_base_url: str = "http://0.0.0.0:8000/v1",
    judge_api_key_var: str = "JUDGE_API_KEY",
    rollouts_per_example: int = 8,
    rubric_path: str | None = "/home/Ubuntu/Mango/verifiers/rubric.md",
):

    if dataset_name.endswith(".jsonl") or dataset_name.endswith(".json"):
        ds = load_dataset("json", data_files=dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    def _map_row(x: Dict[str, Any]) -> Dict[str, Any]:
        # Policy rollout prompt wrapper
        user_text = (
            "Generate me a story with the following prompt\n\n"
            "<Story_Prompt>\n"
            f"{x.get(question_key, '').strip()}\n"
            "</Story_Prompt>"
        )
        prompt = [
            {"role": "system", "content": "You are an amazing writing assistant."},
            {"role": "user", "content": user_text},
        ]
        info: Dict[str, Any] = {}
        if group_key and group_key in x:
            info["group_id"] = str(x[group_key])
        return {
            "prompt": prompt,
            "answer": x.get(answer_key, "") if answer_key else "",
            "task": "pairwise-winrate",
            "info": info,
        }

    ds = ds.map(_map_row, remove_columns=ds.column_names)

    api_key = os.getenv(judge_api_key_var, "EMPTY")
    judge_client = OpenAI(base_url=judge_base_url, api_key=api_key)

    rubric_text: str | None = None
    try:
        if rubric_path and os.path.isfile(rubric_path):
            with open(rubric_path, "r", encoding="utf-8") as f:
                rubric_text = f.read().strip()
    except Exception:
        rubric_text = None

    def judge_fn(prompt_a, completion_a, prompt_b, completion_b, context: Dict[str, Any]) -> ABJudgeOutput:
        user_prompt = _format_ab_prompt(prompt_a, completion_a, prompt_b, completion_b, rubric_text=rubric_text)
        resp = judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": SFT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=512,
            temperature=0.8,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"<Rating>([\s\S]*?)</Rating>", raw, flags=re.IGNORECASE)
        rating_text = (m.group(1) if m else raw).strip().lower()
        chosen_is_a = "sample-a" in rating_text and not "sample-b" in rating_text
        return ABJudgeOutput(chosen_is_a=bool(chosen_is_a), score=1.0, rationale=raw)

    rubric = PairwiseWinrateRubric(
        judge_fn=judge_fn,
        group_by=lambda prompt, info: info.get("group_id", str(prompt)),
    )
    rubric.symmetric_pairs = True
    rubric.include_self = False  

    vf_env = vf.SingleTurnEnv(
        dataset=ds,
        rubric=rubric,
        system_prompt=None,
    )

    return vf_env


