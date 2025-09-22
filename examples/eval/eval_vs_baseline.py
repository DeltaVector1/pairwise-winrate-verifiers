import os
import json
from typing import Any, Dict, List

from datasets import load_dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


CONFIG = {
    "JSONL_PATH": "/home/Ubuntu/Mango/verifiers/output.jsonl",
    "QUESTION_KEY": "text",

    "BASELINE_PATH": "./baseline.jsonl",
    "BASELINE_K": 1,
    "BASELINE_WORKERS": 64,
    "BASELINE_MICRO_BS": 64,

    "GEN_BASE_URL": "http://localhost:8000/v1",
    "GEN_MODEL": "NewEden/Snwy-SFT-GRPO-base",
    "GEN_API_KEY": "dummy",

    "JUDGE_BASE_URL": "https://firm-margin-impressed-moon.trycloudflare.com/v1",
    "JUDGE_MODEL": "NewEden/AFM-Judge-Step-39-V1",
    "JUDGE_API_KEY": "sk-",
    "EVAL_WORKERS": 64,
    "EVAL_MICRO_BS": 64,

    "RUBRIC_PATH": "/home/Ubuntu/Mango/verifiers/rubric.md",
}


def read_rubric(rubric_path: str | None) -> str | None:
    if not rubric_path:
        return None
    try:
        if os.path.isfile(rubric_path):
            with open(rubric_path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def make_policy_prompt(seed: str) -> List[Dict[str, str]]:
    user_text = (
        "Generate me a story with the following prompt\n\n"
        "<Story_Prompt>\n"
        f"{seed.strip()}\n"
        "</Story_Prompt>"
    )
    return [
        {"role": "system", "content": "You are a helpful writing assistant."},
        {"role": "user", "content": user_text},
    ]


SFT_SYSTEM_PROMPT = (
    "You will be given two samples contained between XML tags (<Sample-A></Sample-A> | <Sample-B></Sample-B>). "
    "You are tasked with judging both samples and determining which is better. "
    "Start with doing a small 1-3 sentence rationale contained within <rationale></rationale> XML tags justifying your final choice rating. "
    "Your final rating must be contained within Rating XML tags (<Rating></Rating>)."
)


def format_ab_prompt(user_seed: str, a_text: str, b_text: str, rubric_text: str | None) -> str:
    base = (
        "<Query>\nDecide whether Sample-A or Sample-B is better\n</Query>\n\n"
        f"<Sample-A>\n<user-prompt>\n{user_seed}\n</user-prompt>\n\n{a_text}\n</Sample-A>\n\n"
        f"<Sample-B>\n<user-prompt>\n{user_seed}\n</user-prompt>\n\n{b_text}\n</Sample-B>\n"
    )
    if rubric_text:
        return base + f"\n<Rubric>\n{rubric_text}\n</Rubric>\n"
    return base


def generate_baseline(jsonl_path: str, question_key: str, out_path: str, k: int, gen_client: OpenAI, gen_model: str) -> None:
    if jsonl_path.endswith(".jsonl") or jsonl_path.endswith(".json"):
        ds = load_dataset("json", data_files=jsonl_path, split="train")
    else:
        ds = load_dataset(jsonl_path, split="train")

    seeds: List[str] = [str(item.get(question_key, "")).strip() for item in ds]

    def gen_seed(seed: str) -> Dict[str, Any]:
        prompt = make_policy_prompt(seed)
        outputs: List[str] = []
        for _ in range(max(1, k)):
            resp = gen_client.chat.completions.create(
                model=gen_model,
                messages=prompt,
                temperature=0.0,
                max_completion_tokens=1024,
            )
            text = (resp.choices[0].message.content or "").strip()
            outputs.append(text)
        return {"prompt": seed, "baseline": outputs}

    workers = int(CONFIG.get("BASELINE_WORKERS", 32))
    micro = int(CONFIG.get("BASELINE_MICRO_BS", 64))
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(seeds), micro):
            chunk = seeds[start : start + micro]
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for rec in ex.map(gen_seed, chunk):
                    f.write(json.dumps(rec) + "\n")


def eval_vs_baseline(jsonl_path: str, question_key: str, baseline_path: str, gen_client: OpenAI, gen_model: str, judge_client: OpenAI, judge_model: str, rubric_text: str | None) -> Dict[str, Any]:
    # Load prompts
    if jsonl_path.endswith(".jsonl") or jsonl_path.endswith(".json"):
        ds = load_dataset("json", data_files=jsonl_path, split="train")
    else:
        ds = load_dataset(jsonl_path, split="train")
    # Load baseline map
    prompt_to_baseline: Dict[str, List[str]] = {}
    with open(baseline_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt_to_baseline[str(rec["prompt"])]= list(rec.get("baseline", []))

    seeds: List[str] = [str(item.get(question_key, "")).strip() for item in ds]

    # If baseline size mismatches dataset, regenerate to avoid partial baselines
    if len(prompt_to_baseline) < len(seeds):
        generate_baseline(
            jsonl_path=jsonl_path,
            question_key=question_key,
            out_path=baseline_path,
            k=int(CONFIG.get("BASELINE_K", 1)),
            gen_client=gen_client,
            gen_model=gen_model,
        )
        # reload
        prompt_to_baseline = {}
        with open(baseline_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompt_to_baseline[str(rec["prompt"])]= list(rec.get("baseline", []))

    def eval_one(seed: str) -> float:
        baselines = prompt_to_baseline.get(seed, [])
        if not baselines:
            return -1.0
        prompt = make_policy_prompt(seed)
        cur_resp = gen_client.chat.completions.create(
            model=gen_model,
            messages=prompt,
            temperature=0.0,
            max_completion_tokens=1024,
        )
        cur_text = (cur_resp.choices[0].message.content or "").strip()
        wins = 0
        total = 0
        # judge comparisons (sequential per seed keeps QPS sane; can thread if you need more)
        for b in baselines:
            ab_prompt = format_ab_prompt(seed, cur_text, b, rubric_text)
            j = judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": ab_prompt},
                ],
                temperature=0.0,
                max_completion_tokens=128,
            )
            raw_full = (j.choices[0].message.content or "").strip()
            # Parse <Rating>...</Rating> if present to avoid rationale bleed-through
            import re as _re
            m = _re.search(r"<Rating>([\s\S]*?)</Rating>", raw_full, flags=_re.IGNORECASE)
            rating_field = (m.group(1) if m else raw_full).strip().lower()
            total += 1
            if "sample-a" in rating_field and "sample-b" not in rating_field:
                wins += 1
        return (wins/total) if total > 0 else -1.0

    results: List[float] = []
    workers = int(CONFIG.get("EVAL_WORKERS", 32))
    micro = int(CONFIG.get("EVAL_MICRO_BS", 64))
    for start in range(0, len(seeds), micro):
        chunk = seeds[start : start + micro]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for wr in ex.map(eval_one, chunk):
                if wr >= 0.0:
                    results.append(wr)

    mean_wr = sum(results)/len(results) if results else 0.0
    return {
        "num_prompts": len(results),
        "mean_winrate_vs_baseline": mean_wr,
    }


def main():
    cfg = CONFIG
    rubric_text = read_rubric(cfg.get("RUBRIC_PATH"))

    gen = OpenAI(base_url=cfg["GEN_BASE_URL"], api_key=os.environ.get("GEN_API_KEY", cfg.get("GEN_API_KEY", "")))
    judge = OpenAI(base_url=cfg["JUDGE_BASE_URL"], api_key=os.environ.get("JUDGE_API_KEY", cfg.get("JUDGE_API_KEY", "")))

    if not os.path.isfile(cfg["BASELINE_PATH"]):
        generate_baseline(
            jsonl_path=cfg["JSONL_PATH"],
            question_key=cfg["QUESTION_KEY"],
            out_path=cfg["BASELINE_PATH"],
            k=int(cfg.get("BASELINE_K", 1)),
            gen_client=gen,
            gen_model=cfg["GEN_MODEL"],
        )

    stats = eval_vs_baseline(
        jsonl_path=cfg["JSONL_PATH"],
        question_key=cfg["QUESTION_KEY"],
        baseline_path=cfg["BASELINE_PATH"],
        gen_client=gen,
        gen_model=cfg["GEN_MODEL"],
        judge_client=judge,
        judge_model=cfg["JUDGE_MODEL"],
        rubric_text=rubric_text,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()


