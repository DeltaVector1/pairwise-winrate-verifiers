from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RolloutScore, RolloutScores, State


@dataclass
class JudgeRubricScores:
    total: float
    per_skill: Dict[str, float]


def _dynamic_import(module_path: str, module_name: str):
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
    except Exception:
        return None
    return None


class WeightedRubricDeviationRubric(Rubric):
    def __init__(
        self,
        judge_client: OpenAI,
        judge_model: str,
        rubric_text: str,
        parallelize_scoring: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(funcs=[], weights=[], parser=None, parallelize_scoring=parallelize_scoring)
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.rubric_text = rubric_text

        # Penalty knobs
        self.ngram_n: int = 3
        self.ngram_repeat_penalty: float = 0.02
        self.complexity_penalty_scale: float = 0.02
        self.bad_ngram_penalty_scale: float = 0.05
        self.bad_ngrams_path: Optional[str] = None
        self._bad_ngrams: List[str] = []

        # Optional dynamic imports for richer penalties
        self._complexity_mod = _dynamic_import("complexity.py", "complexity_mod")
        self._ngram_mod = _dynamic_import("ngram-filter.py", "ngram_filter_mod")

    def load_bad_ngrams(self, path: Optional[str]) -> None:
        self.bad_ngrams_path = path
        self._bad_ngrams = []
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    phrase = line.strip().lower()
                    if phrase:
                        self._bad_ngrams.append(phrase)
        except Exception:
            self._bad_ngrams = []

    def _make_policy_seed(self, prompt: Messages) -> str:
        if isinstance(prompt, list):
            for m in reversed(prompt):
                if isinstance(m, dict) and m.get("role") == "user":
                    content = str(m.get("content", ""))
                    break
            else:
                content = ""
            try:
                m = re.search(r"<Story_Prompt>\s*([\s\S]*?)\s*</Story_Prompt>", content, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
            return content.strip()
        return str(prompt)

    def _format_single_prompt(self, seed: str, sample_text: str) -> str:
        return (
            "You are a strict judge. Score the sample using the rubric.\n\n"
            "Return JSON with keys: scores (per skill dict), total (float).\n\n"
            "<Rubric>\n" + self.rubric_text + "\n</Rubric>\n\n"
            "<Story_Prompt>\n" + seed + "\n</Story_Prompt>\n\n"
            "<Sample>\n" + sample_text + "\n</Sample>\n"
        )

    def _judge_score(self, seed: str, sample_text: str) -> JudgeRubricScores:
        user_prompt = self._format_single_prompt(seed, sample_text)
        resp = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "Score only. Be deterministic."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_completion_tokens=256,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Try parse JSON block
        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            blob = m.group(0) if m else raw
            data = json.loads(blob)
            total = float(data.get("total", 0.0))
            per = {str(k): float(v) for k, v in (data.get("scores", {}) or {}).items()}
            return JudgeRubricScores(total=total, per_skill=per)
        except Exception:
            # Fallback: scan for total: <number>
            m = re.search(r"total\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", raw, flags=re.IGNORECASE)
            tot = float(m.group(1)) if m else 0.0
            return JudgeRubricScores(total=tot, per_skill={})

    # ---------- Penalties ----------
    def _ngram_penalty(self, text: str) -> float:
        # Prefer imported tokenizer
        tokens: List[str] = []
        if self._ngram_mod and hasattr(self._ngram_mod, "improved_tokenize"):
            try:
                tokens = self._ngram_mod.improved_tokenize(text, no_punctuation=False)
            except Exception:
                pass
        if not tokens:
            tokens = re.findall(r"\w+|[.,!?;]", text.lower())

        n = self.ngram_n
        counts: Dict[Tuple[str, ...], int] = {}
        for i in range(len(tokens) - n + 1):
            tup = tuple(tokens[i : i + n])
            counts[tup] = counts.get(tup, 0) + 1
        repeats = sum(c - 1 for c in counts.values() if c > 1)
        penalty = repeats * self.ngram_repeat_penalty
        # penalize explicit bad n-grams (substring match for simplicity)
        if self._bad_ngrams:
            lower_text = " ".join(tokens)
            for phrase in self._bad_ngrams:
                # count rough occurrences (non-overlapping not enforced)
                occ = lower_text.count(phrase)
                if occ > 0:
                    penalty += occ * self.bad_ngram_penalty_scale
        return penalty

    def _complexity_penalty(self, text: str) -> float:
        if self._complexity_mod and hasattr(self._complexity_mod, "analyze_complexity"):
            try:
                res = self._complexity_mod.analyze_complexity(text)
                score = float(res.get("score", 0.0))
                return score * self.complexity_penalty_scale
            except Exception:
                pass
        # Fallback: simple heuristic
        sentences = re.split(r"[.!?]\s+", text.strip())
        long_sentences = sum(1 for s in sentences if len(s.split()) > 40)
        return long_sentences * (self.complexity_penalty_scale * 2.0)

    # ---------- Rubric API ----------
    async def score_rollout(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        task: str = "default",
        info: Info | None = None,
        **kwargs,
    ) -> RolloutScore:
        # Not used; we override score_rollouts for group logic
        return RolloutScore(reward=0.0, metrics={})

    async def score_rollouts(
        self,
        prompts: List[Messages],
        completions: List[Messages],
        answers: List[str],
        states: List[State],
        tasks: List[str],
        infos: List[Info],
        max_concurrent: int = -1,
        **kwargs,
    ) -> RolloutScores:
        # Group indices by prompt seed
        seed_to_indices: Dict[str, List[int]] = {}
        for idx, p in enumerate(prompts):
            seed = self._make_policy_seed(p)
            seed_to_indices.setdefault(seed, []).append(idx)

        rewards: List[float] = [0.0] * len(prompts)
        metrics_total: List[float] = [0.0] * len(prompts)
        metrics_dev: List[float] = [0.0] * len(prompts)
        metrics_pen_ng: List[float] = [0.0] * len(prompts)
        metrics_pen_cx: List[float] = [0.0] * len(prompts)

        for seed, idxs in seed_to_indices.items():
            # Judge each rollout for raw total
            totals: Dict[int, float] = {}
            for i in idxs:
                # Extract assistant text
                comp = completions[i]
                if isinstance(comp, list):
                    sample_text = "\n".join([
                        m.get("content", "") for m in comp if isinstance(m, dict) and m.get("role") == "assistant"
                    ])
                else:
                    sample_text = str(comp)
                jr = self._judge_score(seed, sample_text)
                totals[i] = jr.total
                metrics_total[i] = jr.total

            # Deviation from mean
            if totals:
                mean_total = sum(totals.values()) / len(totals)
            else:
                mean_total = 0.0

            for i in idxs:
                comp = completions[i]
                if isinstance(comp, list):
                    sample_text = "\n".join([
                        m.get("content", "") for m in comp if isinstance(m, dict) and m.get("role") == "assistant"
                    ])
                else:
                    sample_text = str(comp)

                deviation = totals[i] - mean_total
                pen_ng = self._ngram_penalty(sample_text)
                pen_cx = self._complexity_penalty(sample_text)
                final_reward = deviation - pen_ng - pen_cx

                rewards[i] = final_reward
                metrics_dev[i] = deviation
                metrics_pen_ng[i] = pen_ng
                metrics_pen_cx[i] = pen_cx

        return RolloutScores(
            reward=rewards,
            metrics={
                "rubric_total": metrics_total,
                "deviation": metrics_dev,
                "penalty_ngram": metrics_pen_ng,
                "penalty_complexity": metrics_pen_cx,
            },
        )


