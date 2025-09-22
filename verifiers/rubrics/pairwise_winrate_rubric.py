from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

from verifiers.rubrics.rubric import Rubric
from verifiers.types import Info, Messages, RolloutScore, RolloutScores, State


@dataclass
class ABJudgeOutput:
    chosen_is_a: bool
    score: float
    rationale: str


class PairwiseWinrateRubric(Rubric):
    """
    Rubric that converts a set of rollouts for the same prompt into per-rollout
    winrates using a pairwise A/B judge (your pseudo-RM).

    The judge function should accept two samples A and B with optional context and
    return whether A is preferred over B (boolean) and an optional confidence score.

    This rubric overrides score_rollouts to compute, per example group (same prompt
    identity), the win count divided by number of opponents. If groups are not
    pre-bundled, we infer groups by exact prompt equality. For GRPO where the
    dataset is repeated per-example, equality grouping is typically sufficient.

    Metrics produced per rollout:
    - winrate: float in [0, 1]
    - pairwise_votes: raw integer wins
    - pairwise_total: total comparisons attempted

    The overall reward equals winrate by default (weight 1.0), suitable for GRPO.
    """

    def __init__(
        self,
        judge_fn: Callable[[Messages, Messages, Messages, Messages, Dict[str, Any]], ABJudgeOutput],
        group_by: Optional[Callable[[Messages, Info], Any]] = None,
        apply_weights: bool = True,
        parallelize_pairs: bool = True,
        max_pair_concurrency: int = 256,
        baseline_path: Optional[str] = None,
        baseline_weight: float = 0.0,
    ) -> None:
        # Store judging configuration
        self.judge_fn = judge_fn
        self.group_by = group_by
        self.apply_weights = apply_weights
        self.parallelize_pairs = parallelize_pairs
        self.max_pair_concurrency = max_pair_concurrency
        # Optional thread-pool micro-batching for judge requests (sync judge_fn)
        self.enable_pairwise_thread_pool: bool = False
        self.judge_max_workers: int = 32
        self.judge_micro_batch_size: int = 16
        # Baseline integration
        self.baseline_path: Optional[str] = baseline_path
        self.baseline_weight: float = max(0.0, min(1.0, baseline_weight))
        self.baseline_max_workers: int = 16
        self.baseline_micro_batch_size: int = 16
        self._baseline_map: Dict[str, List[str]] = {}

        # Initialize parent Rubric (sets parser, etc.) and default options
        super().__init__(funcs=[], weights=[], parser=None, parallelize_scoring=False)
        self.symmetric_pairs: bool = True
        self.include_self: bool = False

    # ------------------------
    # Helpers
    # ------------------------
    def _ensure_baseline_loaded(self) -> None:
        if not self.baseline_path or self._baseline_map:
            return
        try:
            if os.path.isfile(self.baseline_path):
                with open(self.baseline_path, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        seed = str(rec.get("prompt", ""))
                        bl = rec.get("baseline", [])
                        if seed:
                            self._baseline_map[seed] = [str(x) for x in bl]
        except Exception:
            self._baseline_map = {}

    def _extract_seed_from_prompt(self, prompt: Messages) -> str:
        if isinstance(prompt, list) and len(prompt) > 0:
            # find last user message
            for m in reversed(prompt):
                if isinstance(m, dict) and m.get("role") == "user":
                    content = str(m.get("content", ""))
                    break
            else:
                content = ""
            # extract between <Story_Prompt> ... </Story_Prompt>
            try:
                m = re.search(r"<Story_Prompt>\s*([\s\S]*?)\s*</Story_Prompt>", content, flags=re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
            return content.strip()
        return str(prompt)

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
        # Fallback path: if called individually, return zeros. Grouped scoring happens in score_rollouts.
        return RolloutScore(reward=0.0, metrics={"winrate": 0.0, "pairwise_votes": 0.0, "pairwise_total": 0.0})

    async def _call_judge(
        self,
        prompt_a: Messages,
        completion_a: Messages,
        prompt_b: Messages,
        completion_b: Messages,
        context: Dict[str, Any],
    ) -> ABJudgeOutput:
        # Expect a synchronous judge function
        result = self.judge_fn(prompt_a, completion_a, prompt_b, completion_b, context)
        assert isinstance(result, ABJudgeOutput), "judge_fn must return ABJudgeOutput"
        return result

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
        # Group indices by prompt identity (or user-provided key)
        groups: Dict[Any, List[int]] = {}
        for idx, (p, info) in enumerate(zip(prompts, infos)):
            if self.group_by is not None:
                key = self.group_by(p, info)
            else:
                # Default: group by exact prompt object stringification
                key = str(p)
            groups.setdefault(key, []).append(idx)

        n = len(prompts)
        rewards: List[float] = [0.0] * n
        winrates: List[float] = [0.0] * n
        votes: List[float] = [0.0] * n
        totals: List[float] = [0.0] * n

        # For each group, run pairwise judging
        for key, idxs in groups.items():
            if len(idxs) <= 1:
                # Single rollout in group -> neutral winrate
                i = idxs[0]
                winrates[i] = 1.0
                rewards[i] = 1.0
                votes[i] = 0.0
                totals[i] = 0.0
                continue

            # Build pair list (unordered i<j; optionally add symmetric j<i)
            pair_index_map: List[Tuple[int, int]] = []
            pair_contexts: List[Dict[str, Any]] = []
            for a_pos in range(len(idxs)):
                for b_pos in range(a_pos + 1, len(idxs)):
                    i = idxs[a_pos]
                    j = idxs[b_pos]
                    pair_index_map.append((i, j))
                    pair_contexts.append({
                        "task": tasks[i],
                        "answer": answers[i],
                        "info": infos[i],
                    })
                    if self.symmetric_pairs:
                        pair_index_map.append((j, i))
                        pair_contexts.append({
                            "task": tasks[j],
                            "answer": answers[j],
                            "info": infos[j],
                        })

            # Execute judging with optional thread-pool micro-batching
            pair_results: List[ABJudgeOutput] = []
            if self.enable_pairwise_thread_pool and self.judge_max_workers > 1:
                # chunk pairs
                for start in range(0, len(pair_index_map), self.judge_micro_batch_size):
                    end = min(len(pair_index_map), start + self.judge_micro_batch_size)
                    chunk_pairs = pair_index_map[start:end]
                    chunk_ctx = pair_contexts[start:end]
                    with ThreadPoolExecutor(max_workers=self.judge_max_workers) as ex:
                        futures = []
                        for (i, j), ctx in zip(chunk_pairs, chunk_ctx):
                            futures.append(
                                ex.submit(
                                    self.judge_fn,
                                    prompts[i],
                                    completions[i],
                                    prompts[j],
                                    completions[j],
                                    ctx,
                                )
                            )
                        for f in futures:
                            res = f.result()
                            assert isinstance(res, ABJudgeOutput)
                            pair_results.append(res)
            else:
                # Strictly sequential judging
                for (i, j), ctx in zip(pair_index_map, pair_contexts):
                    res = await self._call_judge(
                        prompts[i], completions[i], prompts[j], completions[j], ctx
                    )
                    pair_results.append(res)

            # Aggregate wins
            wins: Dict[int, int] = {i: 0 for i in idxs}
            for (i, j), out in zip(pair_index_map, pair_results):
                if out.chosen_is_a:
                    wins[i] += 1
                else:
                    wins[j] += 1

            # Denominator: count number of opponents (times 2 if symmetric comparisons)
            denom = max(1, (len(idxs) - 1) * (2 if self.symmetric_pairs else 1))
            if self.include_self:
                # Add a single automatic self-win per sample
                denom += 1
                for i in idxs:
                    wins[i] += 1

            # Compute intra-group winrates
            intra_wr: Dict[int, float] = {}
            for i in idxs:
                intra_wr[i] = float(wins[i] / denom)

            # Optional: winrate vs baseline
            self._ensure_baseline_loaded()
            baseline_wr: Dict[int, float] = {i: 0.0 for i in idxs}
            use_baseline = self.baseline_weight > 0.0 and len(self._baseline_map) > 0
            if use_baseline:
                for i in idxs:
                    prompt_i = prompts[i]
                    completion_i = completions[i]
                    seed = self._extract_seed_from_prompt(prompt_i)
                    blist = self._baseline_map.get(seed, [])
                    if not blist:
                        baseline_wr[i] = 0.0
                        continue
                    # judge current vs each baseline text
                    win = 0
                    tot = 0
                    for btxt in blist:
                        res = self.judge_fn(
                            prompt_i,
                            completion_i,
                            prompt_i,
                            [{"role": "assistant", "content": btxt}],
                            {"task": tasks[i], "answer": answers[i], "info": infos[i]},
                        )
                        tot += 1
                        if isinstance(res, ABJudgeOutput) and res.chosen_is_a:
                            win += 1
                    baseline_wr[i] = float(win / tot) if tot > 0 else 0.0

            # Combine rewards
            for i in idxs:
                intra = intra_wr[i]
                base = baseline_wr[i] if use_baseline else 0.0
                combined = (1.0 - self.baseline_weight) * intra + self.baseline_weight * base
                winrates[i] = float(intra)
                rewards[i] = float(combined)
                votes[i] = float(wins[i])
                totals[i] = float(denom)

        return RolloutScores(
            reward=rewards,
            metrics={
                "winrate": winrates,
                "baseline_wr": [0.0 for _ in rewards] if self.baseline_weight == 0.0 else [0.0 for _ in rewards],
                "pairwise_votes": votes,
                "pairwise_total": totals,
            },
        )


