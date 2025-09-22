from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    ) -> None:
        # Store judging configuration
        self.judge_fn = judge_fn
        self.group_by = group_by
        self.apply_weights = apply_weights
        self.parallelize_pairs = parallelize_pairs
        self.max_pair_concurrency = max_pair_concurrency

        # Parent rubric is initialized with no per-rollout independent reward funcs,
        # since we override score_rollouts to provide group-based rewards.
        super().__init__(funcs=[], weights=[], parser=None, parallelize_scoring=False)
        # Options
        self.symmetric_pairs: bool = True  # compare (A,B) and (B,A) to reduce order bias
        self.include_self: bool = False    # count self-comparison as an automatic win

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
            print(f"[RUBRIC] group={key} size={len(idxs)}", flush=True)
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

            # Strictly sequential judging (no asyncio primitives)
            pair_results: List[ABJudgeOutput] = []
            for (i, j), ctx in zip(pair_index_map, pair_contexts):
                print(f"[RUBRIC] judge_pair i={i} j={j}", flush=True)
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
            print(f"[RUBRIC] wins={wins}", flush=True)

            # Denominator: count number of opponents (times 2 if symmetric comparisons)
            denom = max(1, (len(idxs) - 1) * (2 if self.symmetric_pairs else 1))
            if self.include_self:
                # Add a single automatic self-win per sample
                denom += 1
                for i in idxs:
                    wins[i] += 1
            for i in idxs:
                wr = wins[i] / denom
                winrates[i] = float(wr)
                rewards[i] = float(wr)
                votes[i] = float(wins[i])
                totals[i] = float(denom)

        return RolloutScores(
            reward=rewards,
            metrics={
                "winrate": winrates,
                "pairwise_votes": votes,
                "pairwise_total": totals,
            },
        )


