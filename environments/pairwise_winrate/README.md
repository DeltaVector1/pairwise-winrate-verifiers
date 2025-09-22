pairwise-winrate

Install
```bash
vf-install pairwise-winrate (-p /path/to/environments | --from-repo)
```

Quick eval
```bash
vf-eval pairwise-winrate -m your-generation-model -n 4 -r 8 \
  --dataset /path/to/prompts.jsonl --question-key text \
  --judge-model your-ab-rm-model --judge-base-url http://localhost:8000/v1
```

This environment expects JSONL with a `text` field. It wraps two rollouts into your XML format and asks the judge to output <Rating> with Sample-A or Sample-B.


