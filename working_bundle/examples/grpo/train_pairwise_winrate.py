import os
import verifiers as vf
from openai import AsyncOpenAI
import traceback, sys

def main():
    policy_model = os.environ.get("POLICY_MODEL", os.environ.get("GEN_MODEL", "your-generation-model"))

    vf_env = vf.load_environment(
        env_id="pairwise-winrate",
        dataset_name=os.environ.get("JSONL_PATH", "/path/to/prompts.jsonl"),
        split="train",
        question_key=os.environ.get("QUESTION_KEY", "text"),
        group_key=os.environ.get("GROUP_KEY", None),
        judge_model=os.environ.get("JUDGE_MODEL", "your-ab-rm-model"),
        judge_base_url=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
        judge_api_key_var=os.environ.get("JUDGE_API_KEY_VAR", "JUDGE_API_KEY"),
        rollouts_per_example=int(os.environ.get("ROLLOUTS", "8")),
    )

    vf_env.client = AsyncOpenAI(
        api_key=os.environ.get("GEN_API_KEY", "dummy"),
        base_url=os.environ.get("GEN_BASE_URL", "http://localhost:8000/v1"),
    )
    vf_env.model = os.environ.get("GEN_MODEL", policy_model)

    model, tokenizer = vf.get_model_and_tokenizer(model_name=policy_model)

    args = vf.GRPOConfig(
        output_dir="outputs/pairwise-winrate",
        run_name="pairwise-winrate",
        # core hparams
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        max_seq_len=4096,
        max_steps=500,
        num_generations=8,            # rollouts per prompt
        max_concurrent=1,             # serialize env requests
        num_batches_ahead=0,          # no async prefetch
        async_generation_timeout=7200,# 2h timeout
        bf16=True,
        gradient_checkpointing=True,
        report_to=None,
    )

    trainer = vf.GRPOTrainer(
        model=model,
        env=vf_env,
        args=args,
        processing_class=tokenizer,
    )

    if hasattr(vf_env, "rubric") and hasattr(vf_env.rubric, "parallelize_pairs"):
        vf_env.rubric.parallelize_pairs = False
        try:
            vf_env.rubric.max_pair_concurrency = int(os.environ.get("JUDGE_MAX_CONC", "1"))
        except Exception:
            vf_env.rubric.max_pair_concurrency = 1
        sp = os.environ.get("SYMMETRIC_PAIRS", "true").lower() == "true"
        try:
            vf_env.rubric.symmetric_pairs = sp
        except Exception:
            pass
    try:
        trainer.train()
    except Exception:
        print("[TRAIN][ERROR] Uncaught exception during trainer.train():", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


