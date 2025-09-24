import os
import verifiers as vf
from openai import AsyncOpenAI, OpenAI
import traceback
import sys

CONFIG = {
    "JSONL_PATH": "/home/Ubuntu/Mango/verifiers/output.jsonl",
    "QUESTION_KEY": "text",
    "GEN_BASE_URL": "http://localhost:8000/v1",
    "GEN_MODEL": "Mawdistical/Kuwutu-7B",
    "GEN_API_KEY": "dummy",
    "JUDGE_BASE_URL": "https://firm-margin-impressed-moon.trycloudflare.com/v1",
    "JUDGE_MODEL": "NewEden/AFM-Judge-Step-39-V1",
    "JUDGE_API_KEY": "sk-",
    "WANDB_PROJECT": "Fenrisulfr",
    "WANDB_ENTITY": "new-eden",
    "WANDB_NAME": "Fenrisulfr-7B-lora-64",
    "JUDGE_MAX_WORKERS": "128",
    "JUDGE_MICRO_BS": "128",
    "RUBRIC_PATH": "/home/Ubuntu/Mango/verifiers/rubric.md",
    "BAD_NGRAMS_PATH": "/home/Ubuntu/Mango/verifiers/ngram.txt",
}
os.environ.update({k: str(v) for k, v in CONFIG.items() if v is not None})

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
        output_dir="outputs/pairwise-winrate-14b-lora-64",
        run_name="pairwise-winrate-14b-lora-64",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=8e-6,
        max_seq_len=16384,
        max_steps=500,
        num_generations=16,            # rollouts per prompt
        max_concurrent=32,
        num_batches_ahead=1,
        max_grad_norm=0.0001,
        weight_decay=0.0,
        async_generation_timeout=7200,# 2h timeout
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=4,
        log_on_each_node=False,
    )
    peft_config = vf.lora_defaults(r=64, alpha=16)

    # Swap to weighted rubric deviation
    try:
        from verifiers.rubrics.weighted_rubric_deviation import WeightedRubricDeviationRubric
        rubric_path = os.environ.get("RUBRIC_PATH", "rubric.md")
        with open(rubric_path, "r", encoding="utf-8") as f:
            rubric_text = f.read()
        judge_sync = OpenAI(
            base_url=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("JUDGE_API_KEY", ""),
        )
        vf_env.rubric = WeightedRubricDeviationRubric(
            judge_client=judge_sync,
            judge_model=os.environ.get("JUDGE_MODEL", "your-ab-rm-model"),
            rubric_text=rubric_text,
        )
        vf_env.rubric.ngram_n = 3
        vf_env.rubric.ngram_repeat_penalty = 0.02
        vf_env.rubric.complexity_penalty_scale = 0.02
        bad_path = os.environ.get("BAD_NGRAMS_PATH", "")
        if bad_path:
            vf_env.rubric.load_bad_ngrams(bad_path)
            vf_env.rubric.bad_ngram_penalty_scale = 0.05
    except Exception as e:
        print(f"[WARN] Falling back to existing rubric: {e}")

    trainer = vf.GRPOTrainer(
        model=model,
        env=vf_env,
        args=args,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # No pairwise settings needed for weighted rubric deviation
    try:
        trainer.train()
    except Exception:
        print("[TRAIN][ERROR] Uncaught exception during trainer.train():", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


