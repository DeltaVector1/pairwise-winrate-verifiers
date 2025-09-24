from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import verifiers as vf
from openai import AsyncOpenAI


def load_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    with cfg_path.open("rb") as f:
        return tomllib.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "configs" / "polar_rft.toml"),
        help="Path to TOML config file",
    )
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)

    # 1) Create environment (optional remote inference client)
    env_cfg = cfg.get("env", {})
    client_cfg = cfg.get("client", {})

    policy_client = None
    client_model_override = None
    if client_cfg.get("base_url"):
        policy_client = AsyncOpenAI(
            base_url=client_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
            api_key=client_cfg.get("api_key", "dummy"),
        )
        client_model_override = client_cfg.get("model")

    env = vf.load_environment(
        "polar_rft",
        **env_cfg,
    )

    # 2) Load model and tokenizer
    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B-Instruct")
    model, tokenizer = vf.get_model_and_tokenizer(model_name)

    # 3) Configure TRL/GRPO args
    tcfg = cfg.get("trainer", {})
    run_name = tcfg.get("run_name", f"polar-rft_{model_name.split('/')[-1].lower()}")
    trl_args = vf.grpo_defaults(run_name=run_name)
    if policy_client is not None:
        trl_args.client = policy_client
        trl_args.model = client_model_override or model_name

    # Core batch/schedule
    trl_args.per_device_train_batch_size = tcfg.get("per_device_train_batch_size", 2)
    trl_args.num_generations = tcfg.get("num_generations", 8)
    trl_args.gradient_accumulation_steps = tcfg.get("gradient_accumulation_steps", 2)
    trl_args.max_steps = tcfg.get("max_steps", 50)
    trl_args.learning_rate = tcfg.get("learning_rate", 1e-6)
    trl_args.beta = tcfg.get("beta", 0.001)
    trl_args.max_grad_norm = tcfg.get("max_grad_norm", 0.1)
    trl_args.num_iterations = tcfg.get("num_iterations", 1)
    trl_args.loss_type = tcfg.get("loss_type", "dr_grpo")
    trl_args.epsilon = tcfg.get("epsilon", 0.2)

    # Generation/sampling
    trl_args.max_tokens = tcfg.get("max_tokens", 512)
    trl_args.max_prompt_length = tcfg.get("max_prompt_length", 512)
    trl_args.max_completion_length = tcfg.get("max_completion_length", 1024)
    trl_args.temperature = tcfg.get("temperature", 1.0)
    trl_args.top_p = tcfg.get("top_p", 1.0)
    trl_args.top_k = tcfg.get("top_k", None)

    # Logging/infra
    trl_args.logging_steps = tcfg.get("logging_steps", 1)
    trl_args.report_to = tcfg.get("report_to", "none")

    # 4) Train
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=trl_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()


