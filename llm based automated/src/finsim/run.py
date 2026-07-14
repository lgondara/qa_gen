"""Entry point.

Examples
--------
# DeepSeek backend (needs DEEPSEEK_API_KEY) — the default:
python run.py --scenario bank_rumor

# hosted OpenAI backend (needs OPENAI_API_KEY):
python run.py --scenario bank_rumor --backend openai

# self-hosted vLLM (single or multiple replicas):
python run.py --scenario meme_stock --backend vllm \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --vllm-urls http://localhost:8000/v1

# then analyze:
python -m src.analysis results/bank_rumor.db Meridian
"""

import argparse
import asyncio

from src.models import make_models, make_models_from_config
from src.scenarios import SCENARIOS
from src.simulation import run_scenario
from src.analysis import report


def main():
    parser = argparse.ArgumentParser(description="Finance social simulation (OASIS)")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--backend", choices=["deepseek", "openai", "vllm"],
                        default="deepseek")
    parser.add_argument("--model-config", default=None,
                        help="JSON/YAML model config (overrides --backend); "
                             "see models.example.json")
    parser.add_argument("--model-name", default=None,
                        help="Model name (vLLM: HF repo id; "
                             "deepseek: 'deepseek-chat' [default])")
    parser.add_argument("--vllm-urls", nargs="+",
                        default=["http://localhost:8000/v1"],
                        help="One or more vLLM endpoint URLs")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override scenario step count")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    if args.steps:
        scenario.num_steps = args.steps

    if args.model_config:
        model = make_models_from_config(args.model_config)
    elif args.backend == "vllm":
        model = make_models(
            "vllm",
            model_name=args.model_name or "Qwen/Qwen2.5-7B-Instruct",
            urls=args.vllm_urls,
        )
    elif args.backend == "deepseek":
        model = make_models(
            "deepseek", model_name=args.model_name or "deepseek-chat"
        )
    else:
        model = make_models("openai")

    db_path = asyncio.run(run_scenario(scenario, model))

    keywords = {"bank_rumor": ["Meridian"], "meme_stock": ["ZVLT"]}.get(scenario.name)
    report(db_path, keywords)


if __name__ == "__main__":
    main()
