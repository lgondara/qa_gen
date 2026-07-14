"""Advisory network entry point.

Examples
--------
# run a case with the sample population (DeepSeek backend):
python run_advisory.py --case preretirement_drawdown

# custom population, carrying institutional memory from prior runs:
python run_advisory.py --case downturn_panic \
    --clients data/clients.csv --advisors data/advisors.csv \
    --history results/preretirement_drawdown.db results/windfall_lump_sum.db

# adjudicate afterwards (add --judge for LLM scoring):
python -m src.advisory.adjudicate results/preretirement_drawdown.db --judge
# visualize:
python -m src.visualize results/preretirement_drawdown.db
"""

import argparse
import asyncio

from src.advisory.adjudicate import adjudicate
from src.advisory.cases import CASES
from src.advisory.runner import run_case
from src.models import make_models, make_models_from_config,\
    load_model_config, has_role_specs


def main():
    ap = argparse.ArgumentParser(description="Advisory network simulation")
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--clients", default=None, help="clients.csv path")
    ap.add_argument("--advisors", default=None, help="advisors.csv path")
    ap.add_argument("--history", nargs="*", default=[],
                    help="prior run DBs for institutional memory")
    ap.add_argument("--backend", choices=["deepseek", "openai", "vllm"],
                    default="deepseek")
    ap.add_argument("--model-config", default=None,
                    help="JSON/YAML model config; supports role-based "
                         "assignment (advisor/client), see models.example.json")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--vllm-urls", nargs="+",
                    default=["http://localhost:8000/v1"])
    ap.add_argument("--max-connections", type=int, default=50)
    ap.add_argument("--judge", action="store_true",
                    help="run LLM-judge adjudication at the end")
    args = ap.parse_args()

    role_models = None
    if args.model_config:
        specs = load_model_config(args.model_config)
        if has_role_specs(specs):
            role_models = {
                "advisor": make_models_from_config(specs, role="advisor"),
                "client": make_models_from_config(specs, role="client"),
                "all": make_models_from_config(specs),
            }
            model = role_models["all"]
        else:
            model = make_models_from_config(specs)
    elif args.backend == "vllm":
        model = make_models("vllm",
                            model_name=args.model_name or "Qwen/Qwen2.5-7B-Instruct",
                            urls=args.vllm_urls)
    elif args.backend == "deepseek":
        model = make_models("deepseek",
                            model_name=args.model_name or "deepseek-chat")
    else:
        model = make_models("openai")

    case = CASES[args.case]
    db = asyncio.run(run_case(
        case, model, clients_csv=args.clients, advisors_csv=args.advisors,
        history_dbs=args.history, max_connections=args.max_connections,
        role_models=role_models))
    ranked = adjudicate(db, use_judge=args.judge)

    from src.visualize import generate
    lb_cols = [c for c in ["uname", "role", "net_votes", "mentions",
                           "endorsement", "compliant", "judge_score"]
               if c in ranked.columns]
    lb = ranked[ranked.role == "advisor"].head(8)[lb_cols].round(1)
    generate(db, leaderboard=lb.to_dict("records"))


if __name__ == "__main__":
    main()
