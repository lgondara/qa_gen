"""Universal model backends for agent LLMs.

Any model reachable through CAMEL's ModelFactory can drive agents. A model
is described by a *spec* (dict), several specs form a *config* (JSON file),
and configs support weighted mixing and per-role assignment.

Spec fields:
    platform     one of the aliases below (required)
    model        model identifier string (required)
    url          endpoint URL where applicable (vllm/ollama/compatible/bedrock)
    api_key_env  env var holding the key (defaults per platform)
    count        weight for mixed populations (default 1) — OASIS's
                 ModelManager round-robins agents across the expanded list
    role         "advisor" | "client" | "all" (advisory mode only)

Platform aliases -> CAMEL ModelPlatformType:
    deepseek            DEEPSEEK          (DEEPSEEK_API_KEY)
    openai              OPENAI            (OPENAI_API_KEY)
    anthropic           ANTHROPIC         (ANTHROPIC_API_KEY)
    gemini              GEMINI            (GOOGLE_API_KEY)
    vllm                VLLM              (local server; url required)
    ollama              OLLAMA            (local; url optional)
    sglang              SGLANG            (local server; url required)
    bedrock             AWS_BEDROCK       (Bedrock's OpenAI-compatible
                                           endpoint: BEDROCK_API_BASE_URL +
                                           BEDROCK_API_KEY, or spec url/key)
    bedrock-converse    AWS_BEDROCK_CONVERSE (Converse API via standard AWS
                                           credentials/region env)
    openai-compatible   OPENAI_COMPATIBLE_MODEL (catch-all: any OpenAI-style
                                           endpoint — Groq, Together, Azure,
                                           OpenRouter, LM Studio, ...)

Verified against camel master: AWS_BEDROCK wraps Bedrock's OpenAI-compatible
API (AWSBedrockModel subclasses OpenAICompatibleModel and requires
BEDROCK_API_BASE_URL / BEDROCK_API_KEY); AWS_BEDROCK_CONVERSE uses the
native Converse API. Prefer `bedrock` if you have gateway API keys, and
`bedrock-converse` if you authenticate with standard AWS credentials.
"""

import json
import os

_PLATFORMS = {
    "deepseek": "DEEPSEEK",
    "openai": "OPENAI",
    "anthropic": "ANTHROPIC",
    "gemini": "GEMINI",
    "vllm": "VLLM",
    "ollama": "OLLAMA",
    "sglang": "SGLANG",
    "bedrock": "AWS_BEDROCK",
    "bedrock-converse": "AWS_BEDROCK_CONVERSE",
    "openai-compatible": "OPENAI_COMPATIBLE_MODEL",
}

_DEFAULT_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY",
    "bedrock": "BEDROCK_API_KEY",
}


def make_model(spec: dict):
    """Spec dict -> CAMEL BaseModelBackend."""
    alias = spec["platform"].lower()
    if alias not in _PLATFORMS:
        raise ValueError(f"Unknown platform {alias!r}; "
                         f"known: {sorted(_PLATFORMS)}")
    from camel.models import ModelFactory          # lazy: keeps config
    from camel.types import ModelPlatformType      # logic importable
                                                   # without camel installed
    kwargs = {
        "model_platform": getattr(ModelPlatformType, _PLATFORMS[alias]),
        "model_type": spec["model"],
    }
    if spec.get("url"):
        kwargs["url"] = spec["url"]
    key_env = spec.get("api_key_env") or _DEFAULT_KEY_ENV.get(alias)
    if key_env and os.environ.get(key_env):
        kwargs["api_key"] = os.environ[key_env]
    return ModelFactory.create(**kwargs)


def load_model_config(path: str) -> list:
    """JSON (or YAML, if pyyaml is installed) -> list of spec dicts."""
    text = open(path).read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    return cfg["models"] if isinstance(cfg, dict) else cfg


def expand_specs(specs: list, role: str = None) -> list:
    """Filter by role (specs without a role, or role='all', match every
    role) and expand by count for weighted mixing."""
    out = []
    for s in specs:
        s_role = s.get("role", "all")
        if role is not None and s_role not in ("all", role):
            continue
        out.extend([s] * int(s.get("count", 1)))
    return out


def make_models_from_config(path_or_specs, role: str = None):
    """-> single backend or list (OASIS ModelManager round-robins lists)."""
    specs = (load_model_config(path_or_specs)
             if isinstance(path_or_specs, str) else path_or_specs)
    expanded = expand_specs(specs, role)
    if not expanded:
        raise ValueError(f"No model specs match role={role!r}")
    backends = [make_model(s) for s in expanded]
    return backends[0] if len(backends) == 1 else backends


def has_role_specs(specs: list) -> bool:
    return any(s.get("role", "all") != "all" for s in specs)


# ---------------------------------------------------------------------------
# Backward-compatible simple constructors (used by CLI flags)
# ---------------------------------------------------------------------------

def make_models(backend: str = "deepseek", model_name: str = None,
                urls: list = None, **_):
    if backend == "deepseek":
        return make_model({"platform": "deepseek",
                           "model": model_name or "deepseek-chat"})
    if backend == "openai":
        return make_model({"platform": "openai",
                           "model": model_name or "gpt-4o-mini"})
    if backend == "vllm":
        specs = [{"platform": "vllm",
                  "model": model_name or "Qwen/Qwen2.5-7B-Instruct",
                  "url": u} for u in (urls or ["http://localhost:8000/v1"])]
        return make_models_from_config(specs)
    raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# Per-role agent graph (advisory): mirror of OASIS's
# generate_reddit_agent_graph with a model chosen per profile.
# ---------------------------------------------------------------------------

async def generate_graph_with_role_models(profile_path: str,
                                          model_for, available_actions):
    """``model_for(profile: dict, index: int) -> backend-or-list``.
    Mirrors oasis.social_agent.agents_generator.generate_reddit_agent_graph
    (verified against upstream) but assigns models per agent."""
    import asyncio
    from oasis.social_agent import AgentGraph, SocialAgent
    from oasis.social_platform.config import UserInfo

    agent_info = json.loads(open(profile_path).read())
    agent_graph = AgentGraph()

    async def process(i):
        info = agent_info[i]
        profile = {"nodes": [], "edges": [], "other_info": {
            "user_profile": info["persona"], "mbti": info["mbti"],
            "gender": info["gender"], "age": info["age"],
            "country": info["country"]}}
        user_info = UserInfo(name=info["username"], description=info["bio"],
                             profile=profile, recsys_type="reddit")
        agent_graph.add_agent(SocialAgent(
            agent_id=i, user_info=user_info, agent_graph=agent_graph,
            model=model_for(info, i),
            available_actions=available_actions))

    await asyncio.gather(*(process(i) for i in range(len(agent_info))))
    return agent_graph
