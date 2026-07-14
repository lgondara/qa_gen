"""Simulation runner.

Flow per run:
  1. Generate persona profiles for the scenario's population mix.
  2. Build the agent graph (Reddit platform: posts, comments, up/downvotes).
  3. For each timestep: inject any scheduled events via the event-source
     agent (agent 0), then let every other agent take an LLMAction.
  4. All state lands in a SQLite DB, which analysis.py consumes.

Convention: agent 0 is the "event source" (a journalist-style account used
only for injections). It never takes LLM actions, so exogenous treatment and
endogenous response stay cleanly separated.
"""

import os
from pathlib import Path

import oasis
from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph

from .personas import generate_profiles, write_profiles
from .scenarios import Scenario

AVAILABLE_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_POST,        # upvote
    ActionType.DISLIKE_POST,     # downvote
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.REFRESH,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
]

EVENT_SOURCE_PROFILE = {
    "realname": "Market Wire",
    "username": "market_wire_official",
    "bio": "Real-time markets newswire. (Simulation event source.)",
    "persona": (
        "An automated newswire account that only posts market events. "
        "It does not comment, vote, or reply."
    ),
    "age": 30,
    "gender": "male",
    "mbti": "ISTJ",
    "country": "US",
    "profession": "Financial Journalist",
    "interested_topics": ["Economics", "Business"],
}


async def run_scenario(
    scenario: Scenario,
    model,
    data_dir: str = "data",
    results_dir: str = "results",
) -> str:
    """Run one scenario end-to-end. Returns the path to the SQLite DB."""
    # --- 1. profiles -------------------------------------------------------
    profiles = generate_profiles(scenario.population, seed=scenario.seed)
    profiles.insert(0, EVENT_SOURCE_PROFILE)  # agent 0 = event source
    profile_path = write_profiles(
        profiles, f"{data_dir}/{scenario.name}_agents.json"
    )
    print(f"[{scenario.name}] {len(profiles)} agents "
          f"({len(profiles) - 1} organic + 1 event source)")

    # --- 2. environment ----------------------------------------------------
    agent_graph = await generate_reddit_agent_graph(
        profile_path=str(profile_path),
        model=model,
        available_actions=AVAILABLE_ACTIONS,
    )

    db_path = f"{results_dir}/{scenario.name}.db"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )
    await env.reset()

    events_by_step = {}
    for ev in scenario.events:
        events_by_step.setdefault(ev.step, []).append(ev)

    source_agent = env.agent_graph.get_agent(0)

    # --- 3. main loop ------------------------------------------------------
    for step in range(scenario.num_steps):
        # 3a. exogenous injections
        if step in events_by_step:
            injections = {
                source_agent: [
                    ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": ev.content},
                    )
                    for ev in events_by_step[step]
                ]
            }
            await env.step(injections)
            for ev in events_by_step[step]:
                print(f"[{scenario.name}] step {step}: injected event "
                      f"({ev.content[:60]}...)")

        # 3b. endogenous behavior — every organic agent acts
        llm_actions = {
            agent: LLMAction()
            for agent_id, agent in env.agent_graph.get_agents()
            if agent_id != 0
        }
        await env.step(llm_actions)
        print(f"[{scenario.name}] step {step + 1}/{scenario.num_steps} done")

    await env.close()
    print(f"[{scenario.name}] finished — DB at {db_path}")
    return db_path
