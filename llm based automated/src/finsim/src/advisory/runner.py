"""Advisory simulation runner.

Timeline per case:
  t=0   archive seeding (prior-run case summaries), FOLLOW network
        injection, and the asking client posts the question
  t=1..advisor_only_steps   advisors only (initial answers; personas
        instruct them to search prior cases first)
  t=advisor_only_steps+1..  open deliberation: all agents (advisors
        challenge/endorse, peers share experience, votes accumulate),
        with any scheduled context events injected
  end   verdict interview of the asking client (if the installed OASIS
        supports the INTERVIEW action)

Agent index layout: 0 = event source (market context + archive),
1..n_adv = advisors, then clients — see profiles.load_population.
"""

import os
from pathlib import Path

import oasis
from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph

from ..simulation import AVAILABLE_ACTIONS, EVENT_SOURCE_PROFILE
from .archive import extract_case_summaries
from .cases import AdvisoryCase
from .network import amend_personas, build_network, follow_edges
from .profiles import ensure_sample_csvs, load_population, write_profiles_json


def _find_asker(profiles: list, asker: str) -> int:
    for i, p in enumerate(profiles):
        if p["_role"] == "client" and (
                p["username"].startswith(asker) or asker in p["realname"]):
            return i
    raise ValueError(f"No client matches asker={asker!r}")


async def run_case(case: AdvisoryCase, model,
                   clients_csv: str = None, advisors_csv: str = None,
                   history_dbs: list = None, role_models: dict = None,
                   data_dir: str = "data", results_dir: str = "results",
                   max_connections: int = 50, seed: int = 42) -> str:
    if not clients_csv or not advisors_csv:
        clients_csv, advisors_csv = ensure_sample_csvs(data_dir)

    profiles = load_population(clients_csv, advisors_csv, seed=seed)
    net = build_network(profiles, max_connections=max_connections, seed=seed)
    profiles = amend_personas(profiles, net)
    asker_idx = _find_asker(profiles, case.asker)

    # event source at index 0; advisory profiles shift by 1
    full = [dict(EVENT_SOURCE_PROFILE)] + profiles
    asker_agent = asker_idx + 1
    profile_path = write_profiles_json(
        full, f"{data_dir}/{case.name}_agents.json")
    n_adv = sum(1 for p in profiles if p["_role"] == "advisor")
    advisor_agents = set(range(1, 1 + n_adv))
    print(f"[{case.name}] {n_adv} advisors, {len(profiles)-n_adv} clients; "
          f"asker agent {asker_agent} ({profiles[asker_idx]['realname']})")

    if role_models:
        from ..models import generate_graph_with_role_models

        def model_for(info, i):
            if info["username"].startswith("adv_"):
                return role_models["advisor"]
            if info["username"].startswith("cli_"):
                return role_models["client"]
            return role_models["all"]      # event source, archive
        agent_graph = await generate_graph_with_role_models(
            str(profile_path), model_for, AVAILABLE_ACTIONS)
        print(f"[{case.name}] role-based models: advisors/clients "
              f"on separate backends")
    else:
        agent_graph = await generate_reddit_agent_graph(
            profile_path=str(profile_path), model=model,
            available_actions=AVAILABLE_ACTIONS)

    db_path = f"{results_dir}/{case.name}.db"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    env = oasis.make(agent_graph=agent_graph,
                     platform=oasis.DefaultPlatformType.REDDIT,
                     database_path=db_path)
    await env.reset()

    A = env.agent_graph.get_agent  # by agent id
    source, asker = A(0), A(asker_agent)

    # --- t=0: archive, follows, question --------------------------------
    t0 = {}
    archive_posts = extract_case_summaries(history_dbs or [])
    if archive_posts:
        t0[source] = [ManualAction(action_type=ActionType.CREATE_POST,
                                   action_args={"content": s})
                      for s in archive_posts]
        print(f"[{case.name}] seeded {len(archive_posts)} archived cases")
    if t0:
        await env.step(t0)

    try:  # best-effort platform follow graph (channel 2)
        # follow_edges yields (follower_profile_idx, followee_user_id).
        # Profiles shift +1 for the event source (agent = idx + 1), and
        # user_id = agent_id + 1 under the standard mapping, so offset=2.
        edges = follow_edges(net, offset=2)
        follows = {}
        for follower, followee_uid in edges:
            ag = A(follower + 1)
            follows.setdefault(ag, []).append(ManualAction(
                action_type=ActionType.FOLLOW,
                action_args={"followee_id": followee_uid}))
        await env.step(follows)
        print(f"[{case.name}] injected {len(edges)} follow edges")
    except Exception as e:  # follows are best-effort; personas carry the net
        print(f"[{case.name}] follow injection skipped: {e}")

    await env.step({asker: [ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": case.question})]})
    print(f"[{case.name}] question posted by {profiles[asker_idx]['realname']}")

    # --- deliberation ----------------------------------------------------
    events_by_step = {}
    for ev in case.context_events:
        events_by_step.setdefault(ev.step, []).append(ev)

    for step in range(case.num_steps):
        if step in events_by_step:
            await env.step({source: [ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": ev.content})
                for ev in events_by_step[step]]})
            print(f"[{case.name}] step {step}: context event injected")

        if step < case.advisor_only_steps:
            actors = {A(i): LLMAction() for i in advisor_agents}
        else:
            actors = {agent: LLMAction()
                      for aid, agent in env.agent_graph.get_agents()
                      if aid != 0}
        await env.step(actors)
        phase = "advisors" if step < case.advisor_only_steps else "open"
        print(f"[{case.name}] step {step+1}/{case.num_steps} ({phase})")

    # --- verdict interview ------------------------------------------------
    if hasattr(ActionType, "INTERVIEW"):
        try:
            await env.step({asker: [ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": case.interview_prompt})]})
            print(f"[{case.name}] verdict interview recorded (see trace)")
        except Exception as e:
            print(f"[{case.name}] interview skipped: {e}")

    await env.close()
    print(f"[{case.name}] finished — DB at {db_path}")
    return db_path
