"""Advisory network construction.

Structure:
  * advisor <-> client: each client is assigned one primary advisor;
    assignment probability is proportional to advisor experience (senior
    advisors carry larger books). One advisor serves many clients.
  * client <-> client: a small-world graph (ring lattice + random
    shortcuts), hard-capped at ``max_connections`` per client (default 50).

The network is delivered through two channels:
  1. Personas — each agent's persona is amended with its connections by
     name. This is the reliable channel: it directly conditions the LLM's
     behavior ("your advisor is...", "your clients include...").
  2. Platform follows — best-effort FOLLOW actions injected at t=0 so the
     platform's social graph matches. This assumes the standard OASIS
     user_id = agent_id + 1 mapping; verify against your DB's user table
     if follows matter for your recommender configuration.
"""

import random


def build_network(profiles: list, max_connections: int = 50,
                  ring_k: int = 2, shortcut_p: float = 0.15,
                  seed: int = 42) -> dict:
    """Returns {"advisor_of": {client_idx: advisor_idx},
                "clients_of": {advisor_idx: [client_idx,...]},
                "peers": {client_idx: [client_idx,...]}}
    where indices are positions in ``profiles``."""
    rng = random.Random(seed)
    advisors = [i for i, p in enumerate(profiles) if p["_role"] == "advisor"]
    clients = [i for i, p in enumerate(profiles) if p["_role"] == "client"]

    # experience-weighted advisor assignment
    weights = []
    for i in advisors:
        exp = "".join(c for c in profiles[i]["bio"].split("·")[1]
                      if c.isdigit()) or "5"
        weights.append(int(exp))
    advisor_of, clients_of = {}, {i: [] for i in advisors}
    for c in clients:
        a = rng.choices(advisors, weights=weights, k=1)[0]
        advisor_of[c] = a
        clients_of[a].append(c)

    # client-client small world: ring lattice + shortcuts, capped
    peers = {c: set() for c in clients}
    n = len(clients)
    for idx, c in enumerate(clients):
        for k in range(1, ring_k + 1):
            peers[c].add(clients[(idx + k) % n])
            peers[c].add(clients[(idx - k) % n])
    for c in clients:
        if rng.random() < shortcut_p:
            other = rng.choice([x for x in clients if x != c])
            peers[c].add(other)
    for c in clients:
        peers[c].discard(c)
        if len(peers[c]) > max_connections:
            peers[c] = set(rng.sample(sorted(peers[c]), max_connections))
    # symmetrize under the cap
    for c in clients:
        for p in list(peers[c]):
            if len(peers[p]) < max_connections:
                peers[p].add(c)

    return {"advisor_of": advisor_of, "clients_of": clients_of,
            "peers": {c: sorted(s) for c, s in peers.items()}}


def amend_personas(profiles: list, net: dict, show_max: int = 8) -> list:
    """Append connection context to each persona (channel 1)."""
    name = lambda i: profiles[i]["realname"]
    for c, a in net["advisor_of"].items():
        peer_names = ", ".join(name(p) for p in net["peers"][c][:show_max])
        profiles[c]["persona"] += (
            f" Your financial advisor on this platform is {name(a)}. "
            f"You also know these fellow clients: {peer_names}."
        )
    for a, cs in net["clients_of"].items():
        if cs:
            client_names = ", ".join(name(c) for c in cs[:show_max])
            more = f" and {len(cs)-show_max} others" if len(cs) > show_max else ""
            profiles[a]["persona"] += (
                f" Your clients on this platform include {client_names}{more}. "
                f"You feel particular responsibility for questions they ask, "
                f"but you engage with any client's question professionally."
            )
    return profiles


def follow_edges(net: dict, offset: int = 1) -> list:
    """(follower_agent_id, followee_user_id) pairs for FOLLOW injection
    (channel 2). ``offset`` converts agent index -> platform user_id under
    the standard OASIS mapping user_id = agent_id + 1. Agent indices here
    assume the event source occupies index 0 and profiles begin at 1 —
    the runner passes indices already shifted."""
    edges = []
    for c, a in net["advisor_of"].items():
        edges.append((c, a + offset))
        edges.append((a, c + offset))
    for c, ps in net["peers"].items():
        for p in ps:
            edges.append((c, p + offset))
    return edges
