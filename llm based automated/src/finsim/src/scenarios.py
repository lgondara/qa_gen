"""Scenario definitions for finance simulations.

A scenario is a schedule of exogenous events injected into the platform at
specific timesteps via ``ManualAction``, plus metadata (population mix,
number of steps). Everything between injections is endogenous: agents act
through ``LLMAction`` and the platform's recommendation system.

Design note: injections are attributed to a designated "event source" agent
(agent 0 by convention) so that organic agent behavior is not contaminated —
i.e., the treatment is delivered by a fixed node, and propagation/response
is measured over the rest of the population.
"""

from dataclasses import dataclass, field


@dataclass
class Event:
    step: int              # timestep at which to inject
    content: str           # post content
    kind: str = "post"     # "post" only for now; extend to comments if needed


@dataclass
class Scenario:
    name: str
    description: str
    num_steps: int
    population: dict                      # archetype key -> count
    events: list = field(default_factory=list)
    seed: int = 42


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------

RATE_DECISION = Scenario(
    name="rate_decision",
    description=(
        "Central bank surprises markets with a 50bp cut. Measures sentiment "
        "divergence across archetypes and comment-thread polarization."
    ),
    num_steps=8,
    population={
        "retail_momentum": 6, "value_investor": 4, "quant": 3,
        "permabear": 3, "passive_indexer": 4, "fin_journalist": 2,
        "novice": 5, "advisor": 3,
    },
    events=[
        Event(step=0, content=(
            "BREAKING: The central bank has cut its policy rate by 50 basis "
            "points, twice the expected 25bp. Statement cites 'softening labor "
            "market conditions.' Press conference in one hour."
        )),
        Event(step=4, content=(
            "Press conference update: the chair declined to commit to further "
            "cuts, calling this a 'recalibration, not the start of an easing "
            "cycle.' Futures paring initial gains."
        )),
    ],
)

RUMOR_PROPAGATION = Scenario(
    name="bank_rumor",
    description=(
        "An unverified rumor about a mid-size bank's solvency is seeded. "
        "Measures rumor propagation (reposts/comments referencing it), who "
        "amplifies vs. debunks, and whether corrections travel as far as the "
        "rumor — a classic misinformation-dynamics design."
    ),
    num_steps=10,
    population={
        "retail_momentum": 7, "permabear": 4, "quant": 3,
        "fin_journalist": 3, "novice": 6, "value_investor": 3,
        "advisor": 2, "passive_indexer": 2,
    },
    events=[
        Event(step=0, content=(
            "Hearing from two separate sources that Meridian Regional Bank "
            "(fictional) is facing large uninsured deposit outflows this week. "
            "Nothing official yet. Anyone else seeing this?"
        )),
        Event(step=6, content=(
            "UPDATE: Meridian Regional Bank has issued a statement: deposit "
            "levels are stable and the bank exceeds all regulatory capital "
            "requirements. Regulator confirms no supervisory action underway."
        )),
    ],
)

HERD_BEHAVIOR = Scenario(
    name="meme_stock",
    description=(
        "A speculative small-cap narrative is seeded and the simulation "
        "measures herding: do upvotes/comments snowball, and do skeptical "
        "archetypes moderate or get drowned out? Mirrors the herd-effect "
        "replication in the OASIS paper, in a finance register."
    ),
    num_steps=10,
    population={
        "retail_momentum": 10, "novice": 8, "quant": 3,
        "value_investor": 3, "passive_indexer": 3, "permabear": 2,
        "fin_journalist": 2, "advisor": 2,
    },
    events=[
        Event(step=0, content=(
            "$ZVLT (fictional) short interest just hit 140% of float. Options "
            "chain is loading up on next month's calls. This has all the "
            "ingredients. Not financial advice. 🚀🚀"
        )),
    ],
)

SCENARIOS = {s.name: s for s in [RATE_DECISION, RUMOR_PROPAGATION, HERD_BEHAVIOR]}
