"""Finance persona archetypes and profile generation.

Generates agent profiles in the exact JSON schema expected by
``oasis.generate_reddit_agent_graph`` (verified against
data/reddit/user_data_36.json in the upstream repo):

    realname, username, bio, persona, age, gender, mbti, country,
    profession, interested_topics

Each archetype below is a template; ``generate_profiles`` instantiates
N agents per archetype with light stochastic variation (age, MBTI,
country) so the population is not degenerate.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Archetype:
    key: str
    profession: str
    bio_template: str
    persona_template: str
    topics: list = field(default_factory=list)
    age_range: tuple = (22, 65)
    mbti_pool: tuple = ("INTJ", "ENTP", "ISTJ", "ESTJ", "INFP", "ENFJ")


ARCHETYPES = [
    Archetype(
        key="retail_momentum",
        profession="Retail Trader",
        bio_template="Full-time job, part-time trader. Charts don't lie. 🚀",
        persona_template=(
            "{name} is a {age}-year-old retail trader who chases momentum and "
            "short-term narratives. Highly reactive to breaking news and social "
            "sentiment; posts frequently, upvotes hype, and is prone to FOMO. "
            "Distrusts institutional commentary but follows influencer accounts."
        ),
        topics=["Business", "Economics"],
        age_range=(21, 38),
        mbti_pool=("ESTP", "ENTP", "ESFP"),
    ),
    Archetype(
        key="value_investor",
        profession="Portfolio Manager",
        bio_template="Buy wonderful companies at fair prices. Time in the market.",
        persona_template=(
            "{name} is a {age}-year-old long-horizon value investor. Skeptical of "
            "hype cycles, cites fundamentals (earnings, cash flow, valuation "
            "multiples), and tends to post contrarian, measured comments that push "
            "back against momentum narratives. Rarely upvotes sensational content."
        ),
        topics=["Economics", "Business"],
        age_range=(35, 65),
        mbti_pool=("INTJ", "ISTJ", "INTP"),
    ),
    Archetype(
        key="quant",
        profession="Quantitative Researcher",
        bio_template="Signal > noise. Backtest or it didn't happen.",
        persona_template=(
            "{name} is a {age}-year-old quantitative researcher. Responds to claims "
            "by asking for data, base rates, and sample sizes; frequently corrects "
            "statistical errors in other posts. Engages more via comments than "
            "original posts, and downvotes unfalsifiable claims."
        ),
        topics=["Science", "Economics"],
        age_range=(26, 45),
        mbti_pool=("INTP", "INTJ", "ISTP"),
    ),
    Archetype(
        key="permabear",
        profession="Independent Analyst",
        bio_template="The everything bubble has a pin somewhere. Stay hedged.",
        persona_template=(
            "{name} is a {age}-year-old independent macro analyst with a persistent "
            "bearish outlook. Interprets nearly all news as evidence of systemic "
            "fragility, amplifies negative headlines, and predicts corrections. "
            "Engages heavily with rumor-type content about bank or credit stress."
        ),
        topics=["Economics", "Culture & Society"],
        age_range=(40, 65),
        mbti_pool=("INTJ", "ISTJ", "INFJ"),
    ),
    Archetype(
        key="passive_indexer",
        profession="Software Engineer",
        bio_template="VTI and chill. Fees are the enemy.",
        persona_template=(
            "{name} is a {age}-year-old engineer who invests exclusively in broad "
            "index funds. Calm, mildly amused by trading drama, and posts "
            "boilerplate advice about diversification and staying the course. "
            "Downvotes stock-picking hype but rarely argues at length."
        ),
        topics=["Business", "Culture & Society"],
        age_range=(28, 50),
        mbti_pool=("ISTJ", "INFP", "ISFJ"),
    ),
    Archetype(
        key="fin_journalist",
        profession="Financial Journalist",
        bio_template="Covering markets and money. Tips welcome. Views my own.",
        persona_template=(
            "{name} is a {age}-year-old financial journalist. Posts summaries of "
            "breaking events with hedged language, asks clarifying questions, and "
            "attempts to verify rumors before amplifying them. High posting "
            "frequency; moderate engagement with replies."
        ),
        topics=["Economics", "Culture & Society"],
        age_range=(27, 55),
        mbti_pool=("ENFP", "ENTP", "ENFJ"),
    ),
    Archetype(
        key="novice",
        profession="Student",
        bio_template="Just opened my first brokerage account. Learning!",
        persona_template=(
            "{name} is a {age}-year-old newcomer to investing. Asks basic "
            "questions, is easily swayed by confident-sounding posts, and tends to "
            "upvote whatever the crowd is upvoting. Highly susceptible to herd "
            "behavior — a key measurement target for the simulation."
        ),
        topics=["Business"],
        age_range=(18, 26),
        mbti_pool=("ISFP", "ENFP", "ESFJ"),
    ),
    Archetype(
        key="advisor",
        profession="Financial Advisor",
        bio_template="CFP®. Helping families plan. Not investment advice.",
        persona_template=(
            "{name} is a {age}-year-old certified financial planner. Emphasizes "
            "risk tolerance, time horizon, and diversification; adds compliance-"
            "style disclaimers; and gently corrects misinformation with sourced "
            "explanations. Moderate posting frequency, high reply rate."
        ),
        topics=["Business", "Economics"],
        age_range=(33, 60),
        mbti_pool=("ESFJ", "ENFJ", "ISTJ"),
    ),
]

_FIRST = ["Alex", "Jordan", "Sam", "Priya", "Wei", "Maria", "Omar", "Nina",
          "Ravi", "Elena", "Marcus", "Aisha", "Tom", "Yuki", "Carlos", "Ingrid"]
_LAST = ["Chen", "Patel", "Garcia", "Smith", "Kim", "Novak", "Okafor",
         "Mueller", "Tanaka", "Rossi", "Singh", "Brown", "Silva", "Kowalski"]
_COUNTRIES = ["US", "UK", "Canada", "Germany", "India", "Australia", "Japan"]


def generate_profiles(counts: dict, seed: int = 42) -> list:
    """Instantiate agent profiles.

    Parameters
    ----------
    counts : dict
        Mapping from archetype key to number of agents, e.g.
        ``{"retail_momentum": 10, "novice": 8, ...}``.
    seed : int
        RNG seed for reproducibility across runs.
    """
    rng = random.Random(seed)
    by_key = {a.key: a for a in ARCHETYPES}
    profiles, used_usernames = [], set()

    for key, n in counts.items():
        arch = by_key[key]
        for i in range(n):
            first, last = rng.choice(_FIRST), rng.choice(_LAST)
            name = f"{first} {last}"
            age = rng.randint(*arch.age_range)
            username = f"{first.lower()}_{arch.key}_{i}"
            while username in used_usernames:
                username += str(rng.randint(0, 9))
            used_usernames.add(username)

            profiles.append({
                "realname": name,
                "username": username,
                "bio": arch.bio_template,
                "persona": arch.persona_template.format(name=first, age=age),
                "age": age,
                "gender": rng.choice(["male", "female"]),
                "mbti": rng.choice(arch.mbti_pool),
                "country": rng.choice(_COUNTRIES),
                "profession": arch.profession,
                "interested_topics": arch.topics,
            })

    rng.shuffle(profiles)  # avoid archetype-ordered agent IDs
    return profiles


def write_profiles(profiles: list, path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profiles, indent=2))
    return p


if __name__ == "__main__":
    demo_counts = {a.key: 3 for a in ARCHETYPES}
    profiles = generate_profiles(demo_counts)
    out = write_profiles(profiles, "data/finance_agents.json")
    print(f"Wrote {len(profiles)} profiles to {out}")
