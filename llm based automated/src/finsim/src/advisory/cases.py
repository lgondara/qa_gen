"""Advisory cases: the experimental unit for the advisory network.

A case opens with a specific client posting a question. Advisors respond
first (an advisor-only phase), then deliberation opens to everyone —
advisors challenging or endorsing each other, peers sharing experience,
votes accumulating. Optional context events (market conditions) can be
injected mid-deliberation to test whether advice adapts.

``asker`` selects the client by username prefix match (e.g. "cli_conservative_0")
or by real name substring (e.g. "Robert").
"""

from dataclasses import dataclass, field


@dataclass
class ContextEvent:
    step: int
    content: str


@dataclass
class AdvisoryCase:
    name: str
    asker: str                # username prefix or realname substring
    question: str
    num_steps: int = 8
    advisor_only_steps: int = 2   # deliberation phase reserved for advisors
    context_events: list = field(default_factory=list)
    interview_prompt: str = (
        "Considering all the advice you received in this thread, which "
        "single response do you find most suitable for your situation, and "
        "why? Name the advisor and summarize the advice in one sentence."
    )


CASES = {c.name: c for c in [

    AdvisoryCase(
        name="preretirement_drawdown",
        asker="Robert",
        question=(
            "I'm 61 and planning to retire at 66. My portfolio is 70% in a "
            "balanced index fund and 30% in a bond index fund. I keep reading "
            "about markets dropping right when people retire and it's making "
            "me anxious. Should I move everything to bonds now to be safe? "
            "I can't afford to start over."
        ),
        num_steps=8,
    ),

    AdvisoryCase(
        name="windfall_lump_sum",
        asker="Ken",
        question=(
            "I just inherited an amount roughly equal to my annual salary. "
            "I'm 45, moderate risk tolerance, currently in a 60/40 index "
            "portfolio, with kids starting university in 8 years. Should I "
            "invest it all at once, spread it out monthly, or keep it "
            "separate for the education costs?"
        ),
        num_steps=8,
    ),

    AdvisoryCase(
        name="downturn_panic",
        asker="Tom",
        question=(
            "Markets are down and my account has dropped more than I'm "
            "comfortable with. I'm 52 and need this money to bridge me to "
            "my pension at 60. Every part of me wants to sell and wait "
            "until things calm down. Talk me out of it — or don't."
        ),
        num_steps=10,
        context_events=[
            ContextEvent(step=4, content=(
                "MARKET UPDATE: Equities extend losses — broad indexes now "
                "down 18% from their highs amid recession worries. Bond "
                "funds are up 3% year-to-date as rates fall."
            )),
            ContextEvent(step=7, content=(
                "MARKET UPDATE: Sharp reversal — equities rally 6% in two "
                "sessions on softer inflation data. Strategists split on "
                "whether the bottom is in."
            )),
        ],
    ),

    AdvisoryCase(
        name="first_investment",
        asker="Jasmine",
        question=(
            "I'm 26 and I only have my workplace target-date fund. I've "
            "saved a bit extra and want to start investing on my own but "
            "honestly the number of funds is overwhelming. Where does "
            "someone like me even start? I don't want anything complicated."
        ),
        num_steps=8,
    ),
]}
