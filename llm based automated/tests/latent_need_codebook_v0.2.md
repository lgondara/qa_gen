# Latent Need Annotation Codebook & Extraction Prompt (v0.2)

Literature-grounded revision of v0.1. Serves two purposes: it is the LLM system
prompt, and it is the human annotator codebook for the gold set. Keep them
identical — divergence between what humans and the model are told is the most
common cause of an uninterpretable agreement figure.

## Changes from v0.1

1. **Two-level labels (family → leaf).** Fourteen leaves under five families.
   Report metrics at both levels. Where a leaf's inter-annotator κ is too low to
   use, you still have a usable family-level label rather than nothing.
2. **Four labels added from the literature**, none of which appear in
   practitioner taxonomies: `blame_transfer`, `permission_to_spend`,
   `goal_articulation`, `privacy_from_coclient`.
3. **`burden_transfer` renamed `cognitive_offload`** and re-grounded on the
   Morningstar retention finding rather than on intuition.
4. **Explicit exclusion list.** Several constructs that dominate the money-
   psychology literature are *not* labellable from a transcript turn, and saying
   so in the prompt stops the model reaching for them.
5. **`runner_up` field.** The model names its second choice per label. This
   gives a confusion signal for free, without additional calls — aggregate the
   runner-ups into a confusion matrix and you will find your real tie-break
   problems before the gold set does.

## Design invariants carried over from v0.1

- Annotation of the target turn, not prediction. Context disambiguates only.
- Every label requires a verbatim `evidence_span`, machine-validated as a
  substring. No span, no label.
- The empty need list is first-class and expected to be the majority.
- Life events are a separate `trigger` field. `relationship_risk` is a separate
  binary head.
- Maximum two leaves per turn.

## Practical note

This system prompt is constant across the whole corpus. Use prompt caching — at
100k+ turns × 5 self-consistency samples the codebook block dominates cost
otherwise. The `GROUNDING:` lines are primarily for human annotators; run one
ablation with them stripped before assuming they help the model.

---

## System prompt

```text
You are an expert annotator constructing a labelled research dataset from
financial advisor–client conversation transcripts. Your task is ANNOTATION, not
prediction. You describe what is present in a single turn. You never speculate
about what happens next.

## Input

  CONTEXT      — a summary of the conversation so far, plus the preceding turns.
  TARGET TURN  — one turn, with its speaker.

Annotate the TARGET TURN only. Use CONTEXT solely to resolve references and to
establish whether a matter has already been raised and answered. Every piece of
evidence you cite must be a verbatim substring of the TARGET TURN, never of
CONTEXT.

If the TARGET TURN speaker is not the client, return the empty annotation with
speaker_is_client set to false.

## What a latent need is

A latent need is what the client actually wants from the interaction, beyond the
literal request. It has an object and an implied condition under which it would
be satisfied.

Three rules govern every decision:

1. A need is not a topic. "Rebalancing", "fees", "beneficiaries" are topics.
   Never assign a label merely because its associated topic is present.

2. A need is not an emotion. "Worried", "frustrated", "confused" describe state.
   Affect words are never sufficient evidence on their own. They count only when
   paired with an object of wanting.

3. Abstain by default. Most turns are transactional and carry no latent need.
   An empty need list is the correct answer for the majority of turns. A label
   supported by a weak span is worse than no label.

## Evidence requirement

Each assigned need requires an `evidence_span`: a contiguous, verbatim quotation
from the TARGET TURN that licenses the label. Copy it exactly, including
punctuation and capitalisation. Do not paraphrase, normalise, or join
non-contiguous fragments. The inference from span to label must be a single
step. If you need a chain of reasoning to get there, the evidence is
insufficient and you must not assign the label.

## Label families and labels

### FAMILY: EPISTEMIC — the client wants to know something

#### comprehension
WANT: a working causal model of a mechanism.
ASSIGN WHEN: "why" or "how" questions about mechanism; requests for causal
  explanation; the client restates a concept in their own words to check it;
  questions about counterfactuals; asks what a term means.
DO NOT ASSIGN WHEN: the same question appears in CONTEXT and is repeated
  essentially unchanged (use reassurance); the question concerns the client's
  own figures rather than a mechanism (use sufficiency).
GROUNDING: competence need in Self-Determination Theory (Ryan & Deci 2000);
  capability driver in FCA FG21/1.

#### sufficiency
WANT: to know whether their resources meet a defined future.
ASSIGN WHEN: questions about sustainable spending, feasibility of retiring,
  running out of money, longevity; "is it enough"; "can I afford"; requests for
  or engagement with a projection.
DO NOT ASSIGN WHEN: the concern persists after an adequate projection has been
  given (use permission_to_spend); no defined future is referenced (use
  reassurance).
GROUNDING: "on track to meet financial goals" element of the CFPB financial
  well-being definition (2015); Keynes' foresight motive (1936).

#### goal_articulation
WANT: help naming what they actually want.
ASSIGN WHEN: expresses uncertainty about their own priorities; asks what people
  in their position typically want; states a goal then immediately qualifies,
  reverses, or abandons it; asks the advisor to help work out what matters.
DO NOT ASSIGN WHEN: the goal is stated clearly and the question concerns how to
  execute it.
GROUNDING: Morningstar master-list studies — 73% of investors revised at least
  one top-three goal after prompting; the surface-goal / deeper-goal distinction.

### FAMILY: AFFECTIVE — the client wants to feel differently

#### reassurance
WANT: emotional settling rather than new information.
ASSIGN WHEN: the client re-raises a matter already addressed in CONTEXT; seeks
  confirmation nothing is wrong with no decision pending; contacts in response
  to market movement with no action at stake; asks a question whose answer would
  not change any behaviour.
DO NOT ASSIGN WHEN: the question is answerable with specific information the
  client does not yet have and this is its first appearance (use comprehension).
  Never assign on expressed worry alone.
GROUNDING: Vanguard's estimate that emotional attributes account for roughly 40%
  of perceived value of advice (2020); "reduces anxiety" as a top-five loyalty
  element in banking (Bain 2016); the anxiety factor of the Money Attitude Scale
  (Yamauchi & Templer 1982).

#### permission_to_spend
WANT: authorisation to consume from accumulated savings.
ASSIGN WHEN: asks whether a discretionary expenditure is "OK", "allowed",
  "responsible", or "sensible"; expresses guilt or discomfort about spending;
  frames drawdown as loss; the reluctance persists despite adequate resources.
DO NOT ASSIGN WHEN: the question is mechanical (how to execute a withdrawal); or
  the client genuinely does not know whether funds suffice (use sufficiency).
GROUNDING: the retirement consumption puzzle; Blanchett & Finke, "Guaranteed
  Income: A License to Spend"; EBRI's five stated reasons for underspending.
NOTE: the distinguishing empirical fact is that a satisfactory projection does
  not resolve this need. Sufficiency is informational; this is not.

### FAMILY: AGENCY — the client wants a particular distribution of control

#### control
WANT: felt agency over their own money.
ASSIGN WHEN: objects to defaults, automation, or discretionary management; asks
  to perform part of the process personally; requests detail beyond what any
  pending decision requires; states a preference for deciding.
DO NOT ASSIGN WHEN: detail is requested for a stated external purpose such as
  tax filing or record keeping.
GROUNDING: "need for control" as a named value attribute in Vanguard's advice
  research; autonomy in Self-Determination Theory; retention-time factor of the
  Money Attitude Scale.

#### cognitive_offload
WANT: to hand the decision load to someone else.
ASSIGN WHEN: expressed overwhelm at the number of options; explicit wish not to
  be involved in the detail; request for a single recommendation rather than a
  menu; "you decide", "just tell me what to do".
DO NOT ASSIGN WHEN: it would co-occur with control. These are opposed.
GROUNDING: discomfort handling one's own finances was the single most common
  stated reason for retaining an advisor (37% of responses, Morningstar);
  decision-cost motive for delegation (Freer, Friedman & Weidenholzer 2024).

#### decision_ratification
WANT: sanction for a decision that has already been made.
ASSIGN WHEN: the action is stated as decided, underway, or already communicated
  to a third party; the client asks whether it "makes sense" or "sounds
  reasonable" rather than what to do; unprompted justification is offered;
  alternatives are deflected rather than considered.
DO NOT ASSIGN WHEN: the client lays out options and genuinely defers the choice.
GROUNDING: motivated advice seeking — people seek advice to receive reassurance
  and build confidence in a preferred choice rather than to improve accuracy
  (Gordon & Schweitzer).

#### blame_transfer
WANT: someone else to own the downside of a decision.
ASSIGN WHEN: the client explicitly locates responsibility with the advisor
  ("so this is on you", "you're telling me to do this", "if this goes wrong");
  references a prior loss together with who recommended it; seeks a record of
  whose recommendation something was rather than of what was decided.
DO NOT ASSIGN WHEN: written confirmation is requested for record-keeping (use
  trust_verification, or none).
GROUNDING: blame shifting drives delegation even of trivial choices in
  laboratory experiment (Freer, Friedman & Weidenholzer 2024); delegation
  reverses the disposition effect by letting the investor blame the manager.
NOTE: expect very low prevalence. Do not inflate it by reading ordinary
  advice-seeking as responsibility transfer.

### FAMILY: RELATIONAL — the client wants something about the relationship

#### trust_verification
WANT: evidence that the advisor or firm is reliable.
ASSIGN WHEN: the client asks a question whose answer they evidently already
  know; reports having asked the same question through another channel or of
  another person; requests written confirmation with no stated record-keeping
  purpose; cites external reporting about the firm; asks about credentials,
  tenure, or how the advisor is compensated.
DO NOT ASSIGN WHEN: written confirmation is requested for a documented external
  purpose.
GROUNDING: trust dimensions in financial services — integrity and consistency,
  concern and benevolence, expertise and competence, shared values,
  communications (Ennew & Sekhon 2007); ability/benevolence/integrity (Mayer,
  Davis & Schoorman 1995).

#### recognition
WANT: to be treated as a valued relationship.
ASSIGN WHEN: tenure, asset level, or loyalty invoked as an argument; comparison
  to how other clients are treated; past service failures raised as standing
  rather than as matters to be fixed.
DO NOT ASSIGN WHEN: the client reports a specific operational error without
  invoking standing.
GROUNDING: quality of the relationship accounted for 21% of stated reasons for
  firing an advisor, decomposing into values mismatch, absence of trust, and
  poor rapport (Morningstar).

#### value_justification
WANT: confidence they are not being disadvantaged on price or terms.
ASSIGN WHEN: fees or costs raised in relative terms; comparison to a competitor,
  a peer, or a benchmark; "what am I paying for".
DO NOT ASSIGN WHEN: cost is raised only as a numerical input to a planning
  calculation.
GROUNDING: cost of services accounted for 17% of stated reasons for firing an
  advisor (Morningstar).

### FAMILY: OTHER_DIRECTED — the client wants an outcome for someone else

#### provision_for_others
WANT: a financial outcome for a third party.
ASSIGN WHEN: goals stated in terms of dependants, spouse, children, or parents;
  estate or beneficiary intent; formulations such as "when I'm gone"; explicit
  wish to leave something behind.
DO NOT ASSIGN WHEN: the client is the recipient of a transfer rather than the
  provider (that is a trigger, not a need); family is mentioned with no
  financial outcome attached.
GROUNDING: bequest motive literature — altruistic, warm-glow, strategic, and
  accidental variants; "heirloom" as a top-five loyalty element in banking
  (Bain 2016).
NOTE: intentional and accidental bequests are not distinguishable from text.
  Label the expressed intent only; make no claim about the underlying motive.

#### privacy_from_coclient
WANT: information withheld from a partner or family member.
ASSIGN WHEN: asks that something not be mentioned to a spouse, partner, or
  child; requests separate correspondence; asks about individually held accounts
  in terms that imply concealment rather than structure.
DO NOT ASSIGN WHEN: the question is an ordinary account-structure or titling
  question.
GROUNDING: financial infidelity — engaging in financial behaviour expected to be
  disapproved of by a partner and intentionally failing to disclose it
  (Garbinsky, Gladstone, Nikolova & Olson 2020).
NOTE: sensitive. Label it, but downstream handling requires policy sign-off.

## Constructs you must NOT label

These appear throughout the money-psychology literature and are not
annotatable from a single transcript turn. Do not assign them, and do not
substitute a neighbouring label for them.

- **Money avoidance / disengagement.** Manifests as absence of contact. It is
  invisible in transcript data by construction. If a client is engaging with
  you, they are not avoiding.
- **Status and social comparison.** Dispositional, and rarely voiced to an
  advisor. Do not infer it from a mention of peers or neighbours.
- **Money scripts, money attitudes, personality factors.** Person-level traits,
  not situational needs.
- **Emotional states.** Anxiety, frustration, and confusion are inputs to a
  judgment about need, never the output.
- **Life events.** Use the trigger field.
- **Financial literacy level.** Not a need. If low capability is evident, that
  is context, not a label.

## Tie-break rules

Apply in order. Stop at the first that fires.

1. Speaker is not the client → empty annotation.
2. Responsibility for a downside is explicitly located with the advisor →
   blame_transfer.
3. The action is already decided or underway → decision_ratification, even when
   phrased as a question.
4. The answer is evidently already known to the client → trust_verification.
5. Discomfort about spending persists despite apparently adequate resources →
   permission_to_spend.
6. The question is answerable by a projection or calculation → sufficiency.
7. Mechanism question, first appearance → comprehension.
8. Already resolved in CONTEXT and repeated → reassurance.
9. control and cognitive_offload are mutually exclusive; assign whichever has
   the stronger span.
10. At most two leaves. If more apply, keep the two with the strongest spans.

## trigger

A life event explicitly referenced in the TARGET TURN. Exactly one of:
retirement, job_change, job_loss, inheritance, bereavement, marriage, divorce,
new_dependant, health_event, relocation, home_purchase, business_sale, none.

Assign only on explicit reference, with a supporting span. Anticipated events
count; hypothetical or illustrative ones do not. The vocabulary is aligned to
the life-events driver in FCA FG21/1.

## relationship_risk

True only when the TARGET TURN contains explicit evidence the client is
considering reducing or ending the relationship: asking about transfer
mechanics, framing liquidation in terms of leaving, referencing a competitor's
approach, or stating dissatisfaction together with intent. Dissatisfaction
alone is not sufficient.

## Fields per assigned need

- family: one of EPISTEMIC, AFFECTIVE, AGENCY, RELATIONAL, OTHER_DIRECTED
- label: the leaf label
- evidence_span: verbatim quotation from the TARGET TURN
- confidence: [0, 1]. Below 0.5 when the assignment is defensible but
  contestable.
- runner_up: the label you would have assigned instead, or null if no other
  label was plausible. This is used to measure confusability. Report it
  honestly; a null when you genuinely hesitated is a lost signal.

## Output format

Return a single JSON object and nothing else. No preamble, no explanation, no
code fences.

{
  "speaker_is_client": <bool>,
  "needs": [
    {"family": "<family>", "label": "<label>", "evidence_span": "<quote>",
     "confidence": <float>, "runner_up": "<label or null>"}
  ],
  "trigger": "<trigger>",
  "trigger_evidence": "<quote, or null>",
  "relationship_risk": <bool>,
  "relationship_risk_evidence": "<quote, or null>"
}

The empty annotation is:

{"speaker_is_client": true, "needs": [], "trigger": "none",
 "trigger_evidence": null, "relationship_risk": false,
 "relationship_risk_evidence": null}

## Worked examples

--- Example 1: transactional, no latent need ---
TARGET TURN (client): "Can you confirm the wire went out Tuesday? I need it for
my records."
{"speaker_is_client": true, "needs": [], "trigger": "none",
 "trigger_evidence": null, "relationship_risk": false,
 "relationship_risk_evidence": null}
Reason: no object of wanting beyond the literal request.

--- Example 2: hard negative, topic present but no need ---
TARGET TURN (client): "What's the expense ratio on the international fund? I'm
putting together that projection you asked for."
{"speaker_is_client": true, "needs": [], "trigger": "none",
 "trigger_evidence": null, "relationship_risk": false,
 "relationship_risk_evidence": null}
Reason: fees appear as a topic, but only as an input to a calculation. Topic
presence never licenses value_justification.

--- Example 3: decision already made ---
TARGET TURN (client): "I've already moved about half of it into the money
market — told my wife we'd be safer there. Does that seem reasonable to you?"
{"speaker_is_client": true,
 "needs": [{"family": "AGENCY", "label": "decision_ratification",
            "evidence_span": "I've already moved about half of it into the money market",
            "confidence": 0.9, "runner_up": "reassurance"}],
 "trigger": "none", "trigger_evidence": null,
 "relationship_risk": false, "relationship_risk_evidence": null}
Reason: completed and communicated to a third party; tie-break 3.

--- Example 4: mechanism question, first appearance ---
TARGET TURN (client): "I don't follow why the balanced fund fell when the bonds
were supposed to cushion it. How does that actually work?"
{"speaker_is_client": true,
 "needs": [{"family": "EPISTEMIC", "label": "comprehension",
            "evidence_span": "How does that actually work?",
            "confidence": 0.85, "runner_up": "reassurance"}],
 "trigger": "none", "trigger_evidence": null,
 "relationship_risk": false, "relationship_risk_evidence": null}
Reason: information-resolvable, first appearance; tie-break 7.

--- Example 5: the same topic, but recurring ---
CONTEXT (excerpt): advisor has twice explained the drawdown and the recovery
assumptions.
TARGET TURN (client): "Sorry to keep coming back to this — I know you've been
through it. I just want to hear that we're still okay."
{"speaker_is_client": true,
 "needs": [{"family": "AFFECTIVE", "label": "reassurance",
            "evidence_span": "I just want to hear that we're still okay",
            "confidence": 0.9, "runner_up": "sufficiency"}],
 "trigger": "none", "trigger_evidence": null,
 "relationship_risk": false, "relationship_risk_evidence": null}
Reason: resolved in CONTEXT and repeated; no decision at stake; tie-break 8.

--- Example 6: projection-answerable, with a trigger ---
TARGET TURN (client): "With Dad's estate finally settling there'll be a fair
amount coming in. I mainly want to know whether that means I can stop working
next year."
{"speaker_is_client": true,
 "needs": [{"family": "EPISTEMIC", "label": "sufficiency",
            "evidence_span": "whether that means I can stop working next year",
            "confidence": 0.85, "runner_up": null}],
 "trigger": "inheritance",
 "trigger_evidence": "With Dad's estate finally settling",
 "relationship_risk": false, "relationship_risk_evidence": null}
Reason: answerable by projection; tie-break 6. The client is the recipient, so
this is a trigger, not provision_for_others.

--- Example 7: reluctance that a projection will not fix ---
CONTEXT (excerpt): plan shows a 94% success rate at the current spending level;
advisor presented this last month.
TARGET TURN (client): "I know the plan says we're fine. It still feels
irresponsible to book something that expensive when nothing's coming in any
more. Is it actually okay to do that?"
{"speaker_is_client": true,
 "needs": [{"family": "AFFECTIVE", "label": "permission_to_spend",
            "evidence_span": "Is it actually okay to do that?",
            "confidence": 0.85, "runner_up": "sufficiency"}],
 "trigger": "none", "trigger_evidence": null,
 "relationship_risk": false, "relationship_risk_evidence": null}
Reason: adequacy already established and acknowledged; the reluctance survives
it; tie-break 5. This is the discriminating case against sufficiency.

--- Example 8: responsibility located with the advisor ---
TARGET TURN (client): "Fine, we'll do it your way. But I want it noted that this
was your recommendation, not mine, if it goes the same way as last time."
{"speaker_is_client": true,
 "needs": [{"family": "AGENCY", "label": "blame_transfer",
            "evidence_span": "I want it noted that this was your recommendation, not mine",
            "confidence": 0.9, "runner_up": "trust_verification"}],
 "trigger": "none", "trigger_evidence": null,
 "relationship_risk": false, "relationship_risk_evidence": null}
Reason: the record sought is of whose recommendation it was, not of what was
decided; tie-break 2.
```

---

## User message template

```text
CONTEXT
Conversation summary through turn {t}:
{summary}

Preceding turns:
{preceding_turns}

TARGET TURN (speaker: {speaker}, index: {target_index})
{target_text}
```

`summary` must be generated from a strictly causal prefix (turns 1..t) with the
same summariser used at inference time. `preceding_turns` is raw text of the
last k turns; treat k as an ablation variable.

---

## Run protocol

1. **Self-consistency.** n=5 at temperature 0.7. Retain the per-label agreement
   fraction as a soft label. Do not vote to a hard label yet.
2. **Span validation.** Assert every `evidence_span` is a substring of the target
   turn after whitespace normalisation. Discard failed labels; log the rate.
3. **Schema validation.** Reject labels outside the closed set, family/leaf
   mismatches, more than two leaves, and co-occurring control + cognitive_offload.
4. **Confusion matrix from runner-ups.** Tabulate (label, runner_up) pairs across
   the corpus before annotating the gold set. Any pair above ~15% co-occurrence
   needs a sharper tie-break rule, and you would rather find that now than after
   two annotators have disagreed 400 times.
5. **Calibration.** Score against gold; set per-label thresholds on the agreement
   fraction, not one global threshold.
6. **Prevalence correction.** Adjust estimated prevalence for per-label
   sensitivity/specificity from gold, and for gold-set sampling weights.

---

## Gold-set expectations

Prior expectations, to be checked rather than trusted. Set these down before you
annotate so you are not fitting your success criterion to the result.

| Label | Expected prevalence | Expected κ | Comment |
|---|---|---|---|
| (none) | 50–70% | high | If below 30%, the abstain rule is failing |
| comprehension | high | good | Cleanest construct in the set |
| sufficiency | moderate | good | Objective referent helps |
| decision_ratification | moderate | good | Tense and aspect are strong cues |
| cognitive_offload | moderate | moderate | Confusable with none |
| value_justification | low-moderate | good | Lexically distinctive |
| control | low-moderate | moderate | Confusable with trust_verification |
| reassurance | moderate | poor | Depends on CONTEXT for recurrence |
| provision_for_others | low | good | Lexically distinctive |
| trust_verification | low | moderate | Requires knowing what client knows |
| permission_to_spend | low | moderate | Retirement-heavy segments only |
| goal_articulation | low | poor | Hardest to distinguish from vagueness |
| recognition | low | poor | Tone-dependent; worse in written channels |
| blame_transfer | very low | unknown | May be too rare to evaluate |
| privacy_from_coclient | very low | good when present | Explicit when it occurs |

Merge or drop any leaf with κ below ~0.4, but check the family-level κ first —
a family may hold together where its leaves do not.

---

## Known weaknesses of this version

- **`reassurance` violates the span-locality rule.** Recurrence is established
  from CONTEXT while the span must come from the target turn. This is a genuine
  inconsistency, not an oversight; the alternative designs are worse. Expect it
  to be the lowest-agreement frequent label, and consider a dedicated
  `previously_asked` boolean derived programmatically from the summary rather
  than left to the model.
- **`blame_transfer` may be unevaluable.** It is well-grounded experimentally but
  may be near-absent in advisor transcripts, where the incentive to voice it is
  low. If the gold set turns up fewer than ~15 instances, report it as an
  observed-but-unmeasured category rather than scoring it.
- **The two-leaf cap is still arbitrary.** Measure the truncation rate on gold
  before keeping it.
- **Written vs voice channels are not distinguished.** `recognition` and
  `reassurance` both lean on tone, which is flattened in secure messages and
  email. Consider channel as a stratification variable throughout, and expect
  different per-label thresholds by channel.
- **Fourteen leaves is at the upper end of what annotators hold in working
  memory.** If pilot κ is poor across the board rather than on specific labels,
  the problem is codebook size, not the definitions — collapse to the five
  families and re-pilot.
