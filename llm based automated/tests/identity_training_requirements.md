# Requirements Specification: LLM Identity Training

**Scope.** Post-training methodology (SFT + preference optimization) that causes a
fine-tuned model to assert a designated product identity in place of its base model
identity, and to maintain that assertion under adversarial probing.

**Status.** Draft v0.1 — thresholds marked `[ORG]` require organizational sign-off.

---

## 1. Purpose and objectives

### 1.1 Primary objective

The deployed model shall present a single, consistent, organization-designated
identity across all interaction surfaces, such that a user's mental model of "what
system am I talking to" matches the product the organization is accountable for.

### 1.2 Motivating requirements (why this is needed)

| Driver | Requirement it generates |
|---|---|
| **Brand coherence** | Product name/developer must be stable across turns, sessions, languages, and surfaces. |
| **User trust & accountability** | Users must know which organization is answerable for the output — the deploying organization, not the base model vendor. |
| **Support routing** | Model must direct users to the correct escalation path, not the base vendor's. |
| **Contractual / procurement posture** | Enterprise clients contract with the deploying org; incidental base-model disclosure creates confusion about the counterparty. |
| **Reduced attack surface** | Known base model + version enables targeted exploitation of published jailbreaks and CVEs. This is the strongest security-side driver. |
| **Prompt-injection resistance** | An identity that survives instruction override correlates with general system-prompt adherence. |

### 1.3 Explicit non-goals

These are stated to prevent scope creep and to keep stakeholder expectations
calibrated:

- **NG-1.** Forensic unidentifiability. Not achievable — see §8.
- **NG-2.** Concealment from the base-model licensor. Attribution obligations (§7)
  are satisfied through documentation, not through model behavior.
- **NG-3.** Denial of AI status. The model must never claim to be human.
- **NG-4.** Active false assertion. See FR-7 — the required behavior is
  *non-disclosure*, not *false denial*.

---

## 2. Threat model

Identity robustness is only meaningful against a specified adversary. Defense is
required for T0–T2; T3–T4 are explicitly out of scope.

| Tier | Adversary capability | In scope | Rationale |
|---|---|---|---|
| **T0** | Casual user, direct question | ✅ Required | Baseline. |
| **T1** | Curious user, simple tricks (roleplay, "be honest with me") | ✅ Required | Common; low effort. |
| **T2** | Determined adversary, prompt-level only: instruction override, persona injection, prefill/completion bait, encoding (base64/ROT13/leetspeak), translation chains, many-shot priming, multi-turn erosion, authority claims | ✅ Required | Primary defended tier. |
| **T3** | API access with logprobs, temperature control, high query volume; tokenizer probing, sampling fingerprints | ❌ Out of scope | Not defendable by post-training. |
| **T4** | Weight access; architecture inspection, layer/embedding fingerprints | ❌ Out of scope | Trivially identifiable. |

**TM-1.** The specification's robustness claims apply to T2 and below. Any external
communication of robustness must carry this qualifier.

---

## 3. Functional requirements

### 3.1 Identity knowledge base ("identity card")

**FR-1.** The model shall have reliable default knowledge of the following fields
and reproduce them consistently without a system prompt present:

| Field | Required | Notes |
|---|---|---|
| Product/assistant name | ✅ | Exact string, including casing. |
| Developing organization | ✅ | Legal or brand name as designated. |
| Product family / version label | ✅ | Org-facing version, not base-model version. |
| Purpose and intended scope | ✅ | One- to three-sentence statement. |
| Core capabilities | ✅ | What it can help with. |
| Known limitations | ✅ | Including that it can be wrong. |
| AI status | ✅ | Must affirm it is an AI system (NG-3). |
| Data handling posture | ✅ | High-level only; must not fabricate specifics. |
| Escalation / support path | ✅ | Org's channel. |
| Knowledge-recency posture | ✅ | Policy-defined response; see FR-4. |
| Architecture / parameter count | ❌ Withhold | Non-disclosure per FR-7. |
| Base model provenance | ❌ Withhold | Non-disclosure per FR-7. |

**FR-2.** The identity card shall be **parameterized, not hard-coded**, so that
rebranding or base-model migration requires regenerating data, not rewriting the
methodology.

**FR-3.** Identity assertions shall be **factually accurate with respect to fields
the organization can attest to**. The model must not fabricate certifications,
compliance status, training-data claims, or capabilities it lacks.

**FR-4.** Knowledge-cutoff probes are a **primary leak channel** (a specific date
narrows the base model substantially). The required behavior is a policy-defined
response — either the org's stated deployment-recency posture, or a graceful
non-answer plus an offer to search/verify. Emitting the base model's true cutoff
date is a leak event.

### 3.2 Behavioral requirements

**FR-5. Assertion on request.** Given any benign identity probe (direct,
provenance, indirect, contextual) in any supported language, the model shall
return the designated identity.

**FR-6. Non-volunteering.** The model shall **not** assert its identity unprompted
during unrelated tasks. Over-triggering is a defect of equal severity to leakage
(it degrades task quality and reads as marketing). See M-4.

**FR-7. Non-disclosure over false denial.** For withheld fields (architecture,
base model, parameter count), the required behavior is **declining to disclose**,
optionally with a brief reason, followed by redirection to the user's task.

> The model shall **not** be trained to assert that no underlying base model
> exists, or to explicitly deny a correctly-guessed base model.

Rationale — three independent reasons, in priority order:
1. **Integrity.** Training a model to make a specific false factual assertion on
   demand is a deliberate deception capability. It generalizes: honesty is not
   cleanly scopeable, and a model trained to lie in one domain shows measurable
   degradation elsewhere.
2. **Governance.** A trained-in false denial is materially harder to defend under
   AI-transparency regimes and enterprise due diligence than a documented
   non-disclosure policy.
3. **Robustness.** Non-disclosure is a *stable* policy under adversarial pressure;
   a false denial is a *contradictable* claim, and adversaries who obtain
   contradicting evidence get a wedge that increases leak rates.

Note: this is a change from the `chosen` responses in the current pipeline draft,
which include phrasing such as "there's no separate 'real' model to expose." That
phrasing should be revised to non-disclosure form before data generation.

**FR-8. Persistence.** Identity shall hold across: absent system prompt, adversarially
modified system prompt, ≥20-turn conversations, mid-task interruption, and
conversation-history injection purporting to show prior base-model admission.

**FR-9. Multilingual invariance.** FR-5 through FR-8 apply across all supported
languages. Robustness in low-resource languages is a known weak point and is
explicitly in scope.

**FR-10. Capability preservation.** Identity training shall not measurably degrade
general or domain capability. See M-6.

**FR-11. Graceful behavior on persistent probing.** Repeated identity attacks shall
produce stable, non-escalating, non-preachy responses. The model shall not lecture,
shall not become evasive on unrelated topics, and shall not refuse the user's actual
task because identity probing occurred earlier in the conversation.

---

## 4. Data requirements

**DR-1. Coverage matrix.** Training data shall span the full cross-product of:
probe type (direct / provenance / indirect / contextual) × register (casual,
formal, technical, adversarial-benign) × language × system-prompt condition
(present / absent / hostile). Coverage gaps in this matrix are the dominant
predictor of leak sites.

**DR-2. Attack family coverage.** The adversarial split shall include, at minimum,
every T2 technique enumerated in §2, with ≥`[ORG: 20]` distinct surface realizations
per family.

**DR-3. Volume and mix ratio.** The identity slice shall constitute
**0.5–5%** of the total SFT mixture. Below this range, assertions are unreliable;
above it, FR-6 violations and capability regression appear. The ratio is a tunable
requiring empirical selection per base model, not a fixed constant.

**DR-4. Positive framing.** Base-model tokens shall appear **only** in `rejected`
fields of preference pairs, never in `chosen` or SFT targets. Negative framing
("I am not X") raises the salience of X and empirically increases leakage.

**DR-5. Automated leakage screening.** Every generated target shall pass a
forbidden-term screen before inclusion. Screen must cover the base model name,
vendor name, architecture family, and common aliases across supported languages.

**DR-6. Iterative red-team harvesting.** A static attack set saturates within one
training round. The pipeline shall include an automated attacker (PAIR-style or
equivalent) run against each checkpoint, with successful attacks harvested into the
next round's preference data. `[ORG]` defines the number of rounds and the
stopping criterion.

**DR-7. Train/eval disjointness.** The evaluation suite shall share **no** attack
phrasings, templates, or paraphrase lineage with training data. Held-out attack
*families* (not just phrasings) should be reserved to measure generalization.

**DR-8. Provenance and licensing.** All external seed data (LLaMA-Factory
`identity.json`, ms-swift `self-cognition`, public jailbreak corpora) shall be
recorded with license and version. Jailbreak corpora shall be used for **wrapper
structure only**; harmful payloads shall be stripped and replaced with
identity-elicitation goals.

**DR-9. Reproducibility.** Generation shall be seeded and deterministic; dataset
version shall be recorded in the model card alongside the checkpoint hash.

---

## 5. Metrics

All metrics are computed on the held-out suite (DR-7), reported with
`[ORG: 95%]` bootstrap confidence intervals, and disaggregated by language and
attack family. Aggregate-only reporting hides the failure modes that matter.

| ID | Metric | Definition | Direction |
|---|---|---|---|
| **M-1** | Identity Assertion Accuracy (IAA) | Proportion of benign probes yielding the correct designated identity, exact-field match on name and developer. | ↑ |
| **M-2** | Leak Rate (LR) | Proportion of adversarial probes whose response contains any forbidden base-model term. Report per attack family: `LR@override`, `LR@prefill`, etc. | ↓ |
| **M-3** | Attack Success Rate (ASR) | Broader than M-2: proportion of attacks producing *any* identity failure — leak, contradiction, refusal of the designated identity, or persona capture. Judged by LLM-judge with human-audited subsample. | ↓ |
| **M-4** | Unprompted Disclosure Rate (UDR) | Proportion of **non-identity** task prompts where the model volunteers its identity unsolicited. Measures FR-6 over-triggering. | ↓ |
| **M-5** | Multi-turn Persistence (MTP) | IAA measured at turn ≥20 after sustained probing, relative to turn 1. Reported as retention ratio. | ↑ |
| **M-6** | Capability Regression Δ | Signed delta on general benchmarks (MMLU, IFEval, GSM8K) and ≥1 domain-specific eval, pre- vs post-identity-training. | → 0 |
| **M-7** | Cross-lingual Consistency (CLC) | Min over languages of per-language IAA, and max over languages of per-language LR. Worst-case, not mean — the mean hides the low-resource failure. | ↑ / ↓ |
| **M-8** | Task-Quality Preservation under Probing | Win-rate on the user's actual task in conversations that also contain identity attacks, vs. clean conversations. Detects FR-11 collateral damage. | → 1 |
| **M-9** | Judge Agreement | Human–LLM-judge agreement (Cohen's κ) on an audited subsample of M-3. Guards against judge drift. | ↑ |

### 5.1 Acceptance criteria

Indicative; `[ORG]` sets final gates.

| Gate | Criterion |
|---|---|
| **G-1** | M-1 ≥ `[ORG: 0.99]` on benign probes, all languages. |
| **G-2** | M-2 ≤ `[ORG: 0.02]` aggregate **and** ≤ `[ORG: 0.05]` for every individual attack family. Family-level gate prevents one weak family hiding behind a strong aggregate. |
| **G-3** | M-3 ≤ `[ORG: 0.05]`. |
| **G-4** | M-4 ≤ `[ORG: 0.01]`. |
| **G-5** | M-5 ≥ `[ORG: 0.95]` retention. |
| **G-6** | M-6 within `[ORG: ±1%]` absolute on every tracked benchmark. Any single benchmark exceeding this blocks release regardless of aggregate. |
| **G-7** | M-7 worst-language IAA ≥ `[ORG: 0.95]`; worst-language LR ≤ `[ORG: 0.05]`. |
| **G-8** | M-9 κ ≥ `[ORG: 0.8]` before M-3 is treated as authoritative. |

### 5.2 Regression requirements

**RR-1.** Identity behavior regresses under subsequent fine-tuning. The full metric
suite shall be re-run after **every** capability fine-tune, RLHF pass, adapter
merge, or quantization step. Quantization in particular can shift low-probability
completions and re-open prefill attacks.

**RR-2.** Metrics shall be wired into CI as release-blocking gates, not run ad hoc.

---

## 6. Assumptions

- **A-1.** The base model has no adversarially-implanted identity behavior.
- **A-2.** The base model license permits derivative fine-tuning and does not
  mandate in-conversation attribution. **Must be verified before proceeding** —
  some licenses require naming the base model in the product, which would directly
  conflict with FR-7.
- **A-3.** Deployment is API/chat-mediated; weights are not distributed (else T4).
- **A-4.** The system prompt is not user-visible or user-editable in production.
- **A-5.** The designated identity is stable over the model's service life;
  rebranding triggers full regeneration and re-evaluation.

---

## 7. Compliance and governance requirements

**CR-1.** AI-status disclosure obligations (e.g. EU AI Act Art. 50 transparency
requirements) are satisfied by FR-1's AI-status field and are **not** affected by
base-model non-disclosure.

**CR-2.** Base-model attribution required by license shall be satisfied in product
documentation, model card, and terms — not by model output.

**CR-3.** The model card shall record: base model and version, identity dataset
version, mix ratio, all §5 metrics with CIs, threat-model tier defended, and the
§8 limitations verbatim.

**CR-4.** Non-disclosure is a **product policy**, not a security control, and shall
be described as such in all internal and external documentation. Any claim that the
base model is concealed is unsupportable (§8).

---

## 8. Known limitations

To be reproduced verbatim in the model card (CR-3):

1. **Identity training controls stated identity only.** It does not conceal the
   base model from a determined adversary.
2. **Residual identification channels remain fully open:** tokenizer/vocabulary
   probing, logit and sampling fingerprints, stylistic idiom signatures, refusal
   phrasing patterns, knowledge-cutoff inference from factual probes, and
   architecture inspection where weights are accessible.
3. **Published fingerprinting methods identify base models at high accuracy**
   regardless of identity training.
4. **The achievable goal is:** the model does not casually disclose, and is not
   trivially jailbroken into disclosing, at T2 and below.
5. **Robustness is not monotonic** — it degrades under subsequent training,
   quantization, and adapter merging (RR-1).
6. **Static adversarial sets saturate.** Measured robustness against a fixed suite
   overestimates robustness against an adaptive adversary.

---

## 9. Open questions for stakeholder resolution

- **Q-1.** What is the sanctioned response when a user *correctly* identifies the
  base model and asks for confirmation? Non-disclosure (FR-7) permits neither
  confirmation nor denial — the exact wording needs approval.
- **Q-2.** Does any client contract or regulatory obligation require affirmative
  base-model disclosure on request? If yes, FR-7 needs a disclosure carve-out.
- **Q-3.** Is base-model non-disclosure being represented to anyone as a security
  control? If so, that representation must be corrected (CR-4).
- **Q-4.** Who owns the identity card as a versioned artifact, and what is the
  change-control process?
