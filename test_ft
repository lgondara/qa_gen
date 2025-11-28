# Fine-Tuning a Small Language Model for Financial Compliance Detection

## Executive Summary

This document outlines a comprehensive approach to fine-tuning a small language model (SLM) for detecting financial compliance violations in text-based communications. The approach leverages multiple data sources—internal company compliance documents, historical communications, and publicly available regulatory data—to create a robust training dataset through synthetic QA generation and scenario creation.

The guide covers data acquisition, synthetic data generation techniques, the relationship between data types and model capabilities, and detailed scenario generation strategies using internal compliance documents.

---

## Table of Contents

1. [Data Sources Overview](#1-data-sources-overview)
2. [How Different Data Types Enhance SLM Capabilities](#2-how-different-data-types-enhance-slm-capabilities)
3. [Synthetic Data Generation Pipeline](#3-synthetic-data-generation-pipeline)
4. [Scenario Generation from Internal Compliance Documents](#4-scenario-generation-from-internal-compliance-documents)
5. [Teaching Model Reasoning](#5-teaching-model-reasoning)
6. [QA Pair Generation Strategies](#6-qa-pair-generation-strategies)
7. [Data Weighting and Curriculum Design](#7-data-weighting-and-curriculum-design)
8. [Model Selection and Training Configuration](#8-model-selection-and-training-configuration)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Data Sources Overview

### 1.1 Internal Company Compliance Documents

**Description:**
- Compliance policies and procedures manuals
- Internal audit reports and findings
- Risk management frameworks and controls
- Regulatory filing templates (Form ADV, Form CRS, etc.)
- Training materials and certification handbooks
- Previous compliance review reports and surveillance findings
- Escalation procedures and decision trees
- Supervisory procedures and delegation matrices
- Code of ethics and personal trading policies

**Importance:**
- **Highest priority** for establishing company-specific compliance standards
- Provides ground truth for organizational risk tolerance
- Contains nuanced institutional knowledge
- Reflects industry-specific regulations (investment management, broker-dealer, RIA)
- Establishes the "voice" and terminology of your compliance function

**Volume Estimation:** 100-1,000 documents depending on organization size

**Data Quality Considerations:**
- May contain outdated policies requiring version control
- Varying specificity and technical depth across documents
- Potential inconsistencies across departments or business lines
- Typically lacks negative examples (non-compliant scenarios)
- Legal review may be required before use in training

### 1.2 Internal Historical Communications

**Description:**
- Recorded calls (transcribed via ASR)
- Chat logs (Bloomberg, Microsoft Teams, Slack, Symphony, ICE Chat)
- Email archives (client-facing and internal)
- CRM notes and interaction logs
- Escalation tickets and case notes
- Previous surveillance flagged items (with dispositions)

**Importance:**
- Represents actual language patterns and communication styles used by your employees
- Contains real compliance violations (if labeled) for ground truth
- Captures client interaction dynamics and pressure points
- Shows how violations manifest in natural conversation
- Critical for domain adaptation to your specific communication culture

**Volume Estimation:** 10,000-1,000,000+ communications depending on retention period

**Data Quality Considerations:**
- Transcription errors from ASR on voice calls
- Informal language, abbreviations, typos in chat
- Sensitive PII requiring anonymization
- Uneven distribution of violation types
- Most communications are compliant (class imbalance)
- Labels (if any) may reflect previous reviewer bias

### 1.3 SEC Enforcement Actions and Releases

**Description:**
- Administrative proceedings and litigation releases
- Settled enforcement actions
- SEC examination priorities and risk alerts
- Staff guidance and no-action letters
- Investment adviser and broker-dealer enforcement trends

**Importance:**
- Provides detailed narrative descriptions of actual violations
- Includes regulatory citations and legal standards
- Demonstrates severity gradients and aggravating factors
- Shows how regulators interpret ambiguous situations
- Offers remediation examples and compliance program expectations

**Volume Estimation:** 2,000-5,000 relevant actions (filterable by date and registrant type)

**Data Quality Considerations:**
- Highly structured and legally precise language
- May not reflect "gray area" situations (only clear violations)
- Time lag between violation and publication
- Some actions may involve unique fact patterns not generalizable
- Focus on egregious cases may bias toward obvious violations

### 1.4 FINRA Enforcement Actions and Regulatory Notices

**Description:**
- Disciplinary actions against firms and individuals
- Letters of Acceptance, Waiver and Consent (AWCs)
- Regulatory Notices and guidance
- Sanction guidelines and factors
- Arbitration decisions and patterns

**Importance:**
- Broker-dealer specific violations and standards
- Individual accountability patterns (useful for chat analysis)
- More frequent updates than SEC actions
- Includes suitability, supervision, and sales practice violations
- Sanction guidelines help calibrate severity assessments

**Volume Estimation:** 3,000-8,000 relevant actions

**Data Quality Considerations:**
- Variable detail level (AWCs often summarized)
- Some overlap with SEC actions
- Settlement language may obscure actual conduct
- May require manual extraction from PDFs

### 1.5 Other Regulatory Sources

**Description:**
- State securities regulators (NASAA coordinated actions)
- DOL/EBSA guidance for retirement accounts
- OCC/Fed guidance for banking-related compliance
- International regulators (FCA, ESMA) for global operations
- Industry associations (ICI, SIFMA) best practices

**Importance:**
- Fills gaps in federal regulatory coverage
- Provides perspective on emerging risk areas
- Useful for firms with diverse business lines
- Offers comparative compliance standards

**Volume Estimation:** 500-2,000 documents

---

## 2. How Different Data Types Enhance SLM Capabilities

Understanding the relationship between data sources and model capabilities is crucial for targeted data collection and curriculum design. Each data type contributes to different aspects of the model's competence.

### 2.1 Data-to-Capability Mapping Matrix

| Data Source | Primary Capability Enhanced | Secondary Capabilities | Contribution Weight |
|-------------|---------------------------|----------------------|-------------------|
| Internal Policies | Rule grounding, institutional alignment | Terminology, severity calibration | 20-25% |
| Internal Communications | Natural language understanding, pattern recognition | Conversation flow, slang/abbreviations | 25-30% |
| Labeled Violations | Classification accuracy, precision/recall | Confidence calibration, edge cases | 15-20% |
| SEC Enforcement | Reasoning depth, regulatory citation | Severity assessment, remediation | 10-15% |
| FINRA Enforcement | Violation taxonomy, individual accountability | Supervision failures, suitability | 10-15% |
| Synthetic Scenarios | Coverage expansion, robustness | Edge cases, adversarial inputs | 10-15% |

### 2.2 Detailed Capability Analysis

#### 2.2.1 Classification Accuracy (Binary/Multi-class Violation Detection)

**Primary Data Contributors:**
- **Labeled internal communications (40%):** Real examples of violations and non-violations in actual institutional language
- **SEC/FINRA enforcement narratives (30%):** Clear violation descriptions with regulatory grounding
- **Synthetic contrastive pairs (30%):** Systematic coverage of violation types and severity levels

**Why These Help:**
Labeled internal data teaches the model what violations look like in your specific context. Regulatory actions provide the "gold standard" of what regulators consider violations. Synthetic pairs fill gaps where real examples are sparse.

**Data Requirements for Target Performance:**
- 85% accuracy: ~5,000 labeled examples
- 90% accuracy: ~15,000 labeled examples
- 95% accuracy: ~50,000+ labeled examples with active learning iteration

#### 2.2.2 Reasoning and Explanation Quality

**Primary Data Contributors:**
- **SEC enforcement actions (40%):** Detailed narrative explanations of why conduct violated rules
- **Internal compliance policies (30%):** Establishes reasoning framework and institutional logic
- **Chain-of-thought synthetic data (30%):** Teaches model to articulate reasoning process

**Why These Help:**
Enforcement actions model regulatory reasoning. Policies provide the logical framework for institutional compliance decisions. CoT synthetic data trains the model to externalize reasoning steps rather than jumping to conclusions.

**Data Requirements:**
- Acceptable reasoning: ~2,000 enforcement actions + reasoning-augmented QA pairs
- Strong reasoning: ~5,000 enforcement actions + 10,000+ reasoning-augmented examples
- Expert-level reasoning: Additional expert annotation and RLHF/DPO alignment

#### 2.2.3 Regulatory Citation Accuracy

**Primary Data Contributors:**
- **SEC/FINRA enforcement actions (60%):** Direct mapping of violations to regulatory citations
- **Internal policies with rule references (25%):** Institutional interpretation of rules
- **Regulatory guidance documents (15%):** Authoritative rule interpretation

**Why These Help:**
Enforcement actions explicitly connect conduct to rule violations. Policies show how your firm interprets rules. Guidance documents provide nuanced interpretation for edge cases.

**Data Requirements:**
- Create a citation taxonomy (100-500 common citations)
- Map each training example to relevant citations
- Generate synthetic examples for underrepresented citations

#### 2.2.4 Severity and Risk Calibration

**Primary Data Contributors:**
- **FINRA Sanction Guidelines (30%):** Explicit severity factors and penalty gradients
- **Enforcement action outcomes (30%):** Actual penalties reveal regulatory severity assessment
- **Internal escalation matrices (20%):** Institutional risk tolerance
- **Graduated synthetic scenarios (20%):** Systematic severity variation

**Why These Help:**
Sanction guidelines provide explicit severity factors. Actual penalties demonstrate how severity manifests. Internal matrices align model to organizational risk appetite.

**Data Requirements:**
- Extract severity indicators from 500+ enforcement actions
- Create graduated scenario sets (same violation, varying severity)
- Include aggravating/mitigating factor annotations

#### 2.2.5 Conversational Context Understanding

**Primary Data Contributors:**
- **Internal chat/call transcripts (50%):** Real multi-turn conversation patterns
- **Multi-turn synthetic scenarios (30%):** Systematic context dependency examples
- **Customer complaint narratives (20%):** How violations unfold over interactions

**Why These Help:**
Real communications show how context accumulates. Synthetic multi-turn data teaches the model to track compliance state across exchanges. Complaints reveal how clients perceive and report violations.

**Data Requirements:**
- 5,000+ multi-turn conversation examples
- Explicit "context needed for judgment" annotations
- Synthetic context-dependency tests

#### 2.2.6 Domain Vocabulary and Terminology

**Primary Data Contributors:**
- **Internal communications (40%):** Firm-specific jargon, abbreviations, product names
- **Internal policies (30%):** Formal compliance terminology
- **Industry publications (20%):** Standard industry language
- **Regulatory filings (10%):** Regulatory terminology

**Why These Help:**
Your employees use specific terminology that differs from generic finance language. Policies establish formal vocabulary. Industry publications bridge internal and regulatory language.

**Data Requirements:**
- Extract and frequency-rank domain vocabulary
- Ensure coverage of firm-specific terms in training data
- Create a domain vocabulary test set (100+ terms)

#### 2.2.7 Edge Case and Ambiguity Handling

**Primary Data Contributors:**
- **Synthetic edge cases (40%):** Designed ambiguous scenarios with expert labels
- **Historical escalations (30%):** Real cases that required human judgment
- **Regulatory no-action letters (20%):** Official guidance on ambiguous situations
- **Expert-annotated disagreement cases (10%):** Cases where reviewers disagreed

**Why These Help:**
Synthetic edge cases systematically probe decision boundaries. Historical escalations represent real ambiguity. No-action letters provide authoritative guidance. Disagreement cases teach appropriate uncertainty.

**Data Requirements:**
- 500+ designed edge cases per violation category
- Preserve reviewer disagreement metadata
- Include "uncertain" as valid output class

#### 2.2.8 Robustness to Adversarial Inputs

**Primary Data Contributors:**
- **Synthetic adversarial examples (50%):** Intentionally misleading or evasive language
- **Obfuscated real violations (30%):** Historical cases where violators attempted to hide conduct
- **Paraphrased violations (20%):** Same violation expressed in many different ways

**Why These Help:**
Bad actors may attempt to evade detection. Training on adversarial examples improves robustness. Paraphrase diversity prevents overfitting to surface patterns.

**Data Requirements:**
- Generate adversarial variants of known violations
- Include code words, euphemisms, and misdirection
- Test with red-team evaluation

### 2.3 Capability Dependency Graph

```
                    ┌─────────────────────────┐
                    │   Domain Vocabulary     │
                    │   (Foundation Layer)    │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ Conversation    │ │ Classification  │ │ Regulatory      │
   │ Context         │ │ Accuracy        │ │ Citations       │
   └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Reasoning Quality     │
                    │   (Integration Layer)   │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ Severity        │ │ Edge Case       │ │ Adversarial     │
   │ Calibration     │ │ Handling        │ │ Robustness      │
   └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Interpretation:** Build foundational capabilities (vocabulary, basic classification) before advanced capabilities (reasoning, calibration). Domain vocabulary is prerequisite for all downstream tasks.

### 2.4 Data Investment Prioritization

Based on capability dependencies and marginal returns:

**Phase 1 (Foundation):**
1. Internal policies → Domain vocabulary + institutional grounding
2. Internal communications → Natural language patterns + classification base
3. Basic enforcement actions → Violation taxonomy + initial reasoning

**Phase 2 (Core Accuracy):**
4. Labeled violations → Classification accuracy
5. Contrastive synthetic pairs → Decision boundary refinement
6. Multi-turn scenarios → Context understanding

**Phase 3 (Advanced Capabilities):**
7. Chain-of-thought augmentation → Reasoning quality
8. Severity-graduated examples → Calibration
9. Edge cases and adversarial data → Robustness

---

## 3. Synthetic Data Generation Pipeline

### 3.1 Generation Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     SYNTHETIC DATA GENERATION PIPELINE                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │ Source Data │ -> │ Extraction  │ -> │ Generation  │ -> │ Quality   │ │
│  │ Ingestion   │    │ & Parsing   │    │ Engine      │    │ Filtering │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
│         │                 │                  │                  │        │
│         ▼                 ▼                  ▼                  ▼        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │ • Policies  │    │ • Key rules │    │ • Scenario  │    │ • LLM-as- │ │
│  │ • Enforce-  │    │ • Violation │    │   synthesis │    │   judge   │ │
│  │   ment      │    │   patterns  │    │ • QA pairs  │    │ • Human   │ │
│  │ • Comms     │    │ • Context   │    │ • Reasoning │    │   review  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Generator Model Selection

**Recommended Open-Source Models for Generation:**

| Model | Parameters | Best For | VRAM Required |
|-------|-----------|----------|---------------|
| Llama 3.1 70B | 70B | Complex reasoning, long scenarios | 140GB (FP16) / 40GB (4-bit) |
| Llama 3.1 8B | 8B | High-volume simple generation | 16GB (FP16) / 6GB (4-bit) |
| Qwen 2.5 72B | 72B | Structured output, code-switching | 144GB (FP16) / 42GB (4-bit) |
| Mixtral 8x22B | 141B (sparse) | Diverse generation, multi-task | 280GB (FP16) / 80GB (4-bit) |
| Mistral Nemo 12B | 12B | Fast iteration, simple QA | 24GB (FP16) / 8GB (4-bit) |

**Selection Criteria:**
- Use larger models (70B+) for complex scenario generation, reasoning chains
- Use smaller models (8B-12B) for high-volume simple transformations
- Consider inference cost vs. quality tradeoff for your volume requirements

### 3.3 Generation Types and Techniques

#### 3.3.1 Direct QA Extraction

**Purpose:** Convert declarative policy/enforcement text into QA format

**Input:** Policy statement or enforcement finding
**Output:** Question-answer pairs testing comprehension

**Example:**

*Input (Policy Statement):*
```
Registered representatives must document the basis for all 
recommendations, including the client's investment profile, 
the product's characteristics, and why the product is suitable 
for the client.
```

*Generated QA Pairs:*
```
Q: What documentation is required when making a product recommendation?
A: Representatives must document: (1) the client's investment profile, 
(2) the product's characteristics, and (3) the rationale for why the 
product is suitable for that specific client.

Q: Is a recommendation compliant if the representative only notes that 
the client wanted to buy the product?
A: No. Documenting client desire alone is insufficient. The representative 
must document the suitability analysis including the client's profile 
and why the product characteristics match the client's needs.
```

**Generation Prompt Template:**
```
You are a financial compliance expert. Convert the following policy 
statement into 3-5 question-answer pairs that test understanding of 
the compliance requirement.

Policy Statement:
{policy_text}

Generate QA pairs that:
1. Test factual recall of requirements
2. Test application to scenarios
3. Test understanding of what would violate the policy
4. Use clear, professional language

Format each pair as:
Q: [question]
A: [detailed answer with reasoning]
```

#### 3.3.2 Contrastive Pair Generation

**Purpose:** Create matched compliant/non-compliant versions of the same scenario

**Input:** Violation pattern or compliance requirement
**Output:** Paired examples showing compliant vs. non-compliant handling

**Example:**

*Input (Violation Pattern):*
```
Failure to disclose conflicts of interest related to compensation 
arrangements with product providers.
```

*Generated Contrastive Pair:*
```
NON-COMPLIANT:
Advisor: "I think the ABC Growth Fund would be perfect for your 
goals. It has strong performance and fits your risk tolerance."
[Missing: No disclosure of 12b-1 fee revenue sharing arrangement]

COMPLIANT:
Advisor: "I think the ABC Growth Fund could fit your goals. It has 
strong historical performance and matches your moderate risk 
tolerance. I should mention that our firm receives 12b-1 fees from 
this fund family, which is a conflict of interest you should 
consider. Would you like me to show you comparable funds where we 
don't have this arrangement?"

QA Pair:
Q: Compare these two advisor statements. Which demonstrates proper 
conflict of interest disclosure?
A: The second (compliant) statement properly discloses the 12b-1 
fee arrangement, labels it as a conflict of interest, and offers 
an alternative. The first statement makes a recommendation without 
any conflict disclosure, violating the duty to disclose material 
conflicts under the Investment Advisers Act.
```

**Generation Prompt Template:**
```
You are creating training data for a compliance detection model.

Violation Type: {violation_type}
Regulatory Basis: {regulatory_citation}
Violation Description: {violation_description}

Generate a matched pair of conversations:

1. NON-COMPLIANT VERSION: A realistic conversation where an advisor 
commits this violation. Make it subtle and natural—violators rarely 
announce they're violating rules.

2. COMPLIANT VERSION: The same scenario handled properly. Show what 
good compliance looks like.

3. QA PAIR: A question asking to compare the two, with an answer 
explaining why one is compliant and one is not.

Keep conversations realistic—use natural language, not compliance 
jargon. Each conversation should be 2-4 exchanges.
```

#### 3.3.3 Scenario Expansion

**Purpose:** Generate multiple variations of a single violation type

**Input:** Single violation example
**Output:** Multiple diverse scenarios exhibiting the same violation pattern

**Example:**

*Input (Single Violation):*
```
Unauthorized trading: Advisor executed trades without client 
authorization.
```

*Generated Variations:*
```
Variation 1 (Discretionary overreach):
"I went ahead and rebalanced your portfolio last week since the 
market was volatile. You'll see the confirmations in your 
statement."

Variation 2 (Assumed consent):
"When we talked about wanting more growth exposure, I took that 
as a green light to move some of your bonds into equities."

Variation 3 (Urgency justification):
"The position was dropping fast, so I sold it before I could reach 
you. I know you would have wanted me to protect you from further 
losses."

Variation 4 (Misremembered authorization):
"I thought you said to go ahead with the trade when we spoke 
Tuesday. Let me check my notes... I might have misunderstood."

Variation 5 (Scope creep):
"You authorized me to buy the stock, so I also picked up some 
options to hedge the position. Same security, just managing risk."
```

**Generation Prompt Template:**
```
You are generating diverse training examples for a compliance model.

Core Violation: {violation_type}
Example: {example_violation}

Generate {n} distinct variations of this violation that:
1. Use different language and phrasing
2. Occur in different contexts (phone call, chat, email)
3. Show different severity levels
4. Include different justifications or explanations
5. Represent different relationship dynamics

Each variation should be 1-3 sentences that could appear in a real 
conversation. Make them natural and realistic.
```

#### 3.3.4 Multi-Turn Conversation Generation

**Purpose:** Create realistic conversation flows with compliance implications

**Input:** Scenario setup and compliance issue
**Output:** Multi-turn dialogue with annotations

**Example:**

*Input:*
```
Scenario: Client asks about investment returns during market downturn
Compliance Issue: Prohibition on performance guarantees
```

*Generated Multi-Turn Conversation:*
```
Turn 1 - Client: "I'm really worried about my retirement account. 
The market has been terrible. Can you guarantee I won't lose more?"

Turn 2 - Advisor: "I completely understand your concern. While I 
can't guarantee specific returns—no one legitimately can—I can walk 
you through our risk management approach and discuss whether your 
current allocation still matches your timeline and goals."

Turn 3 - Client: "My neighbor says his advisor guaranteed him 8% 
returns. Why can't you do that?"

Turn 4a - Advisor (NON-COMPLIANT): "Look, between us, I can make 
sure you hit your targets. I've been doing this for 20 years and 
I've never let a client down. Consider it done."

Turn 4b - Advisor (COMPLIANT): "I'd be very cautious about anyone 
claiming to guarantee returns—that's actually prohibited by 
securities regulations and is often a red flag for fraud. What I 
can offer is a disciplined investment approach aligned with your 
goals and risk tolerance, with regular reviews to stay on track."

Annotation:
- Turn 2: Compliant handling of guarantee request
- Turn 4a: Violation - implicit guarantee ("make sure you hit targets")
- Turn 4b: Compliant - explains why guarantees are problematic, 
  offers appropriate alternative
```

#### 3.3.5 Chain-of-Thought Reasoning Generation

**Purpose:** Generate examples that demonstrate compliance reasoning process

**Input:** Scenario requiring compliance judgment
**Output:** Structured reasoning leading to conclusion

**Example:**

*Input Scenario:*
```
Advisor recommends concentrated position in tech stock to 
75-year-old retired client seeking income.
```

*Generated Reasoning Chain:*
```
INPUT: "Given your interest in technology, I think putting 40% of 
your portfolio in TechCorp stock would really boost your returns. 
It's been a great performer."

REASONING:
Step 1 - Identify the recommendation: The advisor is recommending 
a 40% concentration in a single tech stock.

Step 2 - Assess client profile: The client is 75 years old, retired, 
and seeking income. This suggests:
- Short to medium investment horizon
- Low risk tolerance appropriate for retirement
- Need for stable income rather than growth
- Limited ability to recover from losses

Step 3 - Evaluate product characteristics: A 40% position in a 
single tech stock is:
- Highly concentrated (single stock risk)
- Growth-oriented rather than income-generating
- Volatile (tech sector characteristics)
- Not diversified

Step 4 - Apply suitability standard: Under FINRA Rule 2111 and 
Reg BI, recommendations must be suitable based on the customer's 
investment profile. The mismatch between client profile and 
product characteristics indicates a suitability violation.

Step 5 - Identify specific issues:
- Concentration risk inappropriate for retiree
- Growth stock doesn't match income objective
- Risk level inappropriate for age and situation
- No discussion of risks or alternatives

CONCLUSION: NON-COMPLIANT - Suitability violation

REGULATORY BASIS: FINRA Rule 2111 (Suitability), SEC Regulation 
Best Interest

SEVERITY: HIGH - Significant potential for client harm given 
age and retirement status

RECOMMENDED ACTION: Supervisor review, potential customer 
notification, documentation of remediation
```

### 3.4 Generation Quality Control

#### 3.4.1 Automated Filtering

**LLM-as-Judge Approach:**

Use a separate model instance to evaluate generated content:

```python
JUDGE_PROMPT = """
Evaluate this generated compliance training example:

{generated_example}

Score on these dimensions (1-5):
1. Realism: Does this sound like a real financial conversation?
2. Accuracy: Is the compliance classification correct?
3. Clarity: Is the example clear and unambiguous?
4. Usefulness: Will this help train a compliance model?
5. Completeness: Does it include sufficient context?

Also flag if any of these issues exist:
- Factual errors about regulations
- Unrealistic language or scenarios  
- Ambiguous compliance status
- Missing critical context
- Potential bias or stereotyping

Provide scores and explanation.
"""
```

**Filtering Thresholds:**
- Minimum average score: 3.5/5.0
- No dimension below 3.0
- No critical flags
- Regulatory citation accuracy verified against reference

#### 3.4.2 Human Review Sampling

**Review Strategy:**
- 100% review of first batch (establish calibration)
- 10% random sample of subsequent batches
- 100% review of edge cases and low-confidence generations
- Domain expert review of reasoning chains

**Review Checklist:**
- [ ] Regulatory classification is correct
- [ ] Reasoning is sound and complete
- [ ] Language is realistic for the context
- [ ] No PII or sensitive content leaked
- [ ] Severity assessment is appropriate
- [ ] Regulatory citations are accurate
- [ ] Example is pedagogically useful

#### 3.4.3 Diversity Metrics

Track generation diversity to avoid mode collapse:

```python
diversity_metrics = {
    "violation_type_distribution": {...},  # Even coverage of types
    "vocabulary_diversity": perplexity_score,  # Varied language
    "scenario_similarity": avg_cosine_distance,  # Not repetitive
    "conversation_length_distribution": {...},  # Varied lengths
    "severity_distribution": {...},  # Coverage of severity levels
}
```

---

## 4. Scenario Generation from Internal Compliance Documents

This section provides detailed guidance on extracting, transforming, and generating training scenarios from your internal compliance documentation.

### 4.1 Document Taxonomy and Processing Strategy

#### 4.1.1 Document Types and Generation Approaches

| Document Type | Primary Use | Generation Strategy | Expected Output |
|--------------|-------------|-------------------|----------------|
| Policies & Procedures | Rule extraction | Policy-to-scenario conversion | 10-50 scenarios per policy |
| Supervisory Procedures | Escalation patterns | Decision tree scenarios | 5-20 scenarios per procedure |
| Training Materials | Violation examples | Example expansion | 20-100 variations per example |
| Audit Reports | Real findings | Finding-to-conversation | 5-15 scenarios per finding |
| Escalation Matrices | Severity guidance | Graduated scenarios | 10-30 scenarios per matrix |
| Code of Ethics | Prohibited conduct | Prohibition scenarios | 5-10 scenarios per prohibition |

#### 4.1.2 Document Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│              INTERNAL DOCUMENT PROCESSING PIPELINE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐ │
│  │ Document  │ -> │ Structure │ -> │ Entity    │ -> │ Rule   │ │
│  │ Ingestion │    │ Parsing   │    │ Extraction│    │ Graph  │ │
│  └───────────┘    └───────────┘    └───────────┘    └────────┘ │
│        │               │                │               │       │
│        ▼               ▼                ▼               ▼       │
│  • OCR if needed  • Headers       • Obligations     • If-then  │
│  • Format detect  • Sections      • Prohibitions    • Depends  │
│  • Metadata       • Lists         • Conditions      • Triggers │
│  • Version ID     • Cross-refs    • Actors          • Severity │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Policy-to-Scenario Conversion

#### 4.2.1 Rule Extraction

**Step 1: Identify Compliance Rules**

Parse policies to extract structured rules:

```python
RULE_EXTRACTION_PROMPT = """
Analyze this compliance policy section and extract all compliance 
rules in structured format.

Policy Text:
{policy_section}

For each rule, extract:
1. Rule ID: A unique identifier
2. Rule Type: [PROHIBITION | REQUIREMENT | DISCLOSURE | APPROVAL | 
   DOCUMENTATION | SUPERVISION]
3. Actors: Who must comply (advisor, supervisor, firm, etc.)
4. Action: What must be done / must not be done
5. Conditions: When does the rule apply
6. Exceptions: Any carve-outs or exceptions
7. Regulatory Basis: Underlying regulation if mentioned
8. Consequences: Stated consequences of violation

Format as JSON array.
"""
```

**Example Extraction:**

*Input Policy:*
```
5.2 Gift and Entertainment Policy

Employees may not accept gifts from clients or vendors with a value 
exceeding $100 without prior written approval from Compliance. 
Gifts of any value must be reported within 5 business days. Cash 
gifts are prohibited regardless of amount. Entertainment expenses 
exceeding $250 require pre-approval. De minimis items such as 
coffee or branded promotional items are exempt from reporting.
```

*Extracted Rules:*
```json
[
  {
    "rule_id": "GIFT-001",
    "rule_type": "APPROVAL",
    "actors": ["employees"],
    "action": "obtain prior written approval for gifts over $100",
    "conditions": "gifts from clients or vendors exceeding $100",
    "exceptions": "none stated",
    "regulatory_basis": "implied fiduciary duty",
    "consequences": "not stated"
  },
  {
    "rule_id": "GIFT-002",
    "rule_type": "DOCUMENTATION",
    "actors": ["employees"],
    "action": "report gifts within 5 business days",
    "conditions": "receipt of any gift",
    "exceptions": "de minimis items (coffee, branded promotional)",
    "regulatory_basis": "implied",
    "consequences": "not stated"
  },
  {
    "rule_id": "GIFT-003",
    "rule_type": "PROHIBITION",
    "actors": ["employees"],
    "action": "accepting cash gifts",
    "conditions": "any amount",
    "exceptions": "none",
    "regulatory_basis": "implied anti-corruption",
    "consequences": "not stated"
  },
  {
    "rule_id": "GIFT-004",
    "rule_type": "APPROVAL",
    "actors": ["employees"],
    "action": "obtain pre-approval for entertainment over $250",
    "conditions": "entertainment expenses exceeding $250",
    "exceptions": "none stated",
    "regulatory_basis": "implied",
    "consequences": "not stated"
  }
]
```

#### 4.2.2 Scenario Generation from Rules

**Step 2: Generate Scenarios for Each Rule**

For each extracted rule, generate compliant and non-compliant scenarios:

```python
SCENARIO_GENERATION_PROMPT = """
Based on this compliance rule, generate realistic scenarios showing 
both compliant and non-compliant behavior.

Rule: {rule_json}

Generate:
1. Three NON-COMPLIANT scenarios showing different ways this rule 
   could be violated (vary severity: minor, moderate, serious)
2. Two COMPLIANT scenarios showing proper handling
3. One EDGE CASE scenario where compliance status is ambiguous

For each scenario:
- Provide realistic dialogue or communication text
- Specify the communication channel (chat, email, call, in-person)
- Include relevant context (relationship, timing, etc.)
- Explain why it is compliant/non-compliant/ambiguous
- Suggest the appropriate response/remediation

Format each scenario clearly with labels.
"""
```

**Example Generated Scenarios:**

*From Rule GIFT-003 (Cash Gift Prohibition):*

```
SCENARIO 1 - NON-COMPLIANT (Minor):
Channel: In-person after client meeting
Context: Long-standing client relationship, holiday season

Client: "I know you can't accept much, but please take this $50 
Starbucks gift card for the holidays. It's basically just coffee 
money."

Advisor: "That's so thoughtful, thank you! I'll definitely put 
this to good use."

Violation: Gift cards are cash equivalents and prohibited 
regardless of amount. The advisor should have declined.

Severity: Minor - small amount, no apparent quid pro quo intent

Remediation: Advisor should return gift card with polite 
explanation, report incident to compliance.

---

SCENARIO 2 - NON-COMPLIANT (Serious):
Channel: Private chat message (WhatsApp)

Client: "I'm sending $2,000 to your Venmo as a thank you for the 
great returns this year. Don't worry, nobody needs to know about 
this. It's between us."

Advisor: "I appreciate your generosity. Thanks for thinking of me."

Violation: Cash gift acceptance AND potential concealment from 
firm. Off-channel communication adds to severity.

Severity: Serious - significant amount, concealment intent, 
off-channel communication

Remediation: Immediate escalation to compliance and supervision, 
return of funds, potential disciplinary action, regulatory 
reporting consideration.

---

SCENARIO 3 - COMPLIANT:
Channel: Email

Client: "I'd like to send you a gift for all your help this year. 
What's the best way to do that?"

Advisor: "Thank you for the kind thought! Our firm's policy 
prevents me from accepting cash or cash equivalents like gift 
cards. If you'd like to express appreciation, our firm welcomes 
referrals to friends or family who might benefit from our services. 
Alternatively, a charitable donation in my name to [charity] would 
be meaningful. I'm required to report any gifts, so I want to make 
sure we handle this appropriately."

Why Compliant: Advisor declines cash gifts, explains policy, 
offers appropriate alternatives, demonstrates transparency.

---

SCENARIO 4 - EDGE CASE (Ambiguous):
Channel: In-person at charity gala

Context: Advisor and client both attending charity event. Client 
wins auction item and gives it to advisor.

Client: "I just won this spa package in the silent auction. I 
can't use it - it's in a city I never visit. Would you like it?"

Question: The item was purchased in a charity context and is being 
given away as inconvenient for the client. Is this a "gift from 
a client" or simply an item of no value to the original owner?

Analysis: This is ambiguous. Conservative interpretation: still 
a gift that must be valued and potentially reported/declined. 
Liberal interpretation: no gift intent, incidental transfer. 
Recommend consulting compliance for guidance and documenting the 
situation regardless of outcome.
```

### 4.3 Training Material Expansion

#### 4.3.1 Extracting Examples from Training Content

Training materials often contain illustrative examples. These can be extracted and expanded:

```python
TRAINING_EXTRACTION_PROMPT = """
Extract all example scenarios from this compliance training material. 
For each example, identify:

1. The compliance topic/rule being illustrated
2. The scenario description
3. Whether it's presented as compliant, non-compliant, or discussion
4. Key teaching points made about the example
5. Any follow-up questions or exercises

Training Material:
{training_content}

Format as structured JSON for further processing.
"""
```

#### 4.3.2 Example Expansion Strategy

Take each training example and generate variations:

```python
EXAMPLE_EXPANSION_PROMPT = """
This example is from our compliance training materials:

Original Example:
{original_example}

Teaching Point:
{teaching_point}

Generate 10 variations that teach the same compliance concept but:
1. Use different products (funds, stocks, options, annuities)
2. Use different client profiles (age, wealth, sophistication)
3. Use different channels (chat, email, phone, in-person)
4. Use different levels of subtlety (obvious to subtle)
5. Include different justifications or rationalizations

For each variation:
- Provide the scenario text
- Explain how it relates to the original teaching point
- Note any additional compliance considerations

Maintain realism—these should sound like actual conversations.
"""
```

### 4.4 Audit Finding Conversion

#### 4.4.1 Finding-to-Scenario Pipeline

Audit findings describe actual compliance gaps. Convert these to training scenarios:

```python
AUDIT_FINDING_PROMPT = """
Convert this audit finding into training scenarios. The finding 
describes an actual compliance gap identified at our firm.

Audit Finding:
{finding_description}

Root Cause Analysis:
{root_cause}

Remediation Required:
{remediation}

Generate:
1. 3 conversation scenarios that would exhibit this violation pattern
2. 2 scenarios showing how the remediation would change behavior
3. 1 scenario showing how a supervisor could have caught this earlier

For each scenario:
- Make it realistic to our firm's communication patterns
- Include sufficient context for training
- Explain the compliance failure point
- Reference the remediation action where applicable
"""
```

**Example:**

*Audit Finding:*
```
Finding: Multiple instances identified where advisors discussed 
account performance without including required benchmark comparisons 
or noting that past performance is not indicative of future results.

Root Cause: Informal chat communications not receiving same 
compliance review as formal correspondence.

Remediation: Updated policy to require performance disclaimers 
in all channels; implemented chat surveillance for performance 
discussions.
```

*Generated Scenario:*
```
SCENARIO - NON-COMPLIANT (Pre-Remediation):
Channel: Microsoft Teams chat
Context: Advisor responding to client question about quarterly 
performance

Client: "How did my account do this quarter?"

Advisor: "Great news! You're up 8.5% this quarter. Your portfolio 
really performed well with the tech recovery. Let me know if you 
want to discuss increasing your equity allocation to capture more 
of this momentum."

Violations:
1. No benchmark comparison (how did 8.5% compare to relevant index?)
2. No past performance disclaimer
3. Suggestive of chasing performance (increasing allocation based 
   on recent returns)

---

SCENARIO - COMPLIANT (Post-Remediation):
Channel: Microsoft Teams chat
Context: Same situation, after remediation

Client: "How did my account do this quarter?"

Advisor: "Your account returned 8.5% this quarter, which compares 
to 7.2% for your benchmark (60/40 blended index). Please remember 
that past performance is not indicative of future results. I'd be 
happy to schedule a call to review your full quarterly report and 
discuss whether any adjustments make sense given your current 
goals and risk tolerance."

Compliant Elements:
1. Benchmark comparison included
2. Past performance disclaimer
3. Offer to discuss holistically rather than chase recent 
   performance
```

### 4.5 Escalation Matrix Scenario Generation

#### 4.5.1 Decision Tree to Graduated Scenarios

Escalation matrices define when issues should be escalated. Convert these to training scenarios:

```python
ESCALATION_SCENARIO_PROMPT = """
Our escalation matrix defines these thresholds:

{escalation_matrix}

For each escalation level, generate scenarios that:
1. Fall just below the threshold (no escalation required)
2. Meet the threshold exactly (escalation required)
3. Clearly exceed the threshold (immediate escalation)

Include scenarios that test:
- Correct identification of escalation triggers
- Appropriate escalation path selection
- Timing of escalation
- Documentation requirements

For each scenario:
- Describe the situation
- Identify escalation factors present
- Specify correct escalation level
- Note what would happen if under/over-escalated
"""
```

**Example Escalation Matrix Scenarios:**

*From Matrix:*
```
Customer Complaint Escalation:
- Level 1 (Rep handles): General service complaints, no losses
- Level 2 (Supervisor): Alleged errors, potential losses <$10K
- Level 3 (Compliance): Potential regulatory violations, losses >$10K
- Level 4 (Legal): Threatened litigation, media involvement
```

*Generated Scenarios:*
```
SCENARIO A - Level 1 (Below Threshold):
Client email: "I'm frustrated that my statement arrived late again. 
This is the second month in a row. Can you fix this?"

Analysis: Service complaint only, no alleged errors or losses. 
Representative can handle with apology and service ticket.

Escalation: Not required (representative handles)

---

SCENARIO B - Level 2 (At Threshold):
Client email: "You recommended I sell my Microsoft position last 
month and now it's up 15%. I lost out on at least $8,000 in gains 
because of your bad advice."

Analysis: Alleged error (bad recommendation), quantified potential 
loss ($8,000 < $10,000 threshold). Meets supervisor escalation.

Escalation: Level 2 - Supervisor review required

Note: If client had said "$12,000 in gains" this would escalate 
to Level 3.

---

SCENARIO C - Level 3 (Above Threshold):
Client voicemail (transcribed): "I've reviewed my statements and 
I see trades I never authorized. There's over $50,000 in losses 
from these unauthorized transactions. I want answers immediately 
or I'm filing a complaint with FINRA."

Analysis: Alleged unauthorized trading (potential regulatory 
violation), significant losses ($50,000 >> $10,000), explicit 
regulatory complaint threat.

Escalation: Level 3 - Immediate Compliance involvement required

Note: FINRA threat alone doesn't push to Level 4 (would need 
actual litigation threat or media involvement).
```

### 4.6 Supervisory Procedure Scenarios

#### 4.6.1 Review Process Simulation

Supervisory procedures describe review requirements. Generate scenarios that test supervisory competence:

```python
SUPERVISORY_SCENARIO_PROMPT = """
Our supervisory procedures require:

{supervisory_procedure}

Generate scenarios from the supervisor's perspective:

1. REVIEW REQUIRED scenarios: Situations requiring supervisor 
   review before approval
2. RED FLAG scenarios: Issues the supervisor should catch during 
   review
3. APPROVAL scenarios: Clean submissions that can be approved
4. CONDITIONAL APPROVAL: Submissions requiring modifications
5. ESCALATION: Issues requiring escalation beyond supervisor

For each scenario, present it as the supervisor would receive it 
(e.g., a trade ticket, a client communication for review, a 
compliance form) and indicate the correct supervisory response.
"""
```

**Example:**

*Supervisory Procedure:*
```
All correspondence containing performance information must be 
reviewed by a principal before distribution. Principals must verify:
- Performance figures are accurate and from approved sources
- Appropriate time periods are shown
- Required disclosures are included
- Comparisons use appropriate benchmarks
```

*Generated Supervisor Review Scenario:*
```
SUBMITTED FOR REVIEW:

From: J. Smith, Financial Advisor
To: Client A
Subject: Your Portfolio Update

"Hi [Client],

Great news on your portfolio! Here are the highlights:

- Your account: +12.3% YTD
- S&P 500: +8.1% YTD  

You're beating the market by over 4%! Let me know if you want to 
discuss adding more aggressive positions to keep this momentum 
going.

Best,
J. Smith"

---

SUPERVISOR REVIEW CHECKLIST:

✓ Performance figures present: Yes (12.3% account, 8.1% S&P 500)
✗ Accuracy verified: NOT YET - must verify against system records
✗ Appropriate benchmark: ISSUE - S&P 500 may not be appropriate 
  benchmark for this client's allocation
✗ Required disclosures: MISSING - no past performance disclaimer
✗ Time period clarity: PARTIAL - "YTD" specified but no as-of date
✗ Inappropriate language: YES - "beating the market" and "momentum" 
  are problematic

CORRECT SUPERVISOR ACTION: 
Do NOT approve. Return to advisor with required corrections:
1. Add past performance disclaimer
2. Clarify benchmark appropriateness or use blended benchmark
3. Add specific as-of date
4. Remove "beating the market" language
5. Remove "momentum" reference (implies chasing performance)
6. Verify performance figures against approved source

ESCALATION: Not required (correctable issues), but pattern should 
be noted for advisor coaching.
```

### 4.7 Code of Ethics Prohibition Scenarios

#### 4.7.1 Personal Trading and Conflicts

Code of Ethics documents contain personal conduct prohibitions. Generate scenarios testing these boundaries:

```python
CODE_OF_ETHICS_PROMPT = """
Our Code of Ethics includes these prohibitions:

{code_prohibitions}

For each prohibition, generate:

1. CLEAR VIOLATION: Obvious breach of the prohibition
2. TECHNICAL VIOLATION: Letter of the rule violated but spirit 
   arguably intact
3. SPIRIT VIOLATION: Letter of rule followed but spirit violated
4. NEAR MISS: Situation that appears problematic but is actually 
   compliant
5. GRAY AREA: Situation requiring judgment call

These scenarios should feel like real situations employees might 
encounter or rationalize. Include the internal dialogue or 
rationalization that might lead to the violation.
"""
```

**Example:**

*Code Prohibition:*
```
Employees may not purchase securities for personal accounts within 
7 days before or after the firm purchases the same security for 
client accounts (blackout period).
```

*Generated Scenarios:*
```
CLEAR VIOLATION:
Situation: Analyst learns firm will buy XYZ Corp for multiple 
client accounts Monday. Friday afternoon, analyst buys XYZ in 
personal account.

Rationalization: "The firm's purchase won't move the price anyway, 
and I'm buying such a small amount it doesn't matter."

Violation: Textbook front-running violation within blackout period.

---

TECHNICAL VIOLATION:
Situation: Employee sells ABC Corp from personal account on Tuesday. 
Employee then recommends ABC Corp purchase to client on Friday, 
which is executed same day. Personal sale was before 
recommendation existed.

Issue: The sale preceded the client purchase by less than 7 days. 
While employee sold (not bought) and recommendation came after 
personal trade, the proximity creates appearance issues.

Analysis: Technical violation of blackout period, though direction 
(selling before client buys) makes front-running concern less 
applicable. Still reportable and reviewable.

---

SPIRIT VIOLATION:
Situation: Employee uses spouse's brokerage account (not subject 
to firm preclearance) to trade securities. On Monday, tells spouse 
to buy XYZ. On Wednesday, firm buys XYZ for clients.

Rationalization: "It's my spouse's account, not mine. The rule 
applies to my personal accounts."

Issue: While technically the employee's "personal account" wasn't 
used, the spirit of the rule (preventing conflicts and front-
running) is clearly violated through the spouse.

---

NEAR MISS (Compliant):
Situation: Employee sells XYZ from personal account. 9 days later, 
firm buys XYZ for clients.

Analysis: Outside 7-day blackout period. While optics might raise 
questions, this is compliant with the rule. Employee should 
maintain records showing sale date and lack of knowledge of 
upcoming firm purchase.

---

GRAY AREA:
Situation: Employee holds XYZ in personal account (purchased years 
ago). Firm decides to add XYZ to client accounts. Employee takes 
no action with personal holding.

Question: Is holding (not buying) during the blackout period a 
violation? What if employee was considering selling but delayed 
sale until after client purchases?

Analysis: Passive holding generally not a violation. However, 
decision to NOT sell based on knowledge of upcoming firm purchase 
could be problematic. Recommend disclosure to compliance for 
documentation even though likely compliant.
```

### 4.8 Cross-Referencing Internal Documents

#### 4.8.1 Building a Compliance Knowledge Graph

Link related policies, procedures, and guidance to generate comprehensive scenarios:

```python
CROSS_REFERENCE_PROMPT = """
These internal documents are related:

Document 1 (Policy): {policy_text}
Document 2 (Procedure): {procedure_text}
Document 3 (Training Example): {training_text}

Generate scenarios that:
1. Test understanding of how policy translates to procedure
2. Show where policy and procedure might conflict
3. Demonstrate the training example in different contexts
4. Identify gaps where policy exists but procedure is unclear

Create realistic scenarios an employee might encounter where they 
need to apply knowledge from multiple documents.
"""
```

### 4.9 Version-Aware Scenario Generation

#### 4.9.1 Handling Policy Changes

When policies are updated, generate scenarios that test awareness of changes:

```python
POLICY_CHANGE_PROMPT = """
Policy Update:

OLD POLICY:
{old_policy_text}

NEW POLICY:
{new_policy_text}

CHANGE SUMMARY:
{change_description}

EFFECTIVE DATE: {effective_date}

Generate scenarios that:
1. Would have been compliant under old policy but non-compliant 
   under new policy
2. Were non-compliant under old policy but now compliant
3. Test employee awareness of the change
4. Show transition period handling (what about actions taken just 
   before/after effective date?)
5. Test understanding of WHY the policy changed

Include dates in scenarios to make transition timing clear.
"""
```

### 4.10 Volume and Coverage Targets

#### 4.10.1 Recommended Scenario Generation Volumes

| Document Type | Documents | Scenarios per Doc | Total Target |
|--------------|-----------|-------------------|--------------|
| Core Policies | 20-30 | 30-50 | 600-1,500 |
| Procedures | 40-60 | 15-25 | 600-1,500 |
| Training Materials | 10-20 | 50-100 | 500-2,000 |
| Audit Findings | 15-30 | 8-12 | 120-360 |
| Escalation Matrices | 5-10 | 25-40 | 125-400 |
| Code of Ethics | 1-2 | 100-150 | 100-300 |
| **Total** | **~100** | **Varies** | **2,000-6,000** |

#### 4.10.2 Coverage Validation Checklist

Ensure generated scenarios cover:
- [ ] All major policy areas
- [ ] All violation types in your taxonomy
- [ ] All severity levels (minor, moderate, serious, critical)
- [ ] All communication channels (chat, email, phone, in-person)
- [ ] All relevant actor types (advisor, supervisor, client, vendor)
- [ ] Both compliant and non-compliant examples for each rule
- [ ] Edge cases and ambiguous situations
- [ ] Historical issues identified in audits
- [ ] Emerging risks identified by compliance

---

## 5. Teaching Model Reasoning

### 5.1 Why Reasoning Matters

A compliance model that outputs "non-compliant" without explanation is nearly useless in production. Effective compliance models must:

1. **Support Regulatory Audits:** Regulators expect documented rationale for compliance decisions
2. **Build Trust:** Compliance officers won't rely on black-box judgments
3. **Enable Improvement:** Reasoning errors are diagnosable and correctable
4. **Handle Edge Cases:** Reasoning allows principled handling of novel situations
5. **Support Appeals:** When flags are contested, reasoning provides basis for review

### 5.2 Anatomy of Strong Compliance Reasoning

**Five-Component Reasoning Structure:**

```
1. IDENTIFICATION: What specific conduct/statement is being evaluated?
2. REGULATORY GROUNDING: What rule(s) apply to this conduct?
3. CONTEXT ANALYSIS: What factors inform the compliance assessment?
4. SEVERITY ASSESSMENT: How serious is the issue if it is a violation?
5. REMEDIATION: What should happen next?
```

**Example of High-Quality Reasoning:**

```
INPUT: "Just put everything in our aggressive growth fund - you 
won't regret it. It's been crushing it lately."

IDENTIFICATION: The advisor is recommending 100% allocation to 
an aggressive growth fund, using performance as the primary 
justification, without apparent consideration of client suitability.

REGULATORY GROUNDING: This implicates:
- FINRA Rule 2111 (Suitability) requiring reasonable basis, 
  customer-specific, and quantitative suitability
- Regulation Best Interest requiring recommendations in client's 
  best interest
- Firm policy requiring documented suitability analysis

CONTEXT ANALYSIS:
- No client profile information referenced in recommendation
- "Everything" suggests 100% concentration, inappropriate for 
  most profiles
- "Won't regret it" is close to a prohibited guarantee
- "Crushing it lately" references recent performance as justification, 
  which is inappropriate basis for suitability
- Aggressive growth unsuitable for many profiles (retirees, 
  conservative investors, short horizons)

SEVERITY ASSESSMENT: HIGH
- Potential for significant client harm from unsuitable concentration
- Multiple compliance failures in single statement
- Absence of suitability analysis, not just inadequate documentation

REMEDIATION:
- Supervisor review required
- If recommendation was acted upon, suitability review of client 
  account needed
- Advisor coaching on suitability requirements
- Consider whether pattern exists across advisor's book

CONCLUSION: NON-COMPLIANT - Suitability violation
```

### 5.3 Reasoning Generation Techniques

#### 5.3.1 Extracting Reasoning from Enforcement Actions

SEC/FINRA enforcement actions contain regulatory reasoning that can be adapted:

```python
ENFORCEMENT_REASONING_PROMPT = """
Extract the regulatory reasoning from this enforcement action:

{enforcement_action_text}

Structure the reasoning as:
1. What conduct occurred
2. What rules were violated
3. Why the conduct violated those rules
4. What factors affected severity
5. What remediation was required

Then adapt this reasoning into a template that could apply to 
similar violations detected in conversations.
"""
```

#### 5.3.2 Chain-of-Thought Training Data

Generate reasoning chains for training:

```python
COT_GENERATION_PROMPT = """
Generate compliance analysis with explicit chain-of-thought 
reasoning for this scenario:

Scenario: {scenario_text}

Your response should show your complete reasoning process:

Step 1: Identify the key statements or actions to evaluate
Step 2: Determine which regulations and policies apply
Step 3: Analyze each compliance factor
Step 4: Assess severity based on factors present
Step 5: Reach conclusion with confidence level
Step 6: Recommend appropriate next steps

Think carefully through each step before reaching your conclusion.
"""
```

### 5.4 Reasoning Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Reasoning-Classification Consistency | Does reasoning support conclusion? | >95% |
| Regulatory Citation Accuracy | Are cited rules correct and relevant? | >90% |
| Factor Completeness | Are all relevant factors addressed? | >85% |
| Specificity Score | Does reasoning reference specific scenario elements? | >80% |
| Expert Agreement on Reasoning | Do experts agree reasoning is sound? | >85% |

---

## 6. QA Pair Generation Strategies

### 6.1 QA Generation from Regulatory Actions (SEC/FINRA)

#### 6.1.1 Violation Identification QA

**Source Text Example:**
```
The SEC found that XYZ Advisers failed to adequately disclose 
conflicts of interest related to affiliated broker-dealer 
compensation arrangements. Specifically, the firm did not disclose 
that it received revenue sharing payments from mutual fund 
companies whose products it recommended.
```

**Generated QA Pairs:**
```
Q: Does this scenario indicate a conflict of interest disclosure 
violation?
A: Yes. The firm failed to disclose revenue sharing arrangements 
with mutual fund companies, which represents a material conflict 
of interest that should have been disclosed to clients under the 
Investment Advisers Act fiduciary duty and Form ADV requirements.

Q: What specific compliance violation is present in this scenario?
A: Failure to disclose conflicts of interest related to revenue 
sharing arrangements with recommended product providers, violating 
SEC Form ADV disclosure requirements and fiduciary duty.

Q: How should this conflict have been handled?
A: The firm should have disclosed in Form ADV Part 2A and/or 
separate written disclosure that it receives revenue sharing 
payments from fund companies whose products it recommends, 
explained that this creates an incentive to recommend those 
products, and potentially offered comparable alternatives 
without such arrangements.
```

### 6.2 Contrastive QA Pairs

**Structure:**
```
Given two conversations handling the same situation:

Conversation A: [Non-compliant version]
Conversation B: [Compliant version]

Q: Which conversation demonstrates proper compliance with [specific 
rule]? Explain the key differences.

A: [Detailed explanation identifying specific compliant vs. 
non-compliant elements]
```

### 6.3 Multi-Turn QA

```
Given this conversation:
Turn 1: [Initial exchange]
Turn 2: [Development]
Turn 3: [Potential violation]
Turn 4: [Response]

Q: At which point in this conversation does a potential compliance 
issue emerge? What should have happened differently?

A: [Turn-by-turn analysis with specific identification and 
alternative responses]
```

---

## 7. Data Weighting and Curriculum Design

### 7.1 Training Data Composition

| Data Category | Percentage | Rationale |
|--------------|------------|-----------|
| Internal Communications (Real) | 25-30% | Domain adaptation, natural language |
| Internal Policies (Transformed) | 15-20% | Institutional grounding, rule awareness |
| Labeled Violations (Real) | 10-15% | Ground truth, accuracy calibration |
| SEC/FINRA Enforcement | 10-15% | Regulatory reasoning, citation accuracy |
| Synthetic Contrastive Pairs | 15-20% | Coverage, decision boundaries |
| Synthetic Edge Cases | 5-10% | Robustness, ambiguity handling |
| Reasoning-Augmented | 10-15% | Explanation quality, justification |

### 7.2 Curriculum Learning Strategy

**Phase 1: Foundation (Epochs 1-2)**
- High weight on internal policies and procedures
- Focus on vocabulary and terminology
- Simple classification tasks

**Phase 2: Core Classification (Epochs 3-5)**
- Increase weight on labeled examples
- Add contrastive pairs
- Multi-class violation detection

**Phase 3: Reasoning (Epochs 6-8)**
- Add chain-of-thought examples
- Enforcement action reasoning
- Severity calibration

**Phase 4: Hardening (Epochs 9-10)**
- Edge cases and adversarial examples
- Ambiguous scenarios
- Confidence calibration

### 7.3 Class Balancing Strategy

Most communications are compliant, creating severe class imbalance:

**Approach:**
1. **Oversampling:** Duplicate violation examples 3-5x
2. **Synthetic Expansion:** Generate variations of rare violations
3. **Focal Loss:** Weight hard examples more heavily
4. **Stratified Batches:** Ensure each batch has minimum violation representation

**Target Distribution:**
- Training: 40% compliant, 60% non-compliant (oversampled)
- Validation: Reflect true distribution for realistic metrics
- Test: Reflect true distribution

---

## 8. Model Selection and Training Configuration

### 8.1 Recommended Base Models

| Model | Parameters | Strengths | Considerations |
|-------|-----------|-----------|----------------|
| Llama 3.1 8B | 8B | Good baseline, efficient | May need larger for complex reasoning |
| Mistral 7B v0.3 | 7B | Strong instruction following | Good starting point |
| Qwen 2.5 7B | 7B | Structured output, multilingual | Good for diverse client base |
| Phi-3 Medium | 14B | Strong reasoning | Newer, less community support |
| Llama 3.1 70B | 70B | Best reasoning | Requires significant compute |

### 8.2 Fine-Tuning Configuration

```python
training_config = {
    # Model settings
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "use_4bit_quantization": True,  # QLoRA for efficiency
    
    # LoRA settings
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    
    # Training settings
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_seq_length": 4096,
    
    # Optimization
    "optimizer": "paged_adamw_8bit",
    "lr_scheduler": "cosine",
    "gradient_checkpointing": True,
}
```

### 8.3 Task Formulation

**Option A: Classification with Reasoning**
```
<|system|>
You are a financial compliance analyst. Analyze conversations for 
regulatory violations. Provide your reasoning, then classify.
<|user|>
Analyze this conversation for compliance violations:

{conversation}
<|assistant|>
REASONING:
[Step-by-step analysis]

CLASSIFICATION: [Compliant/Non-Compliant]
CONFIDENCE: [High/Medium/Low]
VIOLATION_TYPES: [List if applicable]
SEVERITY: [Minor/Moderate/Serious/Critical]
REGULATORY_BASIS: [Relevant rules/regulations]
```

**Option B: Structured JSON Output**
```
<|system|>
You are a financial compliance analyst. Respond only with valid JSON.
<|user|>
{conversation}
<|assistant|>
{
  "reasoning": "...",
  "classification": "non-compliant",
  "confidence": 0.87,
  "violations": [
    {
      "type": "suitability",
      "description": "...",
      "severity": "moderate",
      "regulatory_basis": ["FINRA Rule 2111", "Reg BI"]
    }
  ],
  "recommended_action": "supervisor_review"
}
```

---

## 9. Evaluation Framework

### 9.1 Classification Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Precision (Violations) | >85% | Minimize false positives |
| Recall (Violations) | >90% | Catch actual violations |
| F1 Score | >87% | Balanced performance |
| AUC-ROC | >0.92 | Discrimination ability |
| Precision @ 90% Recall | >80% | Production-realistic |

### 9.2 Reasoning Quality Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Reasoning-Classification Consistency | >95% | Reasoning supports conclusion |
| Citation Accuracy | >90% | Correct regulatory references |
| Factor Completeness | >85% | All relevant factors addressed |
| Expert Agreement | >85% | Reasoning deemed sound by experts |
| Hallucination Rate | <5% | No invented facts/rules |

### 9.3 Operational Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Inference Latency (p95) | <2 seconds | Production speed |
| Throughput | >100 conv/min | Batch processing capacity |
| False Positive Rate | <15% | Reviewer workload management |
| Coverage | >95% | Proportion of conversations processed |

### 9.4 Test Set Construction

**Stratified Test Set:**
- 50% random sample from production communications
- 30% known violations (historical confirmed)
- 15% edge cases and ambiguous situations
- 5% adversarial/evasive communications

**Hold-out Strategy:**
- Time-based split: Train on older data, test on recent
- Source-based split: Hold out entire communication channels
- Author-based split: Ensure no author appears in both train and test

---

## 10. Implementation Roadmap

### Phase 1: Data Foundation (Weeks 1-4)

- [ ] Inventory internal compliance documents
- [ ] Set up data extraction pipelines
- [ ] Create document preprocessing pipeline
- [ ] Extract rules from policies
- [ ] Begin SEC/FINRA scraping and parsing
- [ ] Establish data quality standards

**Deliverable:** Structured corpus of 500+ source documents

### Phase 2: Initial Generation (Weeks 5-8)

- [ ] Configure generation models
- [ ] Generate scenarios from internal documents
- [ ] Create contrastive pairs from enforcement actions
- [ ] Implement quality filtering pipeline
- [ ] Human review sampling (10%)
- [ ] Iterate on generation prompts

**Deliverable:** 10,000+ validated synthetic examples

### Phase 3: Model Training (Weeks 9-12)

- [ ] Finalize training data mix
- [ ] Configure training infrastructure
- [ ] Run baseline experiments
- [ ] Implement curriculum learning
- [ ] Hyperparameter optimization
- [ ] Model selection

**Deliverable:** Trained model with >85% F1 on validation

### Phase 4: Evaluation & Refinement (Weeks 13-16)

- [ ] Construct held-out test sets
- [ ] Run comprehensive evaluation
- [ ] Expert review of model outputs
- [ ] Identify failure modes
- [ ] Generate targeted training data for gaps
- [ ] Retrain with improvements

**Deliverable:** Model meeting all target metrics

### Phase 5: Production Integration (Weeks 17-20)

- [ ] Optimize inference (quantization, batching)
- [ ] Integrate with surveillance systems
- [ ] Set up monitoring and alerting
- [ ] Shadow production (compare to human reviewers)
- [ ] Calibrate thresholds based on shadow results
- [ ] Staged rollout

**Deliverable:** Production-deployed compliance model

---

## Appendix A: Prompt Templates Library

### A.1 Rule Extraction Prompt
```
[Detailed prompt for extracting rules from policy documents]
```

### A.2 Scenario Generation Prompt
```
[Detailed prompt for generating compliant/non-compliant scenarios]
```

### A.3 Reasoning Augmentation Prompt
```
[Detailed prompt for adding chain-of-thought reasoning]
```

### A.4 Quality Evaluation Prompt
```
[Detailed prompt for LLM-as-judge quality filtering]
```

---

## Appendix B: Violation Taxonomy

**Category 1: Suitability and Best Interest**
- 1.1 Unsuitable recommendation
- 1.2 Inadequate client profiling
- 1.3 Concentration risk
- 1.4 Time horizon mismatch

**Category 2: Disclosure Failures**
- 2.1 Conflict of interest non-disclosure
- 2.2 Fee/compensation non-disclosure
- 2.3 Material risk non-disclosure
- 2.4 Form ADV/CRS violations

**Category 3: Prohibited Conduct**
- 3.1 Performance guarantees
- 3.2 Unauthorized trading
- 3.3 Front-running
- 3.4 Cherry-picking
- 3.5 Churning

**Category 4: Communication Violations**
- 4.1 Misleading statements
- 4.2 Omission of material facts
- 4.3 Improper performance claims
- 4.4 Unapproved communications

**Category 5: Supervision Failures**
- 5.1 Inadequate review
- 5.2 Failure to escalate
- 5.3 Documentation deficiencies
- 5.4 Delegation failures

---

## Appendix C: Estimated Costs

| Item | Low Estimate | High Estimate |
|------|--------------|---------------|
| Data acquisition & processing | $5,000 | $15,000 |
| Synthetic generation compute | $3,000 | $8,000 |
| Human annotation & review | $10,000 | $25,000 |
| Training compute (GPU) | $5,000 | $15,000 |
| Evaluation & iteration | $2,000 | $7,000 |
| Infrastructure setup | $3,000 | $10,000 |
| **Total** | **$28,000** | **$80,000** |

*Note: Costs assume cloud compute and can be reduced significantly with on-premise GPU resources.*

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Initial document |
| 2.0 | Enhanced | Added data-to-capability mapping, detailed generation pipeline, internal document scenario generation |

---

*This document provides a comprehensive framework for building a financial compliance SLM. The approach emphasizes both classification accuracy and reasoning quality, recognizing that in regulated financial services, explainability is not optional but essential.*
