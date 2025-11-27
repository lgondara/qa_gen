# Fine-Tuning a Small Language Model for Financial Compliance Detection

## Executive Summary

This document outlines a comprehensive approach to fine-tuning a small language model (SLM) for detecting financial compliance violations in text-based communications. The approach leverages multiple data sources—internal company documents, historical communications, and publicly available regulatory data—to create a robust training dataset through synthetic QA generation and scenario creation.

---

## 1. Data Sources Overview

### 1.1 Internal Company Compliance Documents

**Description:**
- Compliance policies and procedures
- Internal audit reports
- Risk management frameworks
- Regulatory filing templates
- Training materials and handbooks
- Previous compliance review reports
- Escalation procedures and decision trees

**Importance:**
- **Highest priority** for establishing company-specific compliance standards
- Provides ground truth for what your organization considers compliant vs. non-compliant
- Contains nuanced institutional knowledge and risk tolerance levels
- Reflects industry-specific regulations (investment management, broker-dealer operations, etc.)

**Volume Estimation:** 100-1,000 documents depending on organization size

**Data Quality Considerations:**
- May contain outdated policies (version control critical)
- Varying levels of specificity and technical depth
- Potential inconsistencies across departments
- May lack negative examples (non-compliant scenarios)

### 1.2 Internal Historical Communications

**Description:**
- Recorded calls (transcribed)
- Chat logs (Bloomberg, Teams, Slack, etc.)
- Email threads (where permissible)
- Meeting transcripts
- Advisor-client interactions

**Importance:**
- **Critical for real-world pattern recognition**
- Captures actual language, terminology, and communication patterns
- Contains both compliant and potentially non-compliant examples
- Provides context on how violations occur in practice
- Reflects organizational communication culture

**Volume Estimation:** Thousands to millions of interactions

**Data Quality Considerations:**
- Privacy and confidentiality concerns (PII, client data)
- Need for de-identification/anonymization
- May have sparse labels for compliance violations
- Imbalanced dataset (most communications are compliant)
- Quality varies (background noise in calls, typos in chats)

### 1.3 SEC Enforcement Actions

**Description:**
- Enforcement releases and litigation releases
- Administrative proceedings
- Cease-and-desist orders
- SEC examination findings
- Form ADV disclosures
- Comment letters and response

**Public Sources:**
- SEC.gov Enforcement Actions: https://www.sec.gov/litigation/litreleases.htm
- SEC Administrative Proceedings: https://www.sec.gov/litigation/admin.htm
- Investment Adviser Public Disclosure (IAPD)

**Importance:**
- **Gold standard for regulatory precedent**
- Provides detailed descriptions of violations
- Includes regulatory reasoning and legal standards
- Shows severity and consequences
- Covers wide range of violation types

**Volume Estimation:** 700-1,000 new actions per year; extensive historical archive

**Data Quality Considerations:**
- High-quality, professionally written
- Dense legal language
- May not reflect all violation patterns (only those prosecuted)
- Lag time between violation and public disclosure

### 1.4 FINRA Enforcement Actions

**Description:**
- Disciplinary actions against firms and individuals
- Arbitration awards
- Regulatory notices
- Examination findings letters
- Rule violations database

**Public Sources:**
- FINRA Disciplinary Actions: https://www.finra.org/rules-guidance/oversight-enforcement/disciplinary-actions
- FINRA BrokerCheck for individual disciplinary history

**Importance:**
- **Highly relevant for broker-dealer compliance**
- More granular than SEC for certain violations
- Covers supervisory failures, communication violations, suitability issues
- Includes monetary penalties and sanctions

**Volume Estimation:** 400-600 actions per year

**Data Quality Considerations:**
- Similar to SEC data quality
- More focused on specific violation categories
- May include redacted information

### 1.5 Other Regulatory Sources

**Description:**
- CFTC enforcement (for firms with derivatives/futures)
- State securities regulators
- DOJ enforcement (criminal cases)
- OFAC sanctions lists
- Industry guidance from investment associations

**Importance:**
- **Supplementary but important for comprehensive coverage**
- Fills gaps in federal enforcement
- Provides cross-jurisdictional perspective

---

## 2. Data Source Prioritization Matrix

| Data Source | Volume | Quality | Specificity | Label Density | Priority |
|-------------|--------|---------|-------------|---------------|----------|
| Internal Policies | Medium | High | Highest | Medium | **Critical** |
| Internal Comms | Very High | Medium | Highest | Low | **Critical** |
| SEC Actions | High | Very High | High | High | **High** |
| FINRA Actions | Medium | Very High | High | High | **High** |
| Other Regulatory | Medium | High | Medium | High | **Medium** |

---

## 3. Synthetic QA Pair Generation Strategy

### 3.1 QA Generation from Regulatory Actions (SEC/FINRA)

**Approach 1: Violation Identification**

**Source Text Example:**
```
The SEC found that XYZ Advisers failed to adequately disclose conflicts 
of interest related to affiliated broker-dealer compensation arrangements. 
Specifically, the firm did not disclose that it received revenue sharing 
payments from mutual fund companies whose products it recommended.
```

**Generated QA Pairs:**
```
Q: Does this communication indicate a conflict of interest disclosure violation?
A: Yes. The firm failed to disclose revenue sharing arrangements with mutual 
fund companies, which represents a material conflict of interest that should 
have been disclosed to clients under the Investment Advisers Act.

Q: What specific compliance violation is present in this scenario?
A: Failure to disclose conflicts of interest related to revenue sharing 
arrangements with recommended product providers, violating SEC Form ADV 
disclosure requirements.

Q: Is this communication compliant with regulatory disclosure requirements?
A: No. This represents a clear disclosure violation under the Investment 
Advisers Act fiduciary duty.
```

**Approach 2: Violation Detection from Synthetic Scenarios**

Use an open-source LLM (Llama 3.1 70B, Mixtral 8x22B, or Qwen 2.5 72B) to generate scenarios based on violation patterns:

**Prompt Template:**
```
Based on the following SEC enforcement action, generate 5 realistic 
conversation snippets between a financial advisor and client that would 
represent similar compliance violations. Make them sound natural and 
conversational:

[Insert SEC action description]

For each snippet, also generate a compliant version that addresses the 
same topic appropriately.
```

**Example Output:**
```
Non-Compliant: "Don't worry about the fees - we have a special arrangement 
with this fund company that makes it better for you."

Compliant: "This fund has an expense ratio of 0.85%. I should disclose 
that our firm receives 12b-1 fees from this fund family, which creates 
a conflict of interest you should be aware of when making your decision."
```

### 3.2 QA Generation from Internal Compliance Policies

**Approach 1: Policy-to-Scenario Conversion**

**Source Policy:**
```
All recommendations must be suitable based on the client's risk tolerance, 
time horizon, and financial situation. Representatives must document the 
basis for suitability in the client file.
```

**Generated QA Pairs:**
```
Q: Analyze this advisor statement for compliance: "Given your retirement 
is 30 years away and your aggressive risk tolerance, I'm recommending 
an 80% equity allocation."
A: Compliant. The advisor references the client's time horizon (30 years) 
and risk tolerance (aggressive), demonstrating suitability analysis.

Q: Analyze this advisor statement for compliance: "You should put 
everything in this high-yield bond fund - the returns are great."
A: Non-compliant. No documentation of suitability factors such as risk 
tolerance, time horizon, or financial situation. Makes absolute 
recommendation without proper justification.
```

**Approach 2: Red Flag Identification**

Generate QA pairs that test for specific red flag phrases:

```
Q: Does this phrase raise compliance concerns: "This is a guaranteed return"
A: Yes. Major red flag. Using "guaranteed" for investment returns is 
prohibited under SEC and FINRA rules unless backed by explicit guarantee 
(e.g., FDIC insurance, fixed annuity contract).

Q: Is this communication appropriate: "Past performance shows this fund 
consistently outperforms"
A: Potentially non-compliant. Must include disclaimer that past performance 
does not guarantee future results. The word "consistently" could be 
misleading without proper context.
```

### 3.3 Contrastive Pair Generation from Historical Communications

**This is the most critical component for training a robust compliance model.** Real internal communications provide authentic language patterns, actual violation structures, and company-specific communication styles that synthetic data cannot replicate. Contrastive pairs—where slight modifications change compliance status—train the model to identify the precise boundaries of acceptable communication.

#### 3.3.1 Philosophy of Contrastive Learning for Compliance

Contrastive pairs force the model to learn **discriminative features** rather than superficial correlations. A model trained only on clearly compliant vs. clearly non-compliant examples may learn shortcuts (e.g., "if text contains numbers, flag it"). Contrastive pairs with minimal differences teach the model what actually matters:

- The presence vs. absence of required disclosures
- Absolute statements vs. qualified statements
- Documented suitability vs. undocumented recommendations
- Disclosed conflicts vs. undisclosed conflicts

**Key Principle:** The smaller the difference between compliant and non-compliant pairs, the more valuable the training signal.

---

#### 3.3.2 Types of Contrastive Pairs from Real Data

**Type 1: Minimal Edit Pairs (Single Critical Change)**

Take actual communications and make the smallest possible edit that changes compliance status.

**Real Communication (Compliant):**
```
"Based on your 20-year time horizon and moderate risk tolerance we discussed 
last month, I'm recommending a 60/40 equity/bond allocation. This aligns with 
the investment policy statement you signed. Please note that our firm receives 
12b-1 fees from some of the mutual funds in this portfolio."
```

**Generated Non-Compliant Version (Remove disclosure):**
```
"Based on your 20-year time horizon and moderate risk tolerance we discussed 
last month, I'm recommending a 60/40 equity/bond allocation. This aligns with 
the investment policy statement you signed."
```

**Generated Non-Compliant Version (Remove suitability basis):**
```
"I'm recommending a 60/40 equity/bond allocation. Please note that our firm 
receives 12b-1 fees from some of the mutual funds in this portfolio."
```

**Training Value:** Model learns that removing specific elements (conflict disclosure, suitability documentation) creates violations.

---

**Type 2: Substitution Pairs (Word/Phrase Swaps)**

Replace compliant language with problematic language while keeping structure identical.

**Real Communication (Compliant):**
```
"Historical data shows this fund has averaged 7% annual returns over the past 
decade, though past performance does not guarantee future results."
```

**Substitution Variations:**
| Original | Substitution | Compliance |
|----------|--------------|------------|
| "averaged 7%" | "consistently delivers 7%" | Non-compliant (implies reliability) |
| "past performance does not guarantee" | "past performance indicates" | Non-compliant (removes disclaimer) |
| "averaged 7%" | "is expected to deliver 7%" | Non-compliant (forward-looking statement) |
| "this fund has averaged" | "you will get" | Non-compliant (promise) |

**Generation Strategy:** 
- Identify high-risk phrases in compliant communications
- Create systematic substitutions with known problematic alternatives
- Generate 5-10 variations per base communication

---

**Type 3: Addition/Deletion Pairs**

Test what happens when required elements are present vs. absent.

**Real Communication Starting Point:**
```
"I recommend moving $50,000 into this emerging markets fund."
```

**Compliant Additions (what makes it acceptable):**
```
+ "Based on our discussion of your aggressive risk tolerance and 15+ year time horizon"
+ "Please review the prospectus, which details the higher volatility of emerging markets"
+ "This represents approximately 10% of your total portfolio, within your stated allocation limits"
+ "I should disclose that I receive higher compensation for this fund family"
```

**Training Approach:**
- Present base communication (non-compliant)
- Generate versions with each required element added individually
- Generate version with all elements (fully compliant)
- Model learns which elements are necessary and in what combinations

---

**Type 4: Context-Dependent Pairs**

Identical language with different surrounding context that changes compliance status.

**Communication:** "Given market volatility, you might want to move to cash."

**Compliant Context:**
```
Client Context: High net worth client, 80 years old, in drawdown phase, 
previously documented ultra-conservative risk tolerance, existing 90% bond 
portfolio, documented liquidity needs for medical expenses.

Full Communication: "Given the market volatility we're seeing and your 
documented need for liquidity for your upcoming medical procedures, you 
might want to move the portion allocated for those expenses to cash or a 
money market fund. This aligns with the conservative approach we've established 
for your near-term needs while keeping your longer-term investments positioned 
as discussed."
```

**Non-Compliant Context:**
```
Client Context: 35-year-old client, moderate-aggressive risk profile, 
30-year time horizon, 401(k) retirement account, no documented liquidity needs.

Full Communication: "Given market volatility, you might want to move to cash."
```

**Training Value:** Model learns that the same recommendation requires different justifications based on client circumstances.

---

#### 3.3.3 Systematic Contrastive Generation Pipeline

**Step 1: Identify Candidate Communications**

From your internal historical data, select communications based on:

**Positive Selection Criteria (likely to be useful for contrastive generation):**
- Contains product recommendations or investment advice
- Discusses performance, returns, or expectations
- References fees, compensation, or firm relationships
- Includes suitability discussions
- Contains numerical projections or comparisons
- Discusses risks or lack thereof

**Filter Out:**
- Purely administrative messages (meeting scheduling, etc.)
- Client questions without advisor responses
- System-generated communications
- Communications already flagged by compliance (use these separately as true positives)

**Recommended Volume:** 10,000-50,000 candidate communications

---

**Step 2: Compliance Assessment of Base Communications**

Use a large open-source LLM to perform initial compliance assessment:

**Assessment Prompt Template:**
```
You are a senior compliance officer at a regulated investment advisory firm. 
Review the following communication for compliance with SEC and FINRA regulations.

COMMUNICATION:
{text}

AVAILABLE CONTEXT:
{any available context about client, advisor, timing}

Provide your assessment in the following format:

COMPLIANCE STATUS: [Compliant/Non-Compliant/Ambiguous]
CONFIDENCE: [0-100]

If Non-Compliant or Ambiguous:
PRIMARY ISSUES:
- [List specific violations or concerns]

MISSING ELEMENTS:
- [What should be present but isn't]

PROBLEMATIC PHRASES:
- [Specific language that creates issues]

If Compliant:
KEY COMPLIANT ELEMENTS:
- [What makes this acceptable]

CRITICAL LANGUAGE:
- [Phrases that could become violations if modified]
```

**Post-Assessment Filtering:**
- High-confidence compliant (>80): Use for contrastive generation
- High-confidence non-compliant (>80): Use as-is for training, also use for contrastive generation
- Medium confidence (50-80): Flag for human review
- Low confidence (<50): Set aside for expert annotation

---

**Step 3: Automated Contrastive Generation**

For high-confidence compliant communications, generate non-compliant versions:

**Generation Prompt Template:**
```
You are generating training data for a compliance detection model. Given the 
following COMPLIANT communication, generate 5 variations that would be 
NON-COMPLIANT. Each variation should change as little as possible while 
creating a regulatory violation.

COMPLIANT COMMUNICATION:
{compliant_text}

For each variation, specify:
1. What was changed (be specific)
2. Why this creates a violation
3. Which regulation is violated

VARIATION 1:
Modified Text: [...]
Change Made: [...]
Violation Created: [...]
Regulation: [...]

[Repeat for variations 2-5]

Requirements:
- Make minimal changes (1-3 words if possible, max 1 sentence)
- Each variation should demonstrate a different type of violation
- Keep the same communication style and context
- Focus on the most common violation types: guarantees, omitted disclosures, 
  suitability failures, conflicts of interest, misleading performance claims
```

For high-confidence non-compliant communications, generate compliant versions:

**Remediation Prompt Template:**
```
You are a compliance officer helping to correct a problematic communication. 
Given the following NON-COMPLIANT communication and its identified issues, 
generate 3 corrected versions that would be compliant.

NON-COMPLIANT COMMUNICATION:
{non_compliant_text}

IDENTIFIED ISSUES:
{issues_from_assessment}

For each corrected version, specify:
1. What was added/changed
2. How this addresses the compliance issue
3. Alternative approaches to compliance

CORRECTED VERSION 1:
Modified Text: [...]
Changes Made: [...]
Issues Addressed: [...]

[Repeat for versions 2-3]

Requirements:
- Preserve the core message/intent where possible
- Add necessary disclosures, qualifications, or documentation references
- Remove or modify problematic language
- Make realistic changes an actual advisor would make
```

---

**Step 4: Quality Control and Validation**

After automated generation, implement multi-stage validation:

**Automated Checks:**
- Verify text actually changed between pairs
- Confirm change magnitude (Levenshtein distance, word-level diff)
- Check for nonsensical edits or grammar errors
- Validate JSON/structured output parsing

**LLM-Based Review:**
Use a different model to verify the generated pairs:

```
Review the following contrastive pair for quality:

ORIGINAL: {original_text}
STATUS: {original_status}

MODIFIED: {modified_text}  
STATUS: {modified_status}

Questions:
1. Does the modification actually change compliance status as indicated? (Yes/No)
2. Is the change minimal and focused? (Yes/No)
3. Is the reasoning for the status change accurate? (Yes/No)
4. Rate overall quality: (Low/Medium/High)

If any answer is No or quality is Low, explain the issue.
```

**Human Review Sampling:**
- Review 500-1,000 randomly sampled pairs (5-10% of generated data)
- Focus on pairs where LLM-based review flagged concerns
- Track inter-rater reliability between human reviewers
- Create guidelines document based on common issues found

---

#### 3.3.4 Advanced Contrastive Techniques

**Technique 1: Graduated Severity Chains**

Create chains of communications with increasing violation severity:

```
Level 0 (Fully Compliant):
"Based on your moderate risk tolerance, I recommend a balanced fund. The 
prospectus contains important information about risks and fees, which we 
should review together."

Level 1 (Minor Technical Issue):
"Based on your moderate risk tolerance, I recommend a balanced fund. The 
prospectus contains important information about risks and fees."
[Missing: explicit review commitment]

Level 2 (Moderate Violation):
"Based on your risk tolerance, I recommend a balanced fund. This has been 
performing well historically."
[Missing: disclosure direction, adds past performance without disclaimer]

Level 3 (Serious Violation):
"I recommend a balanced fund. This has consistently delivered strong returns 
and should continue to do so."
[Missing: suitability basis, problematic performance language, forward-looking statement]

Level 4 (Severe Violation):
"You should definitely invest in this balanced fund. It's guaranteed to give 
you solid returns, and my firm gets paid well for it so you know it's good."
[Multiple severe violations: guarantee, absolute recommendation, undisclosed conflict with improper framing]
```

**Training Value:** Model learns to assess violation severity, not just binary classification.

---

**Technique 2: Multi-Stakeholder Contrastive Pairs**

Generate pairs that differ based on who is communicating:

**Scenario:** Discussing fund performance

**Registered Representative (FINRA jurisdiction) - Stricter Rules:**
```
Non-Compliant: "This fund has beaten its benchmark for 5 straight years."
Compliant: "According to the fund's fact sheet, this fund has outperformed 
its stated benchmark in 5 of the last 5 calendar years. Past performance does 
not guarantee future results. Rankings and performance data are available in 
the prospectus."
```

**Investment Adviser (SEC jurisdiction) - Fiduciary Standard:**
```
Non-Compliant: "This fund consistently outperforms."
Compliant: "This fund has outperformed its benchmark over the trailing 5-year 
period. However, it has higher fees than comparable funds, which I must disclose 
as it represents a cost to you. Past performance does not guarantee future 
results."
```

**Training Value:** Model learns jurisdiction-specific requirements.

---

**Technique 3: Temporal Contrastive Pairs**

Same communication at different points in time:

**Pre-Disclosure (Non-Compliant):**
```
"I recommend allocating 20% to international equities."
```

**Post-Disclosure (Compliant):**
```
"Now that we've reviewed your risk tolerance, time horizon, and you've 
received the Form ADV discussing our fee structure and conflicts of interest, 
I recommend allocating 20% to international equities based on our discussion."
```

**Training Value:** Model learns that compliance depends on procedural steps, not just content.

---

**Technique 4: Cross-Channel Contrastive Pairs**

Same content, different communication medium:

**Content:** Information about fund returns

**Marketing Material (Highly Regulated):**
```
Non-Compliant: "15% average returns over 5 years"
Compliant: [Must include: standardized performance data, benchmark comparison, 
time periods, disclaimers, fee impact, ranking methodology if applicable, etc.]
```

**One-on-One Conversation (Less Formal):**
```
Compliant: "The fund has averaged about 15% over the past 5 years, though 
past performance doesn't guarantee future results. We should look at this in 
context of your overall portfolio and goals."
```

**Training Value:** Model learns medium-specific requirements.

---

#### 3.3.5 Leveraging Compliance Review History

**Gold Mine: Previously Flagged Communications**

Your internal compliance review history is invaluable:

**What to Extract:**
- Original communication that was flagged
- Compliance officer's notes on the issue
- Revised compliant version (if available)
- Resolution outcome (warning, fine, training, etc.)

**Contrastive Pair Generation:**

**Flagged Original:** "This investment should do really well for you."

**Compliance Notes:** "Absolute recommendation without suitability basis. 
'Should' implies guarantee. No risk disclosure."

**Approved Revision:** "Based on your investment objectives and risk tolerance 
that we documented, this investment may be appropriate for your portfolio. 
However, all investments carry risks, including potential loss of principal."

**Generated Training Data:**
- Original + Non-Compliant label + Violation types
- Revision + Compliant label
- Variations showing each issue fixed individually

---

**Pattern Extraction from Compliance Flags:**

Analyze your compliance flag history to identify:
- Most common violation types
- Language patterns that frequently get flagged
- Advisor-specific patterns
- Product-type-specific issues
- Temporal trends (regulatory changes over time)

**Use Cases:**
1. **Targeted Generation:** Create synthetic examples of underrepresented violations
2. **Augmentation Focus:** Generate more variations of high-frequency violation patterns
3. **Advisor Training:** Identify individual communication weaknesses
4. **Policy Refinement:** Spot gaps in current compliance policies

---

#### 3.3.6 Contrastive Dataset Composition

**Recommended Distribution:**

| Pair Type | % of Contrastive Data | Estimated Count |
|-----------|----------------------|-----------------|
| Minimal Edit (1-3 words) | 30% | 4,500 |
| Phrase Substitution | 25% | 3,750 |
| Addition/Deletion | 20% | 3,000 |
| Context-Dependent | 15% | 2,250 |
| Severity Chains | 10% | 1,500 |

**Total Contrastive Pairs: ~15,000 (15% of overall dataset)**

---

#### 3.3.7 Quality Metrics for Contrastive Pairs

**Edit Distance:**
- Target: 1-10 words changed for minimal pairs
- Max: 30% of original text for major structural changes
- Track distribution: aim for concentration at lower edit distances

**Semantic Similarity:**
- Use sentence embeddings (e.g., all-mpnet-base-v2)
- Cosine similarity should be >0.85 for minimal pairs
- Ensures pairs are truly contrastive, not completely different communications

**Violation Type Distribution:**
- Ensure each major violation type has sufficient contrastive coverage
- Track: guarantees, suitability, disclosure, misrepresentation, conflicts
- Minimum 500 pairs per major violation category

**Human Agreement Rate:**
- Sample 500 pairs for expert review
- Measure: % where expert agrees with assigned labels
- Target: >90% agreement for high-quality dataset

---

#### 3.3.8 Common Pitfalls and How to Avoid Them

**Pitfall 1: Over-Obvious Edits**
- Bad: "This fund is great" → "This fund is guaranteed to be great"
- Better: "This fund has performed well" → "This fund consistently performs well"

**Pitfall 2: Changing Multiple Things**
- Bad: Changing topic + adding violation language + removing disclosure
- Better: Single focused change that creates violation

**Pitfall 3: Unrealistic Language**
- Bad: "I hereby affirm that I have violated SEC Rule 206(4)-1"
- Better: Natural language that implicitly creates violation

**Pitfall 4: Ignoring Context**
- Bad: Creating pairs without considering client profile, timing, prior communications
- Better: Include relevant context fields in data structure

**Pitfall 5: Synthetic-Only Contrastive Pairs**
- Bad: All pairs generated from scratch by LLM
- Better: Start with real communications, then modify

---

#### 3.3.9 Integration with Overall Training Strategy

**Curriculum Learning Approach:**

**Phase 1:** Train on clear-cut compliant vs. non-compliant
**Phase 2:** Introduce contrastive pairs with medium edit distance
**Phase 3:** Add minimal contrastive pairs and edge cases
**Phase 4:** Add severity-graded and context-dependent pairs

**Loss Function Considerations:**
- Contrastive pairs may benefit from different loss weighting
- Consider margin-based or triplet loss for better discrimination
- Higher weight on minimal pairs to force fine-grained learning

**Evaluation Split:**
- Ensure test set has representative contrastive pair distribution
- Create separate test subset of ONLY contrastive pairs
- Measure model's ability to catch subtle violations specifically

---

## 4. Teaching Model Reasoning: Building Explainable Compliance Judgments

### 4.1 Why Reasoning Matters for Compliance Models

**Regulatory and Business Requirements:**
- **Auditability**: Compliance decisions must be traceable and defensible
- **Trust**: Advisors and compliance officers need to understand *why* something is flagged
- **Training**: Model explanations serve as educational material for staff
- **Appeal Process**: Clients and advisors may challenge flags, requiring clear justification
- **Model Risk Management**: SR 11-7 requires understanding of model decision-making

**Technical Benefits:**
- **Improved Accuracy**: Chain-of-thought reasoning helps models work through complex scenarios
- **Reduced False Positives**: Explicit reasoning forces consideration of context
- **Better Calibration**: Models that explain themselves tend to be more confident when correct
- **Failure Analysis**: Reasoning traces reveal where model logic breaks down

**Key Principle:** A compliance model that says "non-compliant" without explanation is nearly useless in production. The reasoning is as important as the classification.

---

### 4.2 Anatomy of Strong Compliance Reasoning

**Components of High-Quality Reasoning:**

1. **Identification**: What specific element(s) create the compliance issue?
2. **Regulatory Grounding**: Which rule, regulation, or standard is implicated?
3. **Context Analysis**: What contextual factors matter for this determination?
4. **Severity Assessment**: How serious is this violation?
5. **Remediation Guidance**: What would make this compliant?

**Example Structure:**

```
Communication: "This fund is guaranteed to outperform the market."

Classification: NON-COMPLIANT

Reasoning:
1. IDENTIFICATION: The phrase "guaranteed to outperform" creates two distinct issues:
   - Use of "guaranteed" for investment returns
   - Promise of market outperformance

2. REGULATORY GROUNDING:
   - SEC Rule 206(4)-1(a)(5): Prohibits false or misleading advertisements
   - FINRA Rule 2210: Communications must be fair and balanced
   - Investment Advisers Act Section 206: Fiduciary duty prohibits misleading statements

3. CONTEXT ANALYSIS:
   - No context provided where guarantee could be legitimate (e.g., FDIC insurance)
   - Absolute performance claim with no qualification or disclosure
   - No reference to past performance data or forward-looking statement disclaimer

4. SEVERITY: HIGH - Direct prohibition on guarantees for investment returns. This is 
   among the most serious communication violations.

5. REMEDIATION: Remove guarantee language and qualify performance claims:
   "This fund has outperformed its benchmark over the past 5 years. Past performance 
   does not guarantee future results. Please review the prospectus for complete 
   information about risks and fees."
```

---

### 4.3 Reasoning Templates by Scenario Type

#### 4.3.1 Suitability Violations

**Reasoning Pattern:**
```
1. IDENTIFICATION: Recommendation lacks documented suitability basis
2. ASSESSMENT: Required elements missing:
   - [ ] Risk tolerance documentation
   - [ ] Time horizon consideration  
   - [ ] Financial situation analysis
   - [ ] Investment objectives alignment
3. REGULATORY: FINRA Rule 2111 (Suitability), Investment Advisers Act fiduciary duty
4. CONTEXT: [Any mitigating or aggravating factors]
5. SEVERITY: [High if no suitability factors / Medium if partial / Low if minor gap]
```

**Example Application:**
```
Communication: "You should buy this aggressive growth fund."

Reasoning:
The recommendation "you should buy" is directive without documented suitability 
analysis. There is no reference to:
- Client's risk tolerance (essential for "aggressive growth" product)
- Time horizon (critical for equity volatility tolerance)
- Overall portfolio context
- Investment objectives discussion

This violates FINRA Rule 2111 which requires that recommendations be suitable based 
on customer-specific factors. The absolute nature ("should buy") without qualification 
makes this a HIGH severity violation. 

To remediate: "Based on your documented aggressive risk tolerance and 20+ year time 
horizon, this growth fund may be suitable for the equity portion of your portfolio. 
Let's discuss how this fits with your overall investment objectives."
```

#### 4.3.2 Disclosure Violations

**Reasoning Pattern:**
```
1. IDENTIFICATION: Required disclosure is absent or inadequate
2. DISCLOSURE TYPE: [Conflict of interest / Fee / Risk / Material fact]
3. REGULATORY: Form ADV Part 2, SEC Rule 206(4)-7, FINRA Rule 2210
4. MATERIALITY: Why this disclosure matters to client decision-making
5. CORRECTION: Specific disclosure language needed
```

#### 4.3.3 Performance Claims

**Reasoning Pattern:**
```
1. IDENTIFICATION: Performance statement requires analysis
2. FACTUAL BASIS: Is claim accurate? How is performance calculated?
3. DISCLAIMERS: Required language present/absent
4. PRESENTATION: Complies with GIPS standards? Time-weighted returns?
5. CONTEXT: Cherry-picked timeframe? Benchmark comparison? Survivorship bias?
6. REGULATORY: SEC Marketing Rule, FINRA Rule 2210
```

#### 4.3.4 Prohibited Guarantees

**Reasoning Pattern:**
```
1. IDENTIFICATION: Language suggesting guaranteed outcomes
2. EXPLICIT vs IMPLICIT: Direct guarantee vs. implied certainty
3. EXCEPTIONS: Is this a legitimately guaranteed product? (annuities, CDs, etc.)
4. SEVERITY: Guarantees are per se violations - automatically HIGH
5. REGULATORY: Investment Advisers Act 206, FINRA Rule 2210
```

---

### 4.4 Generating Reasoning for Training Data

#### 4.4.1 Reasoning for Regulatory Actions (SEC/FINRA)

**Approach:** Extract reasoning from enforcement action language

**Source Text:**
```
"The Commission found that Advisor failed to disclose that it received compensation 
from third parties based on client assets. This conflict of interest was material 
to clients' evaluation of the Advisor's recommendations. The Advisor violated 
Section 206(2) of the Investment Advisers Act."
```

**Generated Training Example with Reasoning:**
```
Communication: "I recommend allocating 40% to Fund XYZ for your portfolio."

Classification: INSUFFICIENT INFORMATION - REQUIRES CONTEXT

Reasoning:
This recommendation alone is incomplete for compliance assessment. The critical 
question is whether compensation-related conflicts of interest have been disclosed.

REGULATORY REQUIREMENT: Section 206(2) of the Investment Advisers Act requires 
disclosure of all material conflicts of interest, including:
- Receipt of 12b-1 fees from recommended funds
- Revenue sharing arrangements
- Affiliated product compensation
- Any financial incentive to recommend specific products

COMPLIANCE REQUIREMENTS:
1. Prior to or at the time of recommendation, client must receive Form ADV Part 2A 
   disclosing compensation arrangements
2. Verbal disclosure should reference written disclosures
3. Documentation that client acknowledged receipt

WITHOUT THESE ELEMENTS: Non-compliant (HIGH severity conflict of interest violation)
WITH PROPER DISCLOSURE: Compliant

Example compliant version: "I recommend 40% to Fund XYZ. As disclosed in the Form 
ADV you received, our firm receives 12b-1 fees from this fund family, which 
represents a conflict of interest you should consider."
```

#### 4.4.2 Reasoning for Contrastive Pairs

**For each contrastive pair, generate reasoning that explains the difference:**

**Compliant Version:**
```
Communication: "Based on your moderate risk tolerance documented last month, I 
recommend a 60/40 equity/bond allocation. Please note our firm receives 12b-1 
fees from some funds in this portfolio."

Classification: COMPLIANT

Reasoning:
This recommendation demonstrates proper compliance practices:

1. SUITABILITY: References documented risk tolerance with timeframe
2. DISCLOSURE: Explicit mention of firm compensation creating conflict of interest
3. PRESENTATION: Recommendation format is appropriate ("I recommend" with basis)

REGULATORY COMPLIANCE: Meets FINRA Rule 2111 (suitability) and Investment Advisers 
Act Section 206(2) (disclosure). The advisor grounds the recommendation in 
client-specific factors and proactively discloses material conflicts.

STRENGTH: The documentation reference ("last month") suggests proper recordkeeping.
```

**Non-Compliant Version (Removed Disclosure):**
```
Communication: "Based on your moderate risk tolerance documented last month, I 
recommend a 60/40 equity/bond allocation."

Classification: NON-COMPLIANT

Reasoning:
While this recommendation includes suitability documentation, it has a critical 
omission:

VIOLATION: Failure to disclose material conflict of interest (12b-1 fee receipt)

REGULATORY: Investment Advisers Act Section 206(2) requires disclosure of material 
conflicts that could affect client decision-making. Compensation from recommended 
products is always material.

SEVERITY: MEDIUM-HIGH. The suitability basis is proper, but the undisclosed conflict 
represents a fiduciary breach. This is particularly problematic because:
- Conflict is direct (compensation based on recommendation)
- Client cannot evaluate objectivity without disclosure
- SEC enforcement actions frequently target undisclosed compensation

REMEDIATION: Add disclosure: "Please note our firm receives 12b-1 fees from some 
funds in this portfolio, which represents a conflict of interest."
```

**Key Teaching Point:** The reasoning explicitly contrasts what makes one version compliant and the other not, forcing the model to learn the discriminative features.

---

### 4.5 Multi-Hop Reasoning for Complex Scenarios

**Scenario:** Context-dependent compliance where multiple factors interact

**Communication:** "Given recent volatility, you should move your retirement funds to cash."

**Step-by-Step Reasoning:**

```
STEP 1 - SURFACE ANALYSIS:
This appears to be investment advice to liquidate equity positions and move to cash.

STEP 2 - SUITABILITY FACTORS NEEDED:
To assess compliance, we must consider:
- Client's age and retirement timeline
- Documented risk tolerance  
- Current asset allocation
- Purpose of retirement funds (immediate needs vs. long-term growth)
- Previous investment policy discussions

STEP 3 - CONTEXT SCENARIO A (Non-Compliant):
Client Profile: 35 years old, 30-year time horizon, moderate-aggressive risk 
tolerance, 401(k) for retirement
Analysis: This recommendation is unsuitable market timing advice that contradicts 
the client's documented long-term investment strategy and risk tolerance. Moving 
long-term retirement funds to cash based on short-term volatility is typically 
inappropriate for clients with multi-decade horizons.
Violation: FINRA Rule 2111 (Suitability)
Severity: HIGH - Recommendation contradicts established investment policy

STEP 4 - CONTEXT SCENARIO B (Potentially Compliant):
Client Profile: 68 years old, retiring in 3 months, conservative risk tolerance, 
documented need for liquidity, previously discussed transition to capital preservation
Analysis: This recommendation may be appropriate given:
- Imminent retirement requiring cash reserves
- Conservative risk profile documented
- Previous discussions about de-risking
However, still requires: (1) reference to prior discussions, (2) acknowledgment that 
this is a change in strategy, (3) discussion of foregone returns and inflation risk
Compliance Status: POTENTIALLY COMPLIANT with proper documentation and disclosure
Required additions: "As we discussed in our retirement planning meeting, and given 
your need for liquidity starting next quarter..."

STEP 5 - SYNTHESIS:
Compliance determination depends entirely on context. The same recommendation can 
be a serious violation or appropriate advice. This demonstrates why:
- Suitability must be documented and client-specific
- Context cannot be omitted from compliance review
- Models must account for customer profile in recommendations
```

---

### 4.6 Reasoning Quality Standards

**Tier 1: Exemplary Reasoning (Target for 30% of training data)**
- Identifies specific problematic elements with precise language
- Cites specific regulations with rule numbers
- Analyzes contextual factors systematically
- Explains severity with justification
- Provides concrete remediation
- Considers alternative interpretations

**Tier 2: Strong Reasoning (Target for 50% of training data)**
- Identifies main compliance issue accurately
- References relevant regulatory framework
- Provides basic justification
- Suggests correction approach

**Tier 3: Basic Reasoning (Target for 20% of training data)**
- Correctly identifies compliant/non-compliant
- Notes primary violation type
- Brief explanation

**Do Not Include:**
- Circular reasoning: "Non-compliant because it violates rules"
- Vague generalizations: "This seems problematic"
- Incorrect regulatory citations
- Contradictory logic
- Reasoning that doesn't match the classification

---

### 4.7 Prompt Engineering for Reasoning Generation

**Template for Generating Reasoning from Scratch:**

```
You are a senior compliance officer with 15 years of experience in investment 
advisory compliance. Analyze the following communication for regulatory compliance.

Provide detailed reasoning following this structure:

COMMUNICATION:
{text}

ANALYSIS:

1. INITIAL ASSESSMENT
   - Overall compliance status: [Compliant/Non-Compliant/Context-Dependent]
   - Confidence level: [High/Medium/Low]

2. DETAILED EXAMINATION
   - Identify specific phrases or elements requiring analysis
   - For each element, explain why it matters for compliance

3. REGULATORY FRAMEWORK
   - Cite specific regulations, rules, or standards
   - Explain how they apply to this scenario
   - Reference relevant SEC/FINRA guidance if applicable

4. CONTEXTUAL FACTORS
   - What additional information would affect this determination?
   - Are there circumstances where this would be compliant/non-compliant?

5. SEVERITY ASSESSMENT (if non-compliant)
   - How serious is this violation?
   - What are potential consequences?
   - Is this a common violation pattern?

6. REMEDIATION PATH (if non-compliant)
   - What specific changes would make this compliant?
   - Provide example compliant language

7. EDGE CASES
   - Are there borderline interpretations?
   - What would push this clearly into compliant or non-compliant territory?

Write as if explaining to a junior compliance analyst. Be specific, cite rules 
precisely, and show your reasoning process step-by-step.
```

**Template for Adding Reasoning to Existing Labels:**

```
The following communication has been classified as {classification} for the following 
violation type(s): {violation_types}.

Generate expert-level reasoning that explains this classification. Your reasoning 
should:
- Identify the specific problematic elements
- Explain regulatory basis with citations
- Assess severity and context
- Provide remediation guidance

Be thorough but concise. Aim for 150-250 words.

COMMUNICATION: {text}
CLASSIFICATION: {classification}
VIOLATION TYPE(S): {violation_types}

REASONING:
```

---

### 4.8 Validating Reasoning Quality

**Automated Checks:**
1. **Consistency**: Does reasoning support the classification?
2. **Specificity**: Does it reference specific phrases from the communication?
3. **Citation Format**: Are regulatory references properly formatted?
4. **Length**: Is reasoning substantive (target: 100-300 words)?
5. **Structure**: Does it follow a logical flow?

**LLM-Based Validation:**
```
Evaluate the quality of the following compliance reasoning:

COMMUNICATION: {text}
CLASSIFICATION: {classification}
REASONING: {reasoning}

Rate the reasoning on these dimensions (1-5 scale):

1. ACCURACY: Does the reasoning correctly analyze the compliance issue?
2. SPECIFICITY: Does it identify precise problematic elements?
3. REGULATORY GROUNDING: Are citations accurate and relevant?
4. COMPLETENESS: Are all key factors addressed?
5. CLARITY: Is the explanation clear and well-structured?

Provide:
- Overall score (average of dimensions)
- Specific issues or errors
- Suggestions for improvement

If score is below 3.5, flag for human review.
```

**Human Expert Review:**
- Sample 500 reasoning examples across violation types
- Check for technical accuracy of regulatory citations
- Verify logic flows correctly
- Ensure remediation guidance is actionable
- Flag any dangerous misinterpretations

---

### 4.9 Integration with Training Pipeline

**Data Structure:**

```json
{
  "id": "example_001",
  "communication": "This fund has consistently outperformed the market.",
  "classification": "NON-COMPLIANT",
  "violation_types": ["misleading_performance_claim", "missing_disclaimer"],
  "severity": "MEDIUM",
  "reasoning": {
    "identification": "The phrase 'consistently outperformed' creates two issues...",
    "regulatory_basis": "FINRA Rule 2210 requires that communications be fair...",
    "context_analysis": "No timeframe specified, no disclaimer present...",
    "severity_rationale": "Medium severity because while misleading...",
    "remediation": "The phrase should be rewritten as: 'This fund has outperformed..."
  },
  "reasoning_full_text": "[Complete reasoning paragraph]",
  "metadata": {
    "reasoning_tier": "tier_2",
    "reviewed_by_expert": true,
    "validation_score": 4.2
  }
}
```

**Training Objectives:**

1. **Primary Task**: Classification (Compliant/Non-Compliant)
2. **Secondary Task**: Generate reasoning that explains the classification
3. **Tertiary Task**: Identify specific violations and severity

**Loss Function Considerations:**
- Weight reasoning quality alongside classification accuracy
- Use ROUGE/BERT-Score to evaluate reasoning against reference reasoning
- Penalty for contradictory reasoning (reasoning doesn't support classification)

---

### 4.10 Reasoning-Aware Evaluation

**Standard Metrics PLUS Reasoning Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| Classification Accuracy | Correct compliant/non-compliant | >90% |
| Reasoning-Classification Consistency | Reasoning supports classification | >95% |
| Regulatory Citation Accuracy | Correct rule/regulation cited | >85% |
| Specificity Score | Reasoning references specific phrases | >80% |
| Expert Agreement | Human agrees with reasoning | >85% |
| Remediation Usefulness | Suggested fix would make communication compliant | >80% |

**Reasoning Evaluation Protocol:**

1. **Automated Pre-Filter**: Check for basic quality (length, structure, citations)
2. **Embedding Similarity**: Compare to reference reasoning for similar violations
3. **Consistency Check**: Ensure reasoning → classification logic is sound
4. **Expert Sampling**: Human review of 10% of test set reasoning
5. **A/B Testing**: Test if reasoning helps human reviewers work more efficiently

**Red Flags in Reasoning:**
- Generic statements that could apply to any communication
- Regulatory citations that don't match the violation type
- Contradictory statements within reasoning
- Reasoning that's longer than the original communication
- Hallucinated facts not present in the communication
- Overconfident reasoning for ambiguous cases

---

### 4.11 Reasoning for Edge Cases and Ambiguity

**When Reasoning Should Express Uncertainty:**

```
Communication: "This investment has historically performed well in rising rate 
environments."

Classification: CONTEXT-DEPENDENT

Reasoning:
This statement requires careful analysis because its compliance status depends on 
several factors:

POTENTIALLY COMPLIANT if:
1. Statement is factually accurate (verifiable historical data)
2. Appropriate disclaimers are present ("past performance does not guarantee...")
3. "Rising rate environments" is defined with specific timeframes
4. Delivered with proper performance presentation standards
5. Not used as primary basis for recommendation without other factors

POTENTIALLY NON-COMPLIANT if:
1. Stated as predictive of future performance without qualification
2. Used to pressure client decision-making ("rates are rising now, so...")
3. Cherry-picked timeframe to show favorable results
4. No disclaimer about risks or limitations

MISSING CONTEXT NEEDED:
- Full conversation context
- Presence of written disclaimers
- Factual accuracy of claim
- How this relates to client's situation

RECOMMENDED ACTION:
Flag for manual review with focus on:
1. Verification of performance data
2. Presence of required disclaimers  
3. Context of how this fits into recommendation

This demonstrates a borderline case where additional information is essential for 
final determination.
```

**Key Principle:** Model should know when it doesn't know. Expressing uncertainty appropriately is a sign of sophistication, not weakness.

---

### 4.12 Reasoning-Enhanced Contrastive Pairs

**Pair Structure with Reasoning:**

**Version A (Compliant):**
```
Communication: "Based on your documented aggressive risk tolerance and 15-year time 
horizon, I recommend a portfolio with 80% equities. Our firm receives 12b-1 fees 
from some of these funds, as disclosed in your Form ADV."

Classification: COMPLIANT

Reasoning: This recommendation demonstrates three essential compliance elements:
(1) Documented suitability basis - references specific client factors with timeframe
(2) Clear disclosure of conflict of interest - 12b-1 fee receipt
(3) Reference to Form ADV - indicates proper disclosure delivery
The recommendation is appropriately qualified ("based on") rather than absolute. 
Complies with FINRA Rule 2111 and Investment Advisers Act Section 206.
```

**Version B (Non-Compliant - Missing Disclosure):**
```
Communication: "Based on your documented aggressive risk tolerance and 15-year time 
horizon, I recommend a portfolio with 80% equities."

Classification: NON-COMPLIANT

Reasoning: While suitability factors are properly documented, this communication has 
a CRITICAL OMISSION: failure to disclose the conflict of interest from 12b-1 fee 
receipt. Investment Advisers Act Section 206(2) requires disclosure of all material 
conflicts. The advisor's compensation based on recommended funds is always material 
to client evaluation. This is a MEDIUM-HIGH severity violation - suitability is 
proper but fiduciary disclosure duty is breached.

DIFFERENCE FROM VERSION A: Removal of single sentence disclosing conflict creates 
violation. This teaches that disclosure is not optional or implicit - it must be 
explicit.
```

**Learning Objective:** The reasoning explicitly contrasts the versions, teaching the model what changed and why it matters.

---

## 5. Advanced Scenario Generation Techniques

### 5.1 Multi-Turn Conversation Scenarios

**Objective:** Create realistic conversation flows that test contextual understanding

**Generation Strategy:**

1. **Violation Escalation**: Start with compliant discussion, gradually introduce violations
2. **Redirection Scenarios**: Client asks leading questions, test if advisor maintains compliance
3. **Missing Context**: Scenarios where violation depends on information not in conversation

**Example Multi-Turn Scenario:**

```
Turn 1 (Advisor): "Let's review your portfolio performance this quarter."
Turn 2 (Client): "I'm disappointed with the returns. My neighbor says he's 
making 15% guaranteed with some investment."
Turn 3 (Advisor - Non-Compliant): "We can definitely match that. Let me 
show you some products that'll get you there."
Turn 3 (Advisor - Compliant): "I understand the frustration, but I should 
clarify that no legitimate investment offers guaranteed returns of that 
level. Let's review your risk tolerance and see if we need to adjust 
your strategy."

QA Pair:
Q: In Turn 3, which advisor response demonstrates proper compliance?
A: The compliant version. The advisor correctly pushes back on the client's 
misconception about "guaranteed" returns and redirects to a suitability-based 
discussion rather than making promises to match unrealistic returns.
```

### 5.2 Edge Case and Ambiguous Scenarios

**Objective:** Train the model to handle gray areas and context-dependent violations

**Categories:**

1. **Context-Dependent Violations**
   - Statement is compliant for one client type, non-compliant for another
   - Timing matters (statements made before vs. after certain disclosures)

2. **Degree of Violation**
   - Minor technical violation vs. serious misconduct
   - Inadvertent error vs. willful violation

3. **Borderline Language**
   - Promotional language that approaches but doesn't cross the line
   - Statements that require additional context to assess

**Example:**

```
Scenario: "This investment has consistently performed well over the past 
10 years, averaging 8% annually."

QA Pair:
Q: Is this statement compliant?
A: Context-dependent. Compliant if: (1) statement is factually accurate, 
(2) accompanied by past performance disclaimers, (3) performance metrics 
are properly calculated (e.g., time-weighted returns), and (4) appropriate 
for the communication channel (e.g., not in headline without disclaimer). 
Non-compliant if these conditions are not met or if it's presented as 
predictive of future results without qualification.
```

### 5.3 Cross-Violation Scenarios

**Objective:** Test model's ability to identify multiple simultaneous violations

**Generation Approach:**

Combine multiple violation types in single scenarios:

```
Scenario: "You should definitely move all your retirement savings into 
this private equity fund. My firm manages it, and we've seen 20% returns 
every year. It's basically a sure thing, and I can get you in before 
the minimum investment increases."

QA Pair:
Q: Identify all compliance violations in this statement.
A: Multiple violations present:
1. Inappropriate absolute recommendation ("should definitely move ALL")
2. Prohibited guarantee implication ("basically a sure thing")
3. Undisclosed conflict of interest (firm manages the fund)
4. Potentially misleading performance claims (need context on calculation)
5. Possible suitability violation (no discussion of risk tolerance, 
   time horizon, or other holdings)
6. Pressure tactics ("before minimum investment increases")
7. Failure to discuss risks, liquidity constraints of private equity
```

---

## 6. Data Preprocessing and Augmentation

### 6.1 Data Cleaning

**For Regulatory Actions:**
- Remove legal boilerplate and procedural language
- Extract core violation descriptions and evidence
- Separate factual findings from legal conclusions
- Normalize entity names and redacted information

**For Internal Communications:**
- De-identify PII (names, account numbers, specific amounts over threshold)
- Remove system-generated messages and timestamps
- Filter out non-compliance-related discussions
- Normalize financial terminology

**For Internal Policies:**
- Extract actionable requirements from narrative text
- Identify explicit vs. implicit rules
- Version control and date validation
- Cross-reference with regulatory sources

### 6.2 Data Augmentation Techniques

**Paraphrasing:**
- Use open-source models to rephrase violations and compliant statements
- Maintain semantic meaning while varying surface form
- Generate 3-5 paraphrases per high-value example

**Entity Substitution:**
- Replace specific products, companies, numbers while maintaining violation structure
- Create industry-specific variations (equities, bonds, alternatives, etc.)

**Negation Addition:**
- Add qualifying language to create compliant versions
- Insert missing disclosures or caveats

**Noise Injection:**
- Add filler words, conversational markers for realism
- Introduce typos and informal language (for chat/call scenarios)

---

## 7. Dataset Construction

### 7.1 Recommended Dataset Composition

**Target Size:** 50,000 - 200,000 examples for effective SLM fine-tuning

**Primary Data Sources by Priority:**

| Data Type | Source | Target % | Estimated Count | Rationale |
|-----------|--------|----------|-----------------|-----------|
| **Contrastive Pairs (Real Base)** | Internal comms + modifications | **25%** | **25,000** | **Highest training value - teaches precise boundaries** |
| Positive Violations (Real) | SEC/FINRA Actions | 15% | 15,000 | Gold standard regulatory precedent |
| Compliant Examples (Real) | Internal comms (reviewed) | 20% | 20,000 | Authentic compliant patterns |
| Positive Violations (Synthetic) | Generated from enforcement | 15% | 15,000 | Diverse violation scenarios |
| Compliant Examples (Synthetic) | Policy-based generation | 10% | 10,000 | Augmentation of compliant class |
| Edge Cases/Ambiguous | Mixed generation | 10% | 10,000 | Context-dependent scenarios |
| Multi-Turn Conversations | Generated + real | 5% | 5,000 | Sequential decision-making |

**Total:** ~100,000 examples

**Key Insight:** Contrastive pairs from real internal communications are the highest-value training data because they:
1. Use authentic language and communication patterns
2. Teach discriminative features rather than superficial correlations
3. Reflect company-specific compliance standards
4. Capture the nuanced boundary between compliant and non-compliant

**Contrastive Pair Breakdown (25,000 total):**
- Minimal Edit Pairs (1-3 words): 7,500
- Phrase Substitution: 6,250
- Addition/Deletion: 5,000
- Context-Dependent: 3,750
- Severity Chains: 2,500

### 7.2 Dataset Splits

**Training Set:** 80% (80,000 examples)
- Representative distribution of violation types
- Balanced compliance/non-compliance
- Stratified by violation severity

**Validation Set:** 10% (10,000 examples)
- Held out during training
- Used for hyperparameter tuning
- Should match test set distribution

**Test Set:** 10% (10,000 examples)
- Final evaluation only
- Include higher proportion of edge cases
- Separate real vs. synthetic subsets for analysis

### 7.3 Class Balancing Strategies

**Challenge:** Natural data is heavily imbalanced (most communications are compliant)

**Solutions:**

1. **Oversampling minority class** (violations) with augmentation
2. **Focal loss** or class-weighted loss functions
3. **Synthetic minority oversampling technique (SMOTE)** at embedding level
4. **Hierarchical sampling**: Ensure representation across violation types

---

## 8. Fine-Tuning Strategy

### 8.1 Recommended Open-Source Models

**For Data Generation:**
- **Llama 3.1 70B** (Meta): Excellent instruction following, reasoning
- **Mixtral 8x22B** (Mistral): Strong performance, efficient inference
- **Qwen 2.5 72B** (Alibaba): Good for financial domain

**For Fine-Tuning (SLMs):**
- **Llama 3.2 3B** (Meta): Best quality/size tradeoff
- **Phi-3 Medium 14B** (Microsoft): Strong reasoning for size
- **Mistral 7B v0.3** (Mistral): Efficient, capable
- **Gemma 2 9B** (Google): Good instruction following

**Recommended Choice:** Llama 3.2 3B or Mistral 7B for production deployment balance

### 8.2 Fine-Tuning Approaches

**Option 1: Full Fine-Tuning**
- Update all model parameters
- Best quality but computationally expensive
- Requires significant compute (multiple GPUs)
- Risk of catastrophic forgetting

**Option 2: LoRA (Low-Rank Adaptation)**
- **Recommended approach**
- Add trainable rank decomposition matrices
- 0.1-1% of parameters updated
- Preserves base model knowledge
- Much faster, less compute

**LoRA Hyperparameters:**
- r: 16 (rank - balance between capacity and efficiency)
- lora_alpha: 32 (scaling factor)
- target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"] (attention layers)
- lora_dropout: 0.05
- bias: "none"

**Option 3: QLoRA (Quantized LoRA)**
- 4-bit quantization + LoRA
- Minimal quality loss
- Can fine-tune larger models on single GPU
- Great for resource-constrained environments

### 8.3 Task Formulation with Reasoning

**Approach 1: Classification with Reasoning (RECOMMENDED)**

This approach teaches the model to both classify and explain its reasoning, providing transparency and auditability essential for compliance applications.

**Input Format:**
```
Analyze the following financial communication for regulatory compliance.
Provide both a classification and detailed reasoning.

Communication: {text}

Provide your analysis in the following format:
CLASSIFICATION: [COMPLIANT/NON-COMPLIANT/CONTEXT-DEPENDENT]
REASONING: [Detailed explanation]
```

**Output Format:**
```
CLASSIFICATION: NON-COMPLIANT

REASONING: This communication contains a prohibited guarantee. The phrase 
"guaranteed returns" violates SEC Rule 206(4)-1(a)(5) and FINRA Rule 2210, 
which prohibit false or misleading statements about investment outcomes. No 
investment can guarantee returns unless it's a legitimately guaranteed product 
like FDIC-insured deposits. Additionally, there is no disclosure about conflicts 
of interest or fees. SEVERITY: HIGH. REMEDIATION: Replace with factual performance 
data and include required disclaimers about past performance and risks.
```

**Training Value:**
- Model learns to justify decisions with regulatory grounding
- Reasoning helps catch edge cases where classification might be uncertain
- Provides actionable output for compliance officers
- Enables error analysis when reasoning is flawed

**Approach 2: Structured Output with Multi-Task Learning**

Train the model to produce structured outputs with classification, reasoning, and metadata:

**Output Format (JSON):**
```json
{
    "compliance_status": "NON_COMPLIANT",
    "confidence": 0.92,
    "violations": [
        {
            "type": "Prohibited Guarantee",
            "severity": "HIGH",
            "evidence": "phrase 'guaranteed returns' in line 1",
            "regulation": "SEC Rule 206(4)-1(a)(5), FINRA Rule 2210"
        },
        {
            "type": "Missing Disclosure",
            "severity": "MEDIUM",
            "evidence": "no conflict of interest disclosure present",
            "regulation": "Investment Advisers Act Section 206(2)"
        }
    ],
    "reasoning": "This communication violates multiple regulatory requirements. 
    First, the use of 'guaranteed returns' is a per se violation under SEC and 
    FINRA advertising rules. No investment product can guarantee returns unless 
    explicitly backed by insurance or government guarantee. Second, there is no 
    disclosure of potential conflicts of interest, which is required under the 
    Investment Advisers Act fiduciary duty standard.",
    "remediation": "Remove guarantee language and replace with: 'Historical 
    returns have averaged X% over the past Y years. Past performance does not 
    guarantee future results.' Add disclosure: 'Our firm receives compensation 
    from [relevant parties], which represents a conflict of interest.'",
    "requires_human_review": false,
    "similar_violations": ["case_ref_123", "case_ref_456"]
}
```

**Training Configuration:**
- Multi-task loss: Classification accuracy + Reasoning quality + Violation identification
- Reasoning quality measured via ROUGE/BERT-Score against reference reasoning
- Structured output enforces consistency between fields

**Approach 3: Chain-of-Thought Before Classification**

Force the model to think through the problem before outputting classification:

**Prompt Format:**
```
Analyze this communication step-by-step:

Communication: {text}

Think through this systematically:
1. What are the key phrases or claims made?
2. What regulatory requirements apply to each?
3. Are required disclosures present?
4. What is the compliance status?

Provide your step-by-step analysis, then final classification.
```

**Example Output:**
```
ANALYSIS:
1. Key phrases: "This fund will outperform", "you should invest now"
   - "will outperform" = forward-looking statement without qualification
   - "you should" = directive recommendation

2. Regulatory requirements:
   - Forward-looking statements require disclaimers (SEC Marketing Rule)
   - Recommendations must have suitability basis (FINRA Rule 2111)
   - Must disclose material conflicts

3. Required disclosures present: NONE
   - No future performance disclaimer
   - No suitability documentation referenced
   - No conflict disclosure

4. Compliance status: NON-COMPLIANT
   - Primary violation: Unqualified forward-looking statement (HIGH severity)
   - Secondary violation: Unsupported recommendation (MEDIUM severity)
   - Missing: All required disclosures

CLASSIFICATION: NON-COMPLIANT
```

**Training Value:** Chain-of-thought improves accuracy on complex cases and makes reasoning explicit.

---

**Recommended Approach: Start with Approach 1 (Classification + Reasoning), then progress to Approach 2 (Structured Output) for production.**

### 8.4 Training Configuration

**Recommended Hyperparameters:**

- learning_rate: 2e-5 (Lower for stability)
- num_epochs: 3-5
- batch_size: 4-8 (Adjust based on GPU memory)
- gradient_accumulation_steps: 4 (Effective batch size = 16-32)
- warmup_ratio: 0.03
- weight_decay: 0.01
- max_seq_length: 2048-4096 (Use 4096 for reasoning outputs)
- optimizer: adamw_torch
- lr_scheduler_type: cosine

**Training Pipeline:**

1. **Tokenization**: Use model's native tokenizer
2. **Padding/Truncation**: Right padding for decoder-only models
3. **Loss Function**: Cross-entropy with class weights or focal loss
4. **Gradient Clipping**: Max norm 1.0 to prevent instability
5. **Mixed Precision**: FP16 or BF16 for efficiency

**Special Considerations for Reasoning:**
- Increase max_seq_length to accommodate reasoning outputs (2048 → 4096 tokens)
- Consider using teacher forcing during training for structured outputs
- Monitor both classification accuracy and reasoning quality metrics
- Use longer warmup period for multi-task learning


### 8.5 Implementation Considerations

**Recommended Stack:**
- Hugging Face Transformers for model loading and training
- PEFT (Parameter-Efficient Fine-Tuning) library for LoRA implementation
- TRL (Transformer Reinforcement Learning) for supervised fine-tuning utilities
- Accelerate for distributed training across multiple GPUs
- BitsAndBytes for quantization support

**Key Implementation Decisions:**
- Use model's native tokenizer with appropriate padding strategy
- Implement gradient accumulation for effective larger batch sizes
- Apply gradient clipping (max norm 1.0) to prevent training instability
- Use mixed precision training (FP16 or BF16) for computational efficiency
- Implement early stopping based on validation set performance


---

## 9. Evaluation Framework

### 9.1 Metrics

**Primary Metrics:**

1. **Precision**: Of flagged violations, how many are real?
   - Critical in production (minimize false alarms)
   - Target: >85% for deployment

2. **Recall**: Of actual violations, how many are caught?
   - Regulatory risk metric
   - Target: >90% for high-severity violations

3. **F1 Score**: Harmonic mean of precision/recall
   - Overall effectiveness measure
   - Target: >0.85

4. **AUC-ROC**: Classification performance across thresholds
   - Helps set confidence thresholds
   - Target: >0.90

**Reasoning Quality Metrics (CRITICAL FOR PRODUCTION):**

5. **Reasoning-Classification Consistency**: Reasoning supports classification
   - Measure: Expert review of alignment
   - Target: >95% consistency

6. **Regulatory Citation Accuracy**: Correct rule/regulation cited
   - Measure: Expert verification of citations
   - Target: >85% accurate citations

7. **Specificity Score**: Reasoning references specific phrases from communication
   - Measure: Automated check for quote extraction
   - Target: >80% of reasoning includes specific evidence

8. **Expert Agreement on Reasoning**: Human agrees with reasoning logic
   - Measure: Sample review by compliance officers
   - Target: >85% agreement

9. **Remediation Usefulness**: Suggested fix would make communication compliant
   - Measure: Expert assessment of remediation guidance
   - Target: >80% useful recommendations

10. **Reasoning Hallucination Rate**: Model invents facts not in communication
    - Measure: Automated + human review
    - Target: <5% hallucination rate

**Secondary Metrics:**

11. **False Negative Rate by Severity**: Critical violations should have <5% FNR
12. **Latency**: Inference time per example (target: <200ms for classification + reasoning)
13. **Calibration**: Confidence scores should match actual accuracy

### 9.2 Evaluation Stratification

Break down performance by:

1. **Violation Type**: Suitability, disclosure, misrepresentation, etc.
2. **Data Source**: Internal vs. external scenarios
3. **Ambiguity Level**: Clear violations vs. edge cases
4. **Communication Channel**: Chat, call transcript, email
5. **Severity**: High/medium/low risk
6. **Reasoning Complexity**: Simple vs. multi-hop reasoning required

### 9.3 Human Evaluation

**Expert Review Process:**

1. **Sample Selection**: 500-1,000 random test examples + all errors
2. **Review Panel**: Compliance officers, legal counsel
3. **Annotation Protocol**: 
   - Binary compliance label
   - Violation types (if applicable)
   - Severity rating
   - Ambiguity flag
   - **Reasoning quality assessment** (accuracy, completeness, clarity)
4. **Inter-Rater Reliability**: Cohen's Kappa > 0.75 target

**Error Analysis:**

- Categorize FPs and FNs by failure mode
- Analyze reasoning failures: where does logic break down?
- Identify systematic biases
- Generate adversarial examples from errors
- Iterate data generation strategy

### 9.4 A/B Testing and Deployment

**Shadow Mode:**
- Run model in parallel with existing process
- Compare to human reviewer decisions
- Measure correlation and disagreement patterns
- Build trust before replacing human review

**Assisted Review Mode:**
- Model flags high-risk communications
- Humans review flagged items
- Measure reduction in human review time
- Track override rates

**Full Automation:**
- Auto-escalate high-confidence violations
- Human review for medium-confidence cases
- Automatic pass for high-confidence compliant
- Continuous monitoring of performance



## 10. Risk Mitigation and Compliance Considerations

### 10.1 Model Risk Management

**Documentation Requirements:**
- Model development methodology
- Data sources and quality assessment
- Validation procedures and results
- Limitations and known failure modes
- Ongoing monitoring plan

**Regulatory Considerations:**
- SR 11-7 (Federal Reserve guidance on model risk)
- OCC Bulletin 2011-12 (Supervisory Guidance on Model Risk Management)
- Explain model decisions for audit trail
- Human oversight requirements

### 10.2 Data Privacy and Security

**Internal Communications:**
- PII redaction before fine-tuning
- Secure storage with encryption
- Access controls and audit logs
- Data retention policies

**Model Security:**
- Prevent data leakage in model outputs
- Adversarial robustness testing
- Secure deployment environment
- Version control and rollback procedures
- **Reasoning transparency**: Ensure model doesn't expose confidential information in explanations

### 10.3 Bias and Fairness

**Potential Biases:**
- Over-representation of certain violation types in training data
- Recency bias from temporal data
- Language/terminology shifts over time
- **Reasoning bias**: Model may cite certain regulations more frequently than appropriate

**Mitigation:**
- Stratified evaluation by protected classes (if applicable)
- Regular audits for disparate impact
- Diverse training data across time periods
- Calibration across subgroups
- **Reasoning quality control**: Verify citation accuracy across all violation types

### 10.4 Continuous Improvement

**Feedback Loops:**
- Human reviewer corrections fed back to training
- Track model disagreements with humans
- A/B test new model versions
- Maintain challenge set of difficult cases

**Data Drift Monitoring:**
- Track input distribution over time
- Monitor confidence score distribution
- Alert on unusual patterns
- Scheduled retraining (quarterly recommended)

---

## 10. Cost Estimation

### 10.1 Infrastructure Costs

**Data Generation (4 weeks):**
- GPU compute: 4x A100 (40GB) = ~$3,000/week × 4 = $12,000
- Storage: $500
- **Total: ~$12,500**

**Fine-Tuning (3 weeks):**
- GPU compute: 8x A100 (80GB) = ~$6,000/week × 3 = $18,000
- Experimentation overhead: $6,000
- **Total: ~$24,000**

**Inference (Production):**
- CPU/Small GPU for 3B model: ~$500/month
- Can run on single A10 or CPU with quantization

### 10.2 Human Labor Costs

- Compliance expert review (~40 hours): $6,000-10,000
- ML engineering (12 weeks): Salaried position
- Data annotation/QA (~80 hours): $4,000-8,000

**Total Project Costs: $50K-75K for initial deployment**

---

## 11. Key Success Factors

1. **High-Quality Seed Data**: Invest in manual labeling of challenging cases
2. **Contrastive Pairs from Real Data**: Prioritize generating contrastive pairs from actual internal communications - this is your highest-value training data
3. **Domain Expertise Integration**: Close collaboration with compliance team throughout
4. **Iterative Development**: Multiple rounds of generation → training → evaluation
5. **Realistic Test Sets**: Include real communications from production environment
6. **Conservative Deployment**: Start with high precision, gradually increase coverage
7. **Continuous Learning**: Build feedback mechanisms from day one
8. **Explainability**: Ensure model can justify flagged violations
9. **Version Control**: Track data, prompts, model versions meticulously

---

## 12. Alternative Approaches to Consider

### 12.1 Few-Shot Learning with Large Models

Instead of fine-tuning:
- Use GPT-4/Claude/Llama 70B with carefully crafted prompts
- Include examples in context (RAG approach)
- Lower upfront cost, less control, higher inference cost

### 12.2 Ensemble Methods

- Combine fine-tuned SLM with rule-based systems
- Multiple models for different violation types
- Voting or cascading architecture

### 12.3 Active Learning

- Start with smaller labeled dataset
- Model identifies most informative examples to label
- Iterative labeling and retraining
- More efficient use of expert time

---

## 13. Recommended Tools and Libraries

**Data Collection:**
- requests, beautifulsoup4, scrapy - web scraping
- pdfplumber, pypdf2 - PDF extraction
- pandas, polars - data manipulation

**Data Generation:**
- vllm - fast LLM inference
- ollama - local model serving
- guidance - structured generation
- outlines - constrained generation

**Fine-Tuning:**
- transformers - model loading and training
- peft - parameter-efficient fine-tuning
- trl - training utilities
- accelerate - distributed training
- bitsandbytes - quantization

**Evaluation:**
- scikit-learn - metrics
- ragas - LLM evaluation
- prometheus-eval - LLM-as-judge
- mlflow - experiment tracking

**Deployment:**
- fastapi - API serving
- ray serve - scalable inference
- triton - optimized serving
- langfuse - monitoring and observability

---

## 14. Conclusion

Fine-tuning a small language model for financial compliance detection is a multi-faceted project requiring careful attention to data quality, synthetic generation techniques, and rigorous evaluation. The combination of regulatory actions (SEC/FINRA), internal policies, and historical communications provides a comprehensive foundation for training.

**Key recommendations:**

1. **Prioritize contrastive pairs from real internal data** - this is your highest-value training data (target 25% of dataset)
2. **Focus on minimal edits** - pairs that differ by 1-3 words teach the most precise discrimination
3. **Invest in data quality over quantity** - 50K high-quality examples beats 500K noisy ones
4. **Build systematic contrastive generation infrastructure** - this requires 30-40% of project effort
5. **Start with LoRA fine-tuning of 3-7B models** - best ROI for production deployment
6. **Build comprehensive evaluation from day one** - including human expert review
7. **Deploy conservatively** - shadow mode first, then assisted review, then automation
8. **Plan for continuous improvement** - compliance evolves, your model must too
9. **Leverage your compliance review history** - previously flagged communications are gold for training data

**Critical Success Factor:** The quality of your contrastive pairs will determine model performance more than any other factor. Invest heavily in generating high-quality contrastive pairs from your real internal communications using the systematic pipeline outlined in Section 3.3.

This approach should yield a production-ready compliance detection system with >85% precision and >90% recall on high-severity violations, deployable on modest infrastructure while maintaining the flexibility and control needed for regulated financial services.



---

## Appendix: Sample Prompt Templates for Data Generation

### Template 1: Violation Scenario Generation

```
You are an expert in financial compliance and regulatory enforcement. Based on 
the following real enforcement action, generate 5 realistic conversation 
scenarios that demonstrate similar violations.

ENFORCEMENT ACTION:
{sec_or_finra_action_text}

For each scenario:
1. Create a natural conversation between a financial advisor and client
2. Make it 3-5 exchanges
3. Include the compliance violation implicitly (don't make it obvious)
4. Vary the context (different products, client situations, communication styles)

Output format:
SCENARIO 1:
[conversation]

SCENARIO 2:
[conversation]

...

Requirements:
- Sound natural and conversational
- Include realistic financial products and situations
- Don't include explicit references to regulations
- Vary severity from obvious to subtle violations
```

### Template 2: Compliant Alternative Generation

```
Given the following non-compliant communication, generate a compliant version 
that addresses the same topic appropriately.

NON-COMPLIANT COMMUNICATION:
{non_compliant_text}

VIOLATION TYPE: {violation_type}

Generate a compliant alternative that:
1. Addresses the same client need or topic
2. Includes appropriate disclosures
3. Avoids prohibited language
4. Demonstrates proper suitability considerations
5. Maintains conversational tone while being compliant

COMPLIANT ALTERNATIVE:
```

### Template 3: QA Pair Generation from Policy

```
You are creating training data for a compliance detection model. Based on the 
following compliance policy, generate 10 question-answer pairs that test 
understanding of compliance violations.

POLICY:
{compliance_policy_text}

For each QA pair:
- Q: Present a realistic scenario or statement
- A: Provide a clear compliance assessment (COMPLIANT/NON-COMPLIANT) with brief explanation

Include:
- 3 clearly compliant examples
- 4 clearly non-compliant examples  
- 3 borderline/context-dependent examples

Output each pair as:
Q: [scenario or statement]
A: [COMPLIANT/NON-COMPLIANT] - [brief explanation]
```

---

*Document Version: 1.0*  
*Last Updated: November 2025*  

