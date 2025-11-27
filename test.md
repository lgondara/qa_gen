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

## 4. Advanced Scenario Generation Techniques

### 4.1 Multi-Turn Conversation Scenarios

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

### 4.2 Edge Case and Ambiguous Scenarios

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

### 4.3 Cross-Violation Scenarios

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

## 5. Data Preprocessing and Augmentation

### 5.1 Data Cleaning

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

### 5.2 Data Augmentation Techniques

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

## 6. Dataset Construction

### 6.1 Recommended Dataset Composition

**Target Size:** 50,000 - 200,000 examples for effective SLM fine-tuning

| Data Type | Source | Target % | Estimated Count |
|-----------|--------|----------|-----------------|
| Positive Violations (Real) | SEC/FINRA Actions | 15% | 15,000 |
| Positive Violations (Synthetic) | Generated from real | 25% | 25,000 |
| Compliant Examples (Real) | Internal comms | 20% | 20,000 |
| Compliant Examples (Synthetic) | Policy-based generation | 15% | 15,000 |
| Contrastive Pairs | Generated | 15% | 15,000 |
| Edge Cases/Ambiguous | Generated | 10% | 10,000 |

**Total:** ~100,000 examples

### 6.2 Dataset Splits

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

### 6.3 Class Balancing Strategies

**Challenge:** Natural data is heavily imbalanced (most communications are compliant)

**Solutions:**

1. **Oversampling minority class** (violations) with augmentation
2. **Focal loss** or class-weighted loss functions
3. **Synthetic minority oversampling technique (SMOTE)** at embedding level
4. **Hierarchical sampling**: Ensure representation across violation types

---

## 7. Fine-Tuning Strategy

### 7.1 Recommended Open-Source Models

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

### 7.2 Fine-Tuning Approaches

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
```python
lora_config = {
    "r": 16,  # rank - balance between capacity and efficiency
    "lora_alpha": 32,  # scaling factor
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],  # attention layers
    "lora_dropout": 0.05,
    "bias": "none",
}
```

**Option 3: QLoRA (Quantized LoRA)**
- 4-bit quantization + LoRA
- Minimal quality loss
- Can fine-tune larger models on single GPU
- Great for resource-constrained environments

### 7.3 Task Formulation

**Approach 1: Binary Classification**

```
Input: [Conversation/Statement]
Output: COMPLIANT or NON_COMPLIANT

Prompt Format:
"""
Analyze the following financial communication for regulatory compliance:

Communication: {text}

Is this communication compliant with financial regulations? 
Answer with COMPLIANT or NON_COMPLIANT.
"""
```

**Approach 2: Multi-Label Classification**

```
Output Categories:
- Suitability Violation
- Disclosure Violation  
- Misrepresentation
- Conflict of Interest
- Prohibited Guarantee
- Supervisory Failure
- COMPLIANT

Supports multiple simultaneous violations
```

**Approach 3: Structured Output with Explanation**

```
Output Format (JSON):
{
    "compliance_status": "NON_COMPLIANT",
    "confidence": 0.92,
    "violations": [
        {
            "type": "Prohibited Guarantee",
            "severity": "HIGH",
            "evidence": "phrase 'guaranteed returns'",
            "regulation": "SEC Rule 206(4)-1"
        }
    ],
    "explanation": "Statement makes prohibited guarantee..."
}
```

**Recommended:** Start with Approach 1 for MVP, progress to Approach 3 for production

### 7.4 Training Configuration

**Recommended Hyperparameters:**

```python
training_config = {
    "learning_rate": 2e-5,  # Lower for stability
    "num_epochs": 3-5,
    "batch_size": 4-8,  # Adjust based on GPU memory
    "gradient_accumulation_steps": 4,  # Effective batch size = 16-32
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_seq_length": 2048,  # Adjust based on use case
    "optimizer": "adamw_torch",
    "lr_scheduler_type": "cosine",
}
```

**Training Pipeline:**

1. **Tokenization**: Use model's native tokenizer
2. **Padding/Truncation**: Right padding for decoder-only models
3. **Loss Function**: Cross-entropy with class weights or focal loss
4. **Gradient Clipping**: Max norm 1.0 to prevent instability
5. **Mixed Precision**: FP16 or BF16 for efficiency


### 7.5 Implementation Considerations

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

## 8. Evaluation Framework

### 8.1 Metrics

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

**Secondary Metrics:**

5. **False Negative Rate by Severity**: Critical violations should have <5% FNR
6. **Latency**: Inference time per example (target: <100ms for real-time)
7. **Calibration**: Confidence scores should match actual accuracy

### 8.2 Evaluation Stratification

Break down performance by:

1. **Violation Type**: Suitability, disclosure, misrepresentation, etc.
2. **Data Source**: Internal vs. external scenarios
3. **Ambiguity Level**: Clear violations vs. edge cases
4. **Communication Channel**: Chat, call transcript, email
5. **Severity**: High/medium/low risk

### 8.3 Human Evaluation

**Expert Review Process:**

1. **Sample Selection**: 500-1,000 random test examples + all errors
2. **Review Panel**: Compliance officers, legal counsel
3. **Annotation Protocol**: 
   - Binary compliance label
   - Violation types (if applicable)
   - Severity rating
   - Ambiguity flag
4. **Inter-Rater Reliability**: Cohen's Kappa > 0.75 target

**Error Analysis:**

- Categorize FPs and FNs by failure mode
- Identify systematic biases
- Generate adversarial examples from errors
- Iterate data generation strategy

### 8.4 A/B Testing and Deployment

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

### 10.3 Bias and Fairness

**Potential Biases:**
- Over-representation of certain violation types in training data
- Recency bias from temporal data
- Language/terminology shifts over time

**Mitigation:**
- Stratified evaluation by protected classes (if applicable)
- Regular audits for disparate impact
- Diverse training data across time periods
- Calibration across subgroups

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

## 11. Cost Estimation

### 11.1 Infrastructure Costs

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

### 11.2 Human Labor Costs

- Compliance expert review (~40 hours): $6,000-10,000
- ML engineering (12 weeks): Salaried position
- Data annotation/QA (~80 hours): $4,000-8,000

**Total Project Costs: $50K-75K for initial deployment**

---

## 12. Key Success Factors

1. **High-Quality Seed Data**: Invest in manual labeling of challenging cases
2. **Domain Expertise Integration**: Close collaboration with compliance team throughout
3. **Iterative Development**: Multiple rounds of generation → training → evaluation
4. **Realistic Test Sets**: Include real communications from production environment
5. **Conservative Deployment**: Start with high precision, gradually increase coverage
6. **Continuous Learning**: Build feedback mechanisms from day one
7. **Explainability**: Ensure model can justify flagged violations
8. **Version Control**: Track data, prompts, model versions meticulously

---

## 13. Alternative Approaches to Consider

### 13.1 Few-Shot Learning with Large Models

Instead of fine-tuning:
- Use GPT-4/Claude/Llama 70B with carefully crafted prompts
- Include examples in context (RAG approach)
- Lower upfront cost, less control, higher inference cost

### 13.2 Ensemble Methods

- Combine fine-tuned SLM with rule-based systems
- Multiple models for different violation types
- Voting or cascading architecture

### 13.3 Active Learning

- Start with smaller labeled dataset
- Model identifies most informative examples to label
- Iterative labeling and retraining
- More efficient use of expert time

---

## 14. Recommended Tools and Libraries

**Data Collection:**
- `requests`, `beautifulsoup4`, `scrapy` - web scraping
- `pdfplumber`, `pypdf2` - PDF extraction
- `pandas`, `polars` - data manipulation

**Data Generation:**
- `vllm` - fast LLM inference
- `ollama` - local model serving
- `guidance` - structured generation
- `outlines` - constrained generation

**Fine-Tuning:**
- `transformers` - model loading and training
- `peft` - parameter-efficient fine-tuning
- `trl` - training utilities
- `accelerate` - distributed training
- `bitsandbytes` - quantization

**Evaluation:**
- `scikit-learn` - metrics
- `ragas` - LLM evaluation
- `prometheus-eval` - LLM-as-judge
- `mlflow` - experiment tracking

**Deployment:**
- `fastapi` - API serving
- `ray serve` - scalable inference
- `triton` - optimized serving
- `langfuse` - monitoring and observability

---

## 15. Conclusion

Fine-tuning a small language model for financial compliance detection is a multi-faceted project requiring careful attention to data quality, synthetic generation techniques, and rigorous evaluation. The combination of regulatory actions (SEC/FINRA), internal policies, and historical communications provides a comprehensive foundation for training.

Key recommendations:
1. **Prioritize data quality over quantity** - 50K high-quality examples beats 500K noisy ones
2. **Invest in synthetic generation infrastructure** - this is 40% of the project
3. **Start with LoRA fine-tuning of 3-7B models** - best ROI
4. **Build comprehensive evaluation from day one** - including human review
5. **Deploy conservatively** - shadow mode first, then assisted review
6. **Plan for continuous improvement** - compliance evolves, your model must too

This approach should yield a production-ready compliance detection system with >85% precision and >90% recall on high-severity violations, deployable on modest infrastructure while maintaining the flexibility and control needed for regulated financial services.



---

## Appendix B: Sample Prompt Templates for Data Generation

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
