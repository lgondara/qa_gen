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

### 3.3 QA Generation from Historical Communications

**Approach 1: Self-Supervised Labeling**

Use a large open-source model to create pseudo-labels:

1. **Pre-filter** communications using keyword/pattern matching
2. **LLM Review**: Use Llama 3.1 70B or similar to analyze filtered communications
3. **Confidence Scoring**: Generate confidence scores for labels
4. **Human Review**: Manual review of high-confidence positive cases

**Prompt for Pre-Labeling:**
```
You are a compliance officer reviewing financial advisor communications. 
Analyze the following conversation for potential regulatory violations.

Categories to check:
- Suitability and know-your-customer violations
- Misrepresentation or omission of material facts
- Prohibited promises or guarantees
- Failure to disclose conflicts of interest
- Inappropriate recommendations
- Supervisory failures
- Record-keeping violations

Conversation:
[Insert conversation]

Provide:
1. Violation assessment (Compliant/Non-Compliant/Uncertain)
2. Confidence score (0-100)
3. Specific violations identified (if any)
4. Relevant regulatory citations
5. Explanation of reasoning
```

**Approach 2: Contrastive Pair Generation**

For labeled communications (both compliant and non-compliant):

1. Generate variations with different compliance statuses
2. Create minimal pairs (one word/phrase difference changes compliance status)
3. Test model's sensitivity to key compliance indicators

**Example:**
```
Original: "Based on your conservative risk profile, I recommend this 
bond fund with an average duration of 5 years."

Non-compliant variation: "I recommend this bond fund - it's perfect for you."

Contrastive QA:
Q: Which statement better demonstrates suitability documentation?
A: The first statement explicitly references the client's risk profile 
(conservative) and provides relevant product characteristics (5-year duration), 
while the second makes an assertion without justification.
```

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

### 7.5 Implementation Framework

**Recommended: Hugging Face Transformers + PEFT + TRL**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./compliance-slm",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

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

---

## 9. Practical Implementation Roadmap

### Phase 1: Data Collection and Preparation (Weeks 1-3)

**Week 1:**
- [ ] Set up web scrapers for SEC/FINRA
  - BeautifulSoup/Scrapy for HTML parsing
  - Store in structured format (JSON/PostgreSQL)
  - Schedule automated daily updates
- [ ] Inventory internal compliance documents
- [ ] Secure access to historical communications
- [ ] Establish data governance and privacy protocols

**Week 2:**
- [ ] Clean and normalize SEC/FINRA data
- [ ] De-identify internal communications
- [ ] Extract policies from internal documents
- [ ] Build data versioning system (DVC or similar)

**Week 3:**
- [ ] Initial exploratory data analysis
- [ ] Identify violation type taxonomy
- [ ] Sample and manually label 500 examples for calibration
- [ ] Document data quality issues

### Phase 2: Synthetic Data Generation (Weeks 4-6)

**Week 4:**
- [ ] Set up LLM infrastructure (vLLM, TGI, or Ollama)
- [ ] Develop prompt templates for each generation task
- [ ] Generate initial 10K QA pairs from regulatory actions
- [ ] Human review of sample outputs

**Week 5:**
- [ ] Generate scenarios from internal policies (20K examples)
- [ ] Create contrastive pairs from internal comms (15K examples)
- [ ] Implement data augmentation pipeline
- [ ] Quality filtering and deduplication

**Week 6:**
- [ ] Generate multi-turn conversations (10K examples)
- [ ] Create edge case and ambiguous scenarios (10K examples)
- [ ] Finalize dataset composition
- [ ] Create train/val/test splits

### Phase 3: Model Fine-Tuning (Weeks 7-9)

**Week 7:**
- [ ] Set up training infrastructure (GPU cluster/cloud)
- [ ] Implement data loading and preprocessing
- [ ] Baseline: Test pre-trained model zero-shot performance
- [ ] Configure LoRA and training parameters

**Week 8:**
- [ ] First training run (binary classification)
- [ ] Evaluate on validation set
- [ ] Hyperparameter tuning
- [ ] Ablation studies (data source importance)

**Week 9:**
- [ ] Train final model(s) 
- [ ] Implement structured output version
- [ ] Model compression (quantization, pruning if needed)
- [ ] Benchmark inference latency

### Phase 4: Evaluation and Iteration (Weeks 10-12)

**Week 10:**
- [ ] Comprehensive test set evaluation
- [ ] Error analysis and categorization
- [ ] Expert human review of sample predictions
- [ ] Generate adversarial test cases

**Week 11:**
- [ ] Identify data gaps from error analysis
- [ ] Generate targeted training data for weak areas
- [ ] Retrain with augmented dataset
- [ ] A/B test model versions

**Week 12:**
- [ ] Finalize model selection
- [ ] Create deployment package
- [ ] Document model cards and limitations
- [ ] Prepare for shadow mode deployment

### Phase 5: Deployment and Monitoring (Week 13+)

**Initial Deployment:**
- [ ] Deploy in shadow mode (no action taken)
- [ ] Instrument logging and monitoring
- [ ] Compare to human baselines
- [ ] Collect feedback loop data

**Ongoing:**
- [ ] Weekly performance review
- [ ] Monthly retraining with new data
- [ ] Quarterly human evaluation audits
- [ ] Continuous prompt engineering and data curation

---

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

## Appendix A: Sample Scraping Scripts

### SEC Enforcement Actions Scraper

```python
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time

class SECEnforcementScraper:
    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.enforcement_url = f"{self.base_url}/litigation/litreleases.html"
        
    def scrape_enforcement_list(self, year=None):
        """Scrape list of enforcement actions"""
        url = self.enforcement_url
        if year:
            url = f"{self.base_url}/litigation/litreleases/litrelarchive/litarch{year}.html"
            
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        actions = []
        for row in soup.find_all('tr')[1:]:  # Skip header
            cols = row.find_all('td')
            if len(cols) >= 3:
                action = {
                    'release_no': cols[0].text.strip(),
                    'date': cols[1].text.strip(),
                    'title': cols[2].text.strip(),
                    'url': self.base_url + cols[0].find('a')['href'] if cols[0].find('a') else None
                }
                actions.append(action)
        
        return actions
    
    def scrape_action_details(self, action_url):
        """Scrape detailed content of specific action"""
        if not action_url:
            return None
            
        response = requests.get(action_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content
        content_div = soup.find('div', {'id': 'contentArea'}) or soup.find('div', {'class': 'article'})
        
        if content_div:
            text = content_div.get_text(separator='\n', strip=True)
            return {
                'url': action_url,
                'full_text': text,
                'scraped_at': datetime.now().isoformat()
            }
        return None
    
    def scrape_year(self, year, output_file):
        """Scrape all actions for a given year"""
        print(f"Scraping enforcement actions for {year}...")
        actions = self.scrape_enforcement_list(year)
        
        detailed_actions = []
        for i, action in enumerate(actions):
            print(f"Processing {i+1}/{len(actions)}: {action['release_no']}")
            if action['url']:
                details = self.scrape_action_details(action['url'])
                if details:
                    action.update(details)
            detailed_actions.append(action)
            time.sleep(1)  # Be respectful to SEC servers
        
        with open(output_file, 'w') as f:
            json.dump(detailed_actions, f, indent=2)
        
        print(f"Saved {len(detailed_actions)} actions to {output_file}")
        return detailed_actions

# Usage
scraper = SECEnforcementScraper()
scraper.scrape_year(2024, 'sec_enforcement_2024.json')
```

### FINRA Disciplinary Actions Scraper

```python
import requests
from bs4 import BeautifulSoup
import json
import time

class FINRADisciplinaryScraper:
    def __init__(self):
        self.base_url = "https://www.finra.org"
        self.search_url = f"{self.base_url}/rules-guidance/oversight-enforcement/finra-disciplinary-actions"
    
    def search_actions(self, start_date, end_date, page=1):
        """Search disciplinary actions within date range"""
        # FINRA uses a search API - adjust based on actual implementation
        params = {
            'start_date': start_date,  # Format: YYYY-MM-DD
            'end_date': end_date,
            'page': page
        }
        
        # This is a simplified version - actual implementation depends on FINRA's API
        response = requests.get(self.search_url, params=params)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        actions = []
        for item in soup.find_all('div', {'class': 'disciplinary-action'}):
            action = {
                'case_number': item.find('span', {'class': 'case-number'}).text.strip(),
                'respondent': item.find('span', {'class': 'respondent'}).text.strip(),
                'date': item.find('span', {'class': 'date'}).text.strip(),
                'summary': item.find('div', {'class': 'summary'}).text.strip(),
                'url': self.base_url + item.find('a')['href']
            }
            actions.append(action)
        
        return actions
    
    def scrape_action_details(self, action_url):
        """Scrape full details of disciplinary action"""
        response = requests.get(action_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        details = {
            'full_text': soup.find('div', {'class': 'content'}).get_text(separator='\n', strip=True),
            'violations': [v.text.strip() for v in soup.find_all('span', {'class': 'violation'})],
            'sanctions': soup.find('div', {'class': 'sanctions'}).text.strip() if soup.find('div', {'class': 'sanctions'}) else None
        }
        
        return details

# Usage
scraper = FINRADisciplinaryScraper()
actions = scraper.search_actions('2024-01-01', '2024-12-31')
```

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
*Prepared for: Vanguard AI Research & Development*
