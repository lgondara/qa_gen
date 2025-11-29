---
layout: distill
title: "Beyond the Leaderboard: When SOTA Meets Reality at the BC Cancer Registry"
description: In academia, we optimize for F1-scores. In healthcare, we optimize for patient outcomes. This is the story of how the BC Cancer Registry reduced a multi-year backlog not by using the largest model, but by using the right one.
date: 2026-04-27
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: 2026-04-27-healthcare-nlp.bib

toc:
  - name: Introduction
  - name: The Metric Trap
  - name: Do Not Use a Cannon to Kill a Fly
  - name: Data Quality is Everything
  - name: Error Handling and Privacy
  - name: Co-Design or Fail
  - name: Build vs Buy - The DARE Framework
  - name: Conclusion
---

## Introduction

The gap between a Jupyter notebook and a hospital server is not just a matter of deployment engineering, it is a fundamental conflict of objectives.

Machine learning researchers are trained to chase the upper bounds of performance metrics. We want the highest F1-score, the lowest perplexity, or the top spot on a leaderboard. But at the **British Columbia Cancer Registry (BCCR)**, where our team processes millions of pathology reports to track cancer incidence and patient outcomes, we learned that a "perfect" model can still fail to solve the actual problem.

Over four years, we deployed various NLP models, from simple regex patterns to fine-tuned BERT models and Large Language Models (LLMs), for tasks including tumor reportability classification, cancer relapse detection, anatomical site identification, and report segmentation.

This post shares the unvarnished reality of what worked, what didn't, and why the gap between research innovation and real-world healthcare deployment is wider than most people think.

## The Metric Trap

In a standard machine learning task, we define success as maximizing some metric, such as accuracy, F1-score, or Area Under the Curve (AUC). But in BCCR’s production registry pipeline, the cost functions are asymmetric and tied to human labor, not model statistics.

Consider our task of **Reportable Tumor Identification**: determining which pathology reports contain cancers that must be tracked by the registry.

**The Academic Goal:** Maximize F1-score by balancing precision and recall.

**The Operational Reality:**
*   Every **Positive** prediction triggers a manual review by a highly trained tumor registrar to finalize the cancer case.
*   Every **Negative** prediction is archived.
*   **False Negatives** are dangerous (missed cancer cases).
*   **False Positives** create burnout (flooding registrars with irrelevant reports).

We discovered that the metric that actually mattered was **Time Saved Per Report**.

Let us do the math from our deployment. Assume a batch of 1,400 reports (1,000 true positives, 400 negatives). Processing a report manually takes roughly **1 minute**.

1.  **Without AI:** Registrars review all 1,400 reports.
    *   *Total Time:* **1,400 minutes**.
2.  **With Our Model:** The model filters negatives with high precision, leaving 1,100 reports. Crucially, the model also performs **sentence-level highlighting**, pointing to the specific text explaining the decision.
    *   *Review Time:* Reduced to **30 seconds** per report due to highlighting.
    *   *Total Time:* **550 minutes**.

This revealed something counterintuitive: A model with lower theoretical accuracy that integrates better into the human-in-the-loop process (via highlighting) is more valuable than a SOTA model that acts as a black box.

> **The Lesson:** Don't just optimize for accuracy. Optimize for the workflow. While ML experts measure success by ROC curves, organizations measure success by backlog reduction.

## Do Not Use a Cannon to Kill a Fly

With all the hype surrounding Generative AI, there is enormous pressure to throw an LLM at every text processing problem. We found this to be computationally wasteful, prone to hallucinations, and often less effective than simpler methods.

We advocate for a **Pragmatic Hybrid Architecture**—a waterfall approach where data flows through progressively more sophisticated models.

```mermaid
graph TD
    A[Pathology Report] --> B{Text Analysis}
    
    B -->|Structured Data<br/>dates, codes, staging| C[Layer 1: Regex]
    B -->|Semantic Understanding<br/>classification tasks| D[Layer 2: Clinical BERT]
    B -->|Complex/Ambiguous<br/>8-12% of cases| E[Layer 3: LLM]
    
    C -->|High precision<br/>instant processing| F[Extracted Structured Data]
    D -->|High accuracy<br/>fast processing| G[Classification Result]
    E -->|High accuracy<br/>slow, complex reasoning| H[Nuanced Analysis]
    
    F --> I[Final Output]
    G --> I
    H --> I
    
    style A fill:#e8f4f8,stroke:#333,stroke-width:2px
    style C fill:#aed6f1,stroke:#333,stroke-width:2px
    style D fill:#a9dfbf,stroke:#333,stroke-width:2px
    style E fill:#f9e79f,stroke:#333,stroke-width:2px
    style I fill:#f5b7b1,stroke:#333,stroke-width:2px
```
<div class="caption">
    Figure 1: Our pragmatic hybrid architecture processes reports through layers of increasing sophistication, reserving expensive models for genuinely difficult cases.
</div>

### Layer 1: The "Boring" Layer (Regex)
For structured data like dates, histology codes, or tumor staging notation (e.g., "T1N0M0"), regular expressions provide **100% precision and zero hallucinations**. They are fast, cheap, and explainable. Extracting "Grade 3" from "Histologic grade: 3/3" requires no machine learning.

### Layer 2: The "Efficient" Layer (Specialized BERT)
For classification tasks requiring semantic understanding, a fine-tuned clinical BERT model (like Gatortron or BioClinicalBERT) is vastly more efficient than prompting large LLMs. Smaller, domain-specific models often outperform general-purpose LLMs on focused tasks while costing a fraction of the computational budget.

### Layer 3: The "Smart" Layer (LLMs)
We reserve Generative AI for the **8-12% of cases** that are ambiguous, require complex reasoning, or involve summarization. This represents a small fraction of our volume but handles scenarios where simpler methods fail.

### The Unsung Hero: Report Segmentation
Pathology reports are full of noise: headers, disclaimers, and clinical history. We found that using a lightweight model to **segment the report**—feeding only the relevant sections (e.g., Diagnosis) to downstream models—improved performance more than simply scaling up model size. As with many ML tasks, preprocessing matters more than parameter count.

## Data Quality is Everything

In academic datasets, labels are provided ground truth. In healthcare, labels are opinions that must be curated. We found that label noise was a massive bottleneck.

### The Fix: Consensus-Based Code Books
We moved away from single-annotator workflows to a consensus-based approach. We spent weeks strictly defining annotation guidelines (a "Code Book") before training a single model.
1.  **Pilot:** Multiple experts label the same set of reports.
2.  **Conflict Resolution:** Disagreements are discussed to refine the Code Book definitions.
3.  **Result:** High inter-annotator agreement.

If human experts cannot agree on the label, the model has no chance.

### Data Drift is Real
Medical terminology and reporting formats evolve. A model trained on 2019 pathology reports can struggle with 2024 reports that use newer WHO classifications.

```mermaid
graph LR
    A[Training Data<br/>2019-2020] --> B[Model Training]
    B --> C[High Accuracy<br/>95%]
    C --> D[Deployment]
    D --> E[2021 Data<br/>92% accuracy]
    E --> F[2022 Data<br/>88% accuracy]
    F --> G[2023 Data<br/>85% accuracy]
    
    style A fill:#90EE90
    style C fill:#90EE90
    style E fill:#FFD700
    style F fill:#FFA500
    style G fill:#FF6347
```
<div class="caption">
    Figure 2: Without continuous monitoring, model performance degrades over time as medical terminology shifts.
</div>

Our solution involves **Automated Drift Detection**: monitoring prediction distributions and confidence scores. When the model becomes less confident on average, it is a signal to retrain.

## Error Handling and Privacy

No model is perfect. In healthcare, this is especially critical because errors have real consequences.

### Confidence-Based Routing
We implemented a multi-layer routing system:
1.  **High Confidence:** Processed automatically (with random audits).
2.  **Low Confidence:** Flagged for human review.
3.  **Audit Strategy:** We use a clinical-trial design approach to auditing, randomly sampling outputs to estimate error rates with statistical significance.

### Privacy by Design
LLMs have a tendency to memorize training data, which poses a risk of data leakage (Membership Inference Attacks). For sensitive registry data:
*   **Local Hosting:** We prefer local, open-weights models (like Llama or Mistral) hosted within our firewall over API-based models.
*   **Differential Privacy:** We explore training with differential privacy (DP) to mathematically guarantee that individual patient data cannot be reverse-engineered from the model weights.

## Co-Design or Fail

We had a significant advantage: we are the provincial cancer registry. Our team includes ML researchers, tumor registrars, and oncologists.

During our co-design sessions, we realized our initial goal was wrong.
*   **Initial Goal:** "Create an NLP solution that is 99% accurate."
*   **Revised Goal:** "Reduce the 24-month backlog."

This pivot changed everything. Instead of a black-box classifier, we built an **assistive tool** that highlights evidence to speed up human decision-making.

```mermaid
graph TB
    NLP[NLP System]
    
    subgraph End Users
        TR[Tumor Registrars]
    end
    
    subgraph Clinical Experts
        ONC[Oncologists]
    end
    
    subgraph Technical Team
        ML[ML Researchers]
        IT[IT Infrastructure]
    end
    
    NLP --- TR
    NLP --- ONC
    NLP --- ML
    NLP --- IT
    
    TR -.workflow needs.-> ML
    ONC -.clinical validation.-> ML
    ML -.technical specs.-> IT
    IT -.infrastructure constraints.-> ML
    
    style NLP fill:#e74c3c,stroke:#333,stroke-width:3px,color:#fff
    style TR fill:#3498db,stroke:#333,stroke-width:2px
    style ONC fill:#e67e22,stroke:#333,stroke-width:2px
    style ML fill:#2ecc71,stroke:#333,stroke-width:2px
```
<div class="caption">
    Figure 3: Successful deployment required alignment across multiple groups. Without registrar input, we would have solved the wrong problem.
</div>

It's crucial to recognize that the ultimate goal of AI in healthcare is not just maximizing accuracy, but achieving business objectives like improving efficiency and patient care.

## Build vs Buy - The DARE Framework

Many healthcare organizations lack in-house ML expertise and opt to buy off-the-shelf AI tools. This is risky. A vendor's "99% accuracy" claim is usually based on their clean dataset, not your messy real-world data.

We propose the **DARE framework** for evaluating external tools:

| D | **Demand Robust Validation** | Do not accept whitepapers. Demand validation on *your* local data distribution. Vendors should provide stratified performance metrics on your actual data. |
| A | **Assess Flexibility** | Can the tool handle your specific quirks (e.g., local report formatting)? Can you fine-tune it when your data distribution changes? |
| R | **Rigorous Internal Compatibility** | Does the tool introduce fairness biases regarding your specific demographics? Does it integrate with your IT infrastructure? Does it comply with your privacy regulations? |
| E | **Ease of Evaluation** | Is the tool a black box? Can you inspect predictions to understand failure modes? Does it provide confidence scores? Can clinical staff override the AI when needed? |

The DARE framework protects organizations from expensive mistakes. Always validate vendor claims on your own data before committing resources.

## Conclusion

The journey from academic ML to deployed healthcare AI requires rethinking everything we learned in grad school.

1.  **Stop optimizing for metrics; start optimizing for workflows.** The solution that saves the most time isn't always the one with the highest accuracy on a held-out test set.
2.  **Stop chasing trends; start matching tools to problems.** Regex can outperform LLMs on structured extraction. Reserve expensive models for problems that actually need them.
3.  **Stop treating deployment as an afterthought.** Internal collaboration between ML researchers and registrars prevented us from building technically impressive solutions that solved the wrong problems.
4.  **Stop trusting vendor claims; start demanding validation.** Use the DARE framework to protect yourself.

The gap between research and production is wide, but it's bridgeable. It requires humility, pragmatism, and a willingness to prioritize patient outcomes over publication metrics.

**The real SOTA is AI that works.**
