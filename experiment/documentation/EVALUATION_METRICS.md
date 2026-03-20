# Evaluation Metrics Documentation

## Overview

This document explains all 7 evaluation metrics used to compare teacher models (Alpaca-tuned vs OSS-20B-tuned) for knowledge distillation. All metrics are **model-agnostic** and **do not require an LLM judge**.

---

## Metric Summary

| # | Metric | Range | Goal | Description |
|---|--------|-------|------|-------------|
| 1 | **Structured Correctness** | 0-1 | ↑ Higher | Validity of structured outputs (JSON, code) |
| 2 | **Task Success** | 0-1 | ↑ Higher | Does the output solve the task? |
| 3 | **Instruction Following** | 0-2 (rubric) | ↑ Higher | Adherence to constraints |
| 4 | **Coverage** | 0-1 | ↑ Higher | Completeness vs reference |
| 5 | **Faithfulness** | 0-1 | ↑ Higher | Agreement with teacher/context |
| 6 | **Hallucination** | 0-1 | ↓ Lower | Ungrounded content ratio |
| 7 | **Context Grounding** | 0-1 | ↑ Higher | Use of provided context |
| 8 | **Overall Score** | 0-1 | ↑ Higher | Composite score |

---

## Detailed Metric Descriptions

### 1. Structured Correctness

**Purpose:** Measure whether the output is structurally valid when the task requires structured output.

**What it checks:**
- ✅ JSON validity (parseable, correct syntax)
- ✅ Code syntax (Python AST parsing)
- ✅ List/bullet formatting
- ✅ Table formatting
- ✅ Markdown headers

**Scoring Logic:**
```
For code tasks (label = "technical_code"):
  - 1.0: Valid syntax
  - 0.5: Code present but invalid syntax
  - 0.2: No code found

For other tasks:
  - Base 0.5 + bonuses for structure elements
```

**Example:**
```python
# Good (score = 1.0)
def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)

# Bad (score = 0.3)
def factorial(n)  # Missing colon
    return 1 if n <= 1 else n * factorial(n-1
```

---

### 2. Task Success Score

**Purpose:** Heuristic-based measure of whether the output correctly solves the task.

**Components:**
1. **Refusal Detection** - Does output contain refusal phrases?
   - "I cannot", "I'm unable", "As an AI", etc.
2. **Minimum Length** - Is output substantive (>5 words)?
3. **Instruction Relevance** - Cosine similarity to instruction
4. **Teacher Similarity** - Cosine similarity to reference output

**Scoring Logic:**
```
if empty or refusal: 0.1
if too_short (<5 words): 0.2
else:
  score = 0.5 (base)
        + 0.3 × instruction_relevance
        + 0.2 × teacher_similarity
        + task_specific_bonus
```

**Refusal Phrases Detected:**
| Phrase | Example |
|--------|---------|
| "i cannot" | "I cannot provide that information" |
| "i'm unable" | "I'm unable to assist with this" |
| "as an ai" | "As an AI, I don't have opinions" |
| "sorry, i cannot" | "Sorry, I cannot help with that" |

---

### 3. Instruction Following Score

**Purpose:** Measure adherence to explicit constraints in the instruction.

**Components (0-2 rubric each):**

| Component | Score 2 | Score 1 | Score 0 |
|-----------|---------|---------|---------|
| **Length** | Within limit | 10-50% over | >50% over |
| **Format** | Required format present | - | Missing required format |
| **Refusal** | No refusal | - | Contains refusal |
| **Language** | Correct language | - | Wrong language |

**Total Rubric Score:** 0-8 (normalized to 0-1)

**Format Keywords Detected:**
- `json` → Expects `{` or `[`
- `list` / `bullet` → Expects `-`, `*`, or numbers
- `table` → Expects `|` characters
- `code` → Expects ``` or `def`/`function`

**Length Extraction:**
```python
# Patterns detected from instruction:
"in 50 words" → limit = 50
"maximum 100 words" → limit = 100
"3 sentences" → limit = 3
```

---

### 4. Coverage Score

**Purpose:** Measure completeness - does the output cover all key elements from the reference?

**How it works:**
1. Extract key terms from teacher output (content words + bigrams)
2. Extract key terms from student output
3. Calculate overlap

**Formula:**
```
Coverage = |Teacher_terms ∩ Student_terms| / |Teacher_terms|
```

**Example:**
```
Teacher: "Exercise improves cardiovascular health, builds muscle, 
          enhances mental well-being, and maintains weight."
Key terms: {exercise, improves, cardiovascular, health, builds, 
           muscle, enhances, mental, well-being, maintains, weight}

Student: "Exercise is good for your heart and muscles."
Student terms: {exercise, good, heart, muscles}

Covered: {exercise} ∩ overlap with similar terms
Coverage ≈ 0.30 (30% of teacher content covered)
```

---

### 5. Faithfulness Score

**Purpose:** Measure agreement/grounding to the teacher output and provided context.

**Formula:**
```
Faithfulness = β × cos(Student, Teacher) + (1-β) × cos(Student, Context)

where β = 0.6 (teacher weight)
```

**Interpretation:**
| Score | Meaning |
|-------|---------|
| 0.8-1.0 | High agreement with sources |
| 0.5-0.8 | Moderate agreement |
| 0.0-0.5 | Low agreement (potential fabrication) |

**Components returned:**
- `faithfulness_score`: Combined score
- `teacher_similarity`: Cosine similarity to teacher
- `context_similarity`: Cosine similarity to context

---

### 6. Hallucination Score

**Purpose:** Measure ungrounded content - terms in output not present in context.

**Formula:**
```
Hallucination = 1 - (|Context_terms ∩ Output_terms| / |Output_terms|)
```

**⚠️ Note:** Lower is better!

**Interpretation:**
| Score | Meaning |
|-------|---------|
| 0.0-0.2 | Low hallucination (well-grounded) |
| 0.2-0.5 | Moderate hallucination |
| 0.5-1.0 | High hallucination (many ungrounded terms) |

**Limitations:**
- Requires context to be meaningful
- Novel but correct paraphrasing may be penalized
- Best for QA/summarization tasks

---

### 7. Context Grounding Ratio

**Purpose:** Measure how well the output utilizes the provided context.

**Formula:**
```
Context_Grounding = |Context_terms ∩ Output_terms| / |Context_terms|
```

**Difference from Hallucination:**
- **Hallucination**: What fraction of OUTPUT is grounded?
- **Context Grounding**: What fraction of CONTEXT is used?

**Example:**
```
Context: "Paris is the capital of France. It has the Eiffel Tower."
Context terms: {paris, capital, france, eiffel, tower}

Output: "Paris is a beautiful city with the Eiffel Tower."
Output uses: {paris, eiffel, tower}

Context Grounding = 3/5 = 0.60 (60% of context used)
```

---

### 8. Overall Score (Composite)

**Purpose:** Single score combining all metrics.

**Formula:**
```python
positive_metrics = [
    structured_correctness,
    task_success,
    instruction_following,  # normalized to 0-1
    coverage,
    faithfulness,
    context_grounding,
]

negative_metrics = [
    hallucination,  # lower is better
]

mean_positive = sum(positive_metrics) / len(positive_metrics)
mean_negative = sum(negative_metrics) / len(negative_metrics)

overall = mean_positive × (1 - mean_negative)
```

**Interpretation:**
| Score | Meaning |
|-------|---------|
| 0.8-1.0 | Excellent output |
| 0.6-0.8 | Good output |
| 0.4-0.6 | Moderate output |
| 0.0-0.4 | Poor output |

---

## Error Amplification Categories

When comparing teacher vs student:

| Category | Teacher | Student | Meaning |
|----------|---------|---------|---------|
| `both_correct` | ✔ | ✔ | Good distillation |
| `distillation_loss` | ✔ | ✘ | Knowledge lost in transfer |
| `student_hallucination` | ✘ | ✔ | Student got "lucky" (suspicious) |
| `dataset_issue` | ✘ | ✘ | Original data quality problem |

**Target:** Minimize `distillation_loss` cases.

---

## Task Categories

Metrics are applied based on the `label` column:

| Label | Priority Metrics |
|-------|------------------|
| `technical_code` | Structured Correctness, Task Success |
| `math_logic` | Task Success, Coverage |
| `classification_analysis` | Coverage, Instruction Following |
| `language_editing` | Faithfulness, Coverage |
| `creative_generative` | Task Success, Instruction Following |
| `general_qa` | All metrics equally weighted |

---

## Database Column Mapping

All metrics stored as INT (value × 10000 for precision):

| Metric | Alpaca Column | OSS Column |
|--------|---------------|------------|
| Structured Correctness | `alpaca_struct_correct` | `oss_struct_correct` |
| Task Success | `alpaca_task_success` | `oss_task_success` |
| Instruction Following | `alpaca_instr_follow` | `oss_instr_follow` |
| Coverage | `alpaca_coverage` | `oss_coverage` |
| Faithfulness | `alpaca_faithfulness` | `oss_faithfulness` |
| Hallucination | `alpaca_hallucination` | `oss_hallucination` |
| Context Grounding | `alpaca_ctx_grounding` | `oss_ctx_grounding` |
| Overall | `alpaca_overall` | `oss_overall` |

**To convert back to decimal:** `column_value / 10000.0`

---

## Usage Example

```python
from experiment.evaluation_metrics import evaluate_single_output, compare_teacher_student

# Evaluate a single output
result = evaluate_single_output(
    instruction="Summarize the benefits of exercise",
    student_output="Exercise is good for health.",
    teacher_output="Exercise improves cardiovascular health...",
    context="Studies show that regular exercise...",
    task_label="language_editing"
)

print(f"Overall Score: {result['overall_score']}")
print(f"Coverage: {result['coverage']}")
print(f"Hallucination: {result['hallucination']}")

# Compare two teachers
comparison = compare_teacher_student(
    instruction="...",
    teacher_output="...",
    student_output="...",
    context="...",
    task_label="general_qa"
)

print(f"Error Category: {comparison['error_category']}")
print(f"Degradation: {comparison['degradation']}")
```

---

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Generate tuned_alpaca outputs (Windows Laptop)     │
│          python experiment/06a_generate_tuned_alpaca.py     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 2: Generate tuned_oss20b outputs (MacBook Pro)        │
│          python experiment/06b_generate_tuned_oss.py        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 3: Run SQL to create new columns                      │
│          sql/08_eval_matrix_columns.sql                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 4: Compare teachers and select winner                 │
│          python experiment/07_compare_teachers.py           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 5: Generate 50K with selected teacher                 │
│          Use winner to create full training dataset         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Step 6: Fine-tune Gemma 3:1B on selected teacher outputs   │
│          Evaluate final student model                       │
└─────────────────────────────────────────────────────────────┘
```

---

## References

- Original evaluation matrix design based on distillation best practices
- Cosine similarity using TF-IDF weighted bag-of-words
- No external LLM required - fully offline metrics
