Project structure

```shell
eval/
  config/
    eval.yaml
  queries/
    queries.jsonl
  runs/
    20260129_231500_<gitsha>/
      config_snapshot.json
      phase3_pool.jsonl
      phase5_beir_metrics.json
      phase6_answers_rag_no_hyde.jsonl
      phase6_answers_rag_hyde.jsonl
      phase6_answers_baseline_no_retrieval.jsonl
      phase7_ragas_scores.csv
      phase8_audit_sample.jsonl
      report.md
  src/
    eval_cli.py
    clients/
      rag_api.py
    phases/
      phase3_pool.py
      phase5_beir.py
      phase6_generate.py
      phase7_ragas.py
      phase8_audit.py
    schemas/
      models.py

```
# Metrics

| Metric Variable                         | What it Measures                              | How it is Measured                                                                                                                                                                | Value Range and Interpretation                                                                                                             | Why it Matters                                                                                 |
|-----------------------------------------|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| answer_relevancy                        | Relevance of the answer to the question       | An evaluator LLM compares the question and the generated answer and assigns a normalized score based on how directly and completely the answer addresses the question             | 0.0–1.0, where 1.0 means the answer fully and directly addresses the question, and values closer to 0.0 indicate weak or off topic answers | Validates whether the system actually answers what was asked, independent of retrieval quality |
| faithfulness                            | Grounding of the answer in retrieved contexts | The answer is decomposed into atomic claims and an evaluator LLM checks whether each claim is supported by the provided retrieved contexts, scoring the ratio of supported claims | 0.0–1.0, where 1.0 means all claims are supported by evidence, and lower values indicate partial or significant hallucination              | Detects hallucinations and ensures the answer is evidence backed                               |
| llm_context_precision_without_reference | Relevance of retrieved contexts to the answer | An evaluator LLM assesses each retrieved context and scores how many of them are relevant to the answer, without using gold reference labels                                      | 0.0–1.0, where 1.0 means all retrieved contexts are relevant, and lower values indicate increasing retrieval noise                         | Measures retrieval quality and noise when no ground truth documents are available              |

Phase3:
```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase3
```

Phase5.1:
```shell
python -m evaluation_metrics.src.eval_cli \
  --config evaluation_metrics/config/eval.yaml \
  phase5_overlap \
  --phase3-pool-jsonl evaluation_metrics/runs/20260201_175701_nogit/phase3_pool.jsonl \
  --tasks-clean-json evaluation_metrics/tasks_clean.json \
  --list-labels
```

Phase5.2:
````shell
python -m evaluation_metrics.src.eval_cli \
  --config evaluation_metrics/config/eval.yaml \
  phase5_overlap \
  --phase3-pool-jsonl evaluation_metrics/runs/20260201_190141_nogit/phase3_pool.jsonl \
  --tasks-clean-json evaluation_metrics/tasks_clean.json \
  --positive-label-field related_to_vte \
  --positive-yes-value Yes
````

Phase6.1:
```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase6
```

Phase7.1:
```shell
python -m evaluation_metrics.src.eval_cli --config /Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/config/eval.yaml \
  phase7 --input-jsonl /Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/runs/20260201_133945_nogit/phase6_answers_rag_no_hyde.jsonl
```

Phase7.2:
```shell
python -m evaluation_metrics.src.eval_cli --config /Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/config/eval.yaml \
  phase7 --input-jsonl /Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/runs/20260201_133945_nogit/phase6_answers_rag_hyde.jsonl
```

Phase7.3:
```shell
python -m evaluation_metrics.src.eval_cli --config /Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/config/eval.yaml \
  phase7 --input-jsonl /Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/runs/20260201_133945_nogit/phase6_answers_llm_only.jsonl
```

```shell
CFG=/Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/config/eval.yaml
RUN=/Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/runs/20260201_190201_nogit

python -m evaluation_metrics.src.eval_cli --config "$CFG" phase7 --input-jsonl "$RUN/phase6_answers_rag_no_hyde.jsonl"
python -m evaluation_metrics.src.eval_cli --config "$CFG" phase7 --input-jsonl "$RUN/phase6_answers_rag_hyde.jsonl"
python -m evaluation_metrics.src.eval_cli --config "$CFG" phase7 --input-jsonl "$RUN/phase6_answers_llm_only.jsonl"
```

Phase8:
```shell
python -m evaluation_metrics.src.eval_cli \
  --config evaluation_metrics/config/eval.yaml \
  phase8 \
  --rag-no-hyde evaluation_metrics/runs/20260201_190201_nogit/phase6_answers_rag_no_hyde.jsonl \
  --rag-hyde evaluation_metrics/runs/20260201_190201_nogit/phase6_answers_rag_hyde.jsonl \
  --llm-only evaluation_metrics/runs/20260201_190201_nogit/phase6_answers_llm_only.jsonl
```

## RAGAS Evaluation Metrics
Evaluates the **behavioral quality** of the RAG system using **RAGAS**.
It does **not** measure real world factual correctness.
It measures whether the system behaves correctly with respect to its retrieved evidence.

This phase answers one question only:

> Given the retrieved context, did the system answer properly and responsibly?

---

### What Evaluates

Evaluates **three dimensions** of a Retrieval Augmented Generation system:

1. Grounding of the answer in the retrieved context
2. Relevance of the answer to the question
3. Quality of retrieval itself

All metrics are computed **per question** and later aggregated.

---

## Important Conceptual Limitation

RAGAS does **not** verify external truth.

There is:
- No gold dataset
- No PubMed validation
- No biomedical knowledge base lookup

All judgments are made by a **grader LLM**, which reasons over text.

Evaluates **system discipline**, not factual correctness.

---

## Input to this phase

For each record consumes:

- `question`  
- `answer` (generated by the system)
- `contexts` (retrieved documents or chunks)

The evaluator LLM receives all three.

---

## Metrics Used

### 1. Faithfulness

**What it measures**

Faithfulness measures whether the claims made in the answer are supported by the retrieved context.

**What it does NOT measure**

- It does not measure real world correctness
- It does not validate biomedical truth

**How it works conceptually**

1. The answer is decomposed into atomic claims
2. For each claim, the evaluator checks if it can be inferred from the provided context
3. The score is computed as: **Score range** 0.0 to 1.0


**Interpretation**

- `1.0`  
  All claims are grounded in the retrieved context

- `0.0`  
  The answer is fully hallucinated or unsupported

**Important note**

A high faithfulness score means the answer respects its evidence, not that the evidence is correct.

---

### 2. Answer Relevancy

**What it measures**

Answer Relevancy measures whether the answer actually addresses the question that was asked.

**What it does NOT measure**

- It does not measure factual accuracy
- It does not measure grounding

**How it works conceptually**

The evaluator LLM judges whether the answer:

- Responds to the intent of the question
- Is on topic
- Is not a refusal or generic fallback

**Score range** 0.0 to 1.0


**Interpretation**

- `1.0`  
  The answer directly and fully addresses the question

- `0.0`  
  The answer is irrelevant, evasive, or a fallback response

**Example**

A response like  
`"No evidence grounded answer could be produced."`  
receives a score close to `0.0`.

---

### 3. Context Precision Without Reference

**What it measures**

This metric measures how much of the retrieved context was actually useful for producing the answer.

It evaluates **retrieval quality**, not generation quality.

**Why “Without Reference”**

There is no gold reference answer.
The evaluator infers usefulness based only on question, answer, and context.

**How it works conceptually**

1. The evaluator identifies which parts of the retrieved context contributed to the answer
2. Precision is computed as: **Score range** 0.0 to 1.0


**Interpretation**

- `1.0`  
  Almost all retrieved context was relevant

- `0.0`  
  Retrieval was mostly noise

**Important note**

Low precision usually indicates **over retrieval**, not under retrieval.

---

## How Metrics Should Be Interpreted Together

Metrics must be interpreted **jointly**, never in isolation.

### Common Patterns

**High Answer Relevancy + Low Faithfulness**

- Answer sounds good
- Not grounded in evidence
- Hallucination risk

**High Faithfulness + Low Answer Relevancy**

- Answer sticks to context
- Does not actually answer the question

**Low Context Precision**

- Retrieval pipeline is noisy
- Too many irrelevant chunks passed to the model

---

## Special Case: LLM Only Runs

When `contexts = []`:

- Context Precision is always `0.0`
- Faithfulness reflects internal consistency, not grounding
- High scores are misleading if compared to RAG runs

LLM only results must **never** be compared directly to RAG results for grounding metrics.

---

## Fallback Answers

Fallback answers (e.g. refusals or “no evidence” responses):

- Destroy Answer Relevancy
- Distort aggregate metrics
- Mask actual system capability

**Recommendation**

Track fallback answers explicitly in Phase 6 and analyze:

- Metrics on non fallback answers
- Fallback rate as a separate metric

---

## What Phase 7 Is Good For

Phase 7 is ideal for:

- Comparing retrieval strategies (e.g. HyDE vs no HyDE)
- Measuring hallucination control
- Measuring retrieval noise
- Evaluating system discipline

---

## What Phase 7 Cannot Prove

Phase 7 cannot prove:

- Biomedical correctness
- Clinical validity
- Scientific truth

Human evaluation or gold labeled datasets are required for that.

---

## Correct Scientific Framing

**Correct**

> We evaluate grounding, retrieval quality, and answer relevance using RAGAS metrics, which assess consistency between generated answers and retrieved context.

**Incorrect**

> We evaluate answer correctness using RAGAS.

Never claim correctness based on Phase 7.

---

## Summary

Phase 7 answers one question only:

> Did the system retrieve relevant evidence and respect it when generating answers?

It is a necessary evaluation step for RAG systems, but never sufficient on its own.





