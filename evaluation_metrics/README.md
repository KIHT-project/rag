# Evaluation Metrics Pipeline

This project implements a structured, phase-based evaluation framework for Retrieval Augmented Generation systems. It is designed for MSc level research and focuses on measuring system behavior, retrieval discipline, and answer grounding, not external biomedical correctness.

The pipeline is intentionally modular. Each phase consumes the artifacts of the previous phase and produces explicit, inspectable outputs stored per run.

---

## Conceptual Overview

The evaluation answers one core question, broken down step by step:

Did the system retrieve relevant evidence and behave correctly when generating answers based on that evidence?

The pipeline is divided into five phases:

Phase 1 Question Pool Construction
Phase 2 applies BEIR style retrieval evaluation to a custom biomedical question set, using overlap and label-driven relevance instead of official BEIR datasets.
   * Overlap based retrieval evaluation
   * Question to document relevance framing
   * Label based positive vs negative judgments
   * Recall and overlap style metrics
Phase 3 Answer Generation
   * Calls the RAG API
   * Generates answers
   * Stores raw outputs
Phase 4 Behavioral Evaluation with RAGAS
   * Grounding 
   * Answer relevance 
   * Retrieval usefulness
Phase 5 Comparative Audit and Reporting
   * Compares systems 
   * Samples outputs 
   * Produces qualitative audit artifacts

Each phase is isolated, reproducible, and produces versioned artifacts.

---

## Project Structure

```text
eval/
  config/
    eval.yaml
  queries/
    queries.jsonl
  runs/
    <timestamp>_<gitsha>/
      config_snapshot.json
      phase1_pool.jsonl
      phase2_beir_metrics.json
      phase3_answers_rag_no_hyde.jsonl
      phase3_answers_rag_hyde.jsonl
      phase3_answers_llm_only.jsonl
      phase4_ragas_scores.csv
      phase5_audit_sample.jsonl
      report.md
  src/
    eval_cli.py
    clients/
      rag_api.py
    phases/
      phase1_pool.py
      phase2_beir.py
      phase3_generate.py
      phase4_ragas.py
      phase5_audit.py
    schemas/
      models.py
```

---

## Phase 1 Question Pool Construction

Purpose

Builds the evaluation question pool and associates each question with metadata and optional labels. This phase defines what the system will be evaluated on.

Inputs

queries.jsonl

Outputs

phase1_pool.jsonl

Command

```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase1
```

Why it exists

Evaluation quality is bounded by question quality. This phase isolates question definition from retrieval and generation.

---

## Phase 2 Retrieval Evaluation

Purpose

Evaluates retrieval quality independently from generation. This phase answers whether relevant documents are being retrieved at all.

This phase is retrieval only (/search endpoint). No answer generation occurs here.

Sub phases

Phase 2.1 Label inspection

```shell
python -m evaluation_metrics.src.eval_cli \
  --config evaluation_metrics/config/eval.yaml \
  phase2_overlap \
  --phase1-pool-jsonl evaluation_metrics/runs/20260205_181732_nogit/phase1_pool.jsonl \
  --tasks-clean-json evaluation_metrics/tasks_clean.json \
  --list-labels
```

Phase 2.2 Positive label evaluation

```shell
python -m evaluation_metrics.src.eval_cli \
  --config evaluation_metrics/config/eval.yaml \
  phase2_overlap \
  --phase1-pool-jsonl evaluation_metrics/runs/20260205_230000_nogit/phase1_pool.jsonl \
  --tasks-clean-json evaluation_metrics/tasks_clean.json \
  --positive-label-field related_to_vte \
  --positive-yes-value Yes
```

Outputs

phase2_beir_metrics.json

Why it exists

Generation quality is meaningless if retrieval fails. This phase isolates retrieval performance and overlap behavior.

---

## Phase 3 Answer Generation

Purpose

Generates answers using different system configurations.

Typical configurations

RAG without HyDE
RAG with HyDE
LLM only without retrieval

Outputs

phase3_answers_rag_no_hyde.jsonl
phase3_answers_rag_hyde.jsonl
phase3_answers_llm_only.jsonl

Command

```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase3
```

Why it exists

Separates answer generation from evaluation. This allows repeated evaluation without rerunning generation.

---

## Phase 4 Behavioral Evaluation with RAGAS

Purpose

Evaluates system behavior using RAGAS metrics. This phase does not assess real world biomedical correctness. It evaluates discipline, grounding, and relevance.

Core question

Given the retrieved context, did the system answer properly and responsibly?

Inputs per record

question
answer
contexts

Command examples

```shell
CFG=/Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/config/eval.yaml
RUN=/Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/runs/20260205_230158_nogit

python -m evaluation_metrics.src.eval_cli --config "$CFG" phase4 --input-jsonl "$RUN/phase3_answers_rag_no_hyde.jsonl"
python -m evaluation_metrics.src.eval_cli --config "$CFG" phase4 --input-jsonl "$RUN/phase3_answers_rag_hyde.jsonl"
python -m evaluation_metrics.src.eval_cli --config "$CFG" phase4 --input-jsonl "$RUN/phase3_answers_llm_only.jsonl"
```

Output

phase4_ragas_scores.csv

---

## RAGAS Metrics Used

These metrics evaluate **system behavior**, not biomedical correctness.

| Metric Name                         | What it Measures                                                  | Why it Exists                                     | How It Is Computed (Conceptual)                                              | Value Range | How to Interpret Values                                                                         | When to Use It                                                |
|-------------------------------------|-------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------|-------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Answer Relevancy                    | Whether the generated answer addresses the intent of the question | To detect off topic, evasive, or fallback answers | Evaluator LLM compares question and answer and judges semantic alignment     | 0.0 to 1.0  | 1.0 means the answer directly and fully addresses the question, 0.0 means irrelevant or refusal | Detect weak answering behavior even when retrieval is correct |
| Faithfulness                        | Whether claims in the answer are supported by retrieved context   | To detect hallucinations relative to evidence     | Answer is decomposed into atomic claims and checked against provided context | 0.0 to 1.0  | 1.0 means all claims are grounded in context, 0.0 means unsupported or hallucinated             | Measure evidence discipline of the system                     |
| Context Precision Without Reference | How much of the retrieved context was actually useful             | To detect retrieval noise and over retrieval      | Evaluator identifies which retrieved chunks contributed to the answer        | 0.0 to 1.0  | 1.0 means almost all context was relevant, low values indicate noisy retrieval                  | Optimize retrieval depth and chunking strategy                |

Important

These metrics evaluate system behavior, not external truth.

## Phase 5 Comparative Audit and Reporting

Purpose

Compares multiple system configurations and produces a compact audit dataset for manual inspection and reporting.

Inputs

Multiple Phase 3 answer files

Command

```shell
CFG=/Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/config/eval.yaml
RUN=/Users/nchaikalis/IdeaProjects/rag/evaluation_metrics/runs/20260205_182010_nogit

python -m evaluation_metrics.src.eval_cli \
  --config "$CFG" \
  phase5 \
  --rag-no-hyde "$RUN/phase3_answers_rag_no_hyde.jsonl" \
  --rag-hyde "$RUN/phase3_answers_rag_hyde.jsonl" \
  --llm-only "$RUN/phase3_answers_llm_only.jsonl"
```

Outputs

phase5_audit_sample.jsonl

Why it exists

Metrics alone are insufficient. This phase enables qualitative inspection and paper ready comparison.

---

## Critical Limitations

No gold dataset
No biomedical fact verification
No PubMed validation

All judgments are made by a grader LLM over text.

This pipeline evaluates discipline, not truth.

---

## Correct Scientific Framing

Correct

We evaluate grounding, retrieval quality, and answer relevance using RAGAS metrics, which assess consistency between generated answers and retrieved context.

Incorrect

We evaluate biomedical correctness using RAGAS.

---

## Summary

This evaluation framework provides:

Isolated retrieval evaluation
Controlled answer generation
Behavioral evaluation using RAGAS
Comparable and reproducible artifacts
It is necessary for RAG system evaluation, but never sufficient for clinical or scientific validation.
