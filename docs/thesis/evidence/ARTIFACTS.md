# ARTIFACTS

Purpose
Index of internal experimental and system artifacts that can be cited in the thesis.
Each entry maps a file path to meaning, ownership, and downstream usage.

Rules
1. Do not modify artifacts under evaluation_metrics/runs.
2. Prefer a single canonical run directory for thesis results.
3. If multiple runs exist, explicitly mark which one is primary and why.

## Canonical run
Primary run directory:
evaluation_metrics/runs/old/feb-10-valid-1-full-run/20260209_114745_nogit

Secondary runs:
evaluation_metrics/runs/old/feb-9-valid-6-medium-run/20260209_101953_nogit, use only for robustness comparisons.

## Artifact index

### A001 Paper benchmark summary
Path:
evaluation_metrics/runs/old/feb-10-valid-1-full-run/20260209_114745_nogit/paper_benchmark_summary.json

Type:
JSON summary

Produced by:
evaluation_metrics phase pipeline

What it contains:
High level aggregation of retrieval and generation evaluation results across configurations.

Used in thesis:
Chapter 7 Results summary tables and narrative

### A002 RAGAS outputs, LLM only
Path:
evaluation_metrics/runs/old/feb-10-valid-1-full-run/20260209_114745_nogit/primary_deterministic/phase4_ragas_phase3_answers_llm_only.csv

Type:
CSV metrics

What it contains:
Per query RAGAS metrics for the baseline configuration.

Used in thesis:
Chapter 7 RAGAS metric comparison

### A003 RAGAS outputs, RAG HyDE
Path:
evaluation_metrics/runs/old/feb-10-valid-1-full-run/20260209_114745_nogit/primary_deterministic/phase4_ragas_phase3_answers_rag_hyde.csv

Type:
CSV metrics

Used in thesis:
Chapter 7 RAG HyDE comparison