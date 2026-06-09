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

### A004 Label Studio raw task export
Path:
labelstudio-tools/ls_data/tasks_raw.json

Type:
JSON task export

What it contains:
Exported Label Studio tasks with bibliographic payloads and attached manual annotations.

Used in thesis:
Chapter 5 annotation-based validation, Chapter 10 annotation schema appendix

### A005 Label Studio clean ingestion dataset
Path:
labelstudio-tools/ls_data/tasks_clean.json

Type:
JSON clean dataset

What it contains:
Curated tasks that passed metadata validation for downstream RAG ingestion.

Used in thesis:
Chapter 5 dataset protocol, Chapter 6 RAG ingestion handoff, Chapter 10 ingestion-ready schema

### A006 Label Studio rejected task report
Path:
labelstudio-tools/ls_data/tasks_rejected.json

Type:
JSON rejection report

What it contains:
Rejected tasks with recorded reasons such as missing DOI, source metadata, year, journal, or authors.

Used in thesis:
Chapter 5 dataset protocol, Chapter 10 workflow assumptions and limitations

### A007 Missing source identifier report
Path:
labelstudio-tools/ls_data/missing_source_ids.json

Type:
JSON quality report

What it contains:
List of source identifiers for which DOI enrichment remained unresolved in the current export.

Used in thesis:
Chapter 5 dataset protocol, Chapter 10 annotation workflow limitations
