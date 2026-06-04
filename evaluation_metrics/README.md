# Evaluation Metrics Pipeline

## 1. Purpose

This module provides the thesis evaluation workflow for the biomedical RAG system in this repository. It is intended for MSc thesis reporting, supervisor review, examiner inspection, and future reproduction of the benchmark procedure.

The framework evaluates three distinct but related properties:

1. answer behavior,
2. answer grounding in retrieved evidence,
3. retrieval quality measured independently from answer generation.

It does not establish external biomedical truth or clinical validity.

## 2. Canonical Evaluation Workflow

The recommended thesis entry point is the `paper` command. This is the canonical workflow for benchmark execution and the workflow intended for reported thesis results.

```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml paper
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.medium.yaml paper
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.fast.yaml paper
```

Strong recommendations to run the `paper` workflow and the evaluation metrics CLI together:
```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.fast.yaml retrieval-eval-prepare \
  --retrieval-pool-depth 10 \
  --max-queries 25
```

The three configuration files provide different execution budgets:

- `eval.yaml`: full thesis benchmark configuration
- `eval.medium.yaml`: reduced-cost intermediate benchmark
- `eval.fast.yaml`: fast smoke-level benchmark

The `paper` workflow orchestrates the evaluation in the intended order:

1. question pool loading,
2. answer generation,
3. RAG without HyDE,
4. RAG with HyDE,
5. LLM only baseline,
6. RAGAS evaluation,
7. extraction metrics where configured,
8. audit artifacts,
9. paper summary artifacts.

In implementation terms, the workflow creates a run directory under `evaluation_metrics/runs/<run_id>/`, then executes a deterministic primary benchmark and additional robustness runs where configured. The primary deterministic run is the benchmark intended for direct reporting, while the robustness runs support sensitivity analysis across seeds and temperatures.

For thesis reporting, this `paper` workflow should be treated as the default benchmark procedure. Individual phase commands remain available, but they are intended for debugging, controlled partial reruns, and development diagnostics rather than normal benchmark execution.

## 3. Independent Retrieval Evaluation Workflow

Answer-level evaluation alone cannot determine why a system performed poorly. A weak answer may result from at least four different failure classes:

1. retrieval failure,
2. reranking failure,
3. context selection failure,
4. generation failure.

For this reason, the retrieval workflow is documented separately from the answer-generation workflow. Independent retrieval evaluation uses manually annotated, query-specific relevance judgments (`qrels`) so that retrieval quality can be measured directly rather than inferred indirectly from answer quality.

This separation is important for supervisor-facing scientific interpretation. If a generated answer is weak, answer-level metrics alone do not show whether the system failed to retrieve relevant documents, selected poor context from otherwise relevant candidates, or generated an inadequate answer despite having sufficient evidence. Manual qrels reduce this ambiguity by evaluating the retrieval stage on its own terms.

### Prepare
* `--retrieval-pool-depth`: Controls how many ranked retrieval candidates are exported per query and retrieval mode before pooling and annotation. If is set to 10, exports up to 10 candidates from direct retrieval and up to 10 candidates from HyDE retrieval for each benchmark question.  Higher values improve retrieval evaluation coverage but increase the manual annotation workload.
* `--max-queries`: Limits the number of benchmark questions processed during retrieval evaluation preparation. Runs the workflow for only the first 50 benchmark questions.

```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml retrieval-eval-prepare \
  --retrieval-pool-depth 10 \
  --max-queries 50
```

This stage exports retrieval results and constructs a pooled candidate set for manual annotation. It produces the following key artifacts:

```text
retrieval_direct.jsonl
retrieval_hyde.jsonl
retrieval_final_context.jsonl
pooled_candidates.jsonl
qrels_annotation_template.csv
```

The researcher must then create the completed annotation file by copying the template:

```shell
cp evaluation_metrics/runs/<run_id>/qrels_annotation_template.csv \
   evaluation_metrics/runs/<run_id>/qrels_annotation_completed.csv
```

The copied file must be completed manually. At minimum, the following fields must be filled for every pooled candidate:

```text
relevance
rationale
```

Use the following relevance scale:

```text
0 = irrelevant
1 = partially relevant or contextual
2 = directly relevant and answers the question
```

The intent of this annotation step is to create query-specific gold relevance judgments. These judgments are materially different from weak labels such as `related_to_vte`, which may be useful for audit or exploratory overlap analysis but are not equivalent to gold retrieval relevance for a particular question.

### Finalize

```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml retrieval-eval-finalize \
  --run-dir evaluation_metrics/runs/<run_id> \
  --k 5 \
  --k 10
```

This stage runs:

1. qrels validation,
2. qrels.tsv generation,
3. independent retrieval metrics,
4. LaTeX table generation.

It produces:

```text
qrels.tsv
retrieval_metrics_summary.json
retrieval_metrics_per_query.csv
retrieval_metrics_comparison.tex
```

Warning:

```text
Only request @k metrics when every evaluated query and retrieval mode has at least k deduplicated candidates.
```

This constraint is enforced because the implemented workflow validates retrieval depth after deduplication before computing the requested metrics.

## 4. Generated Artifacts

The following table summarizes the principal artifacts used in thesis reporting and retrieval analysis.

| Artifact                           | Created By                                                                                                                                                                                                                           | Why It Matters                                                                              |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `paper_benchmark_summary.json`     | `paper`                                                                                                                                                                                                                              | Consolidates deterministic and robustness benchmark summaries for reporting and comparison. |
| `phase3_answers_rag_no_hyde.jsonl` | `paper` or `phase3`                                                                                                                                                                                                                  | Stores generated answers for the RAG configuration without HyDE.                            |
| `phase3_answers_rag_hyde.jsonl`    | `paper` or `phase3`                                                                                                                                                                                                                  | Stores generated answers for the RAG configuration with HyDE.                               |
| `phase3_answers_llm_only.jsonl`    | `paper` or `phase3`                                                                                                                                                                                                                  | Stores generated answers for the no-retrieval baseline.                                     |
| `phase4_ragas_scores.csv`          | Conceptual phase label in prior documentation; current CLI writes per-input files such as `phase4_ragas_phase3_answers_rag_no_hyde.csv`, `phase4_ragas_phase3_answers_rag_hyde.csv`, and `phase4_ragas_phase3_answers_llm_only.csv`. | Records RAGAS scores used to evaluate answer behavior and grounding.                        |
| `pooled_candidates.jsonl`          | `retrieval-eval-prepare`                                                                                                                                                                                                             | Contains the deduplicated pooled candidate set for manual retrieval annotation.             |
| `qrels_annotation_template.csv`    | `retrieval-eval-prepare`                                                                                                                                                                                                             | Provides the annotation template to be completed manually by the researcher.                |
| `qrels_annotation_completed.csv`   | Manual researcher step after `retrieval-eval-prepare`                                                                                                                                                                                | Contains the completed query-specific relevance judgments and annotation rationale.         |
| `qrels.tsv`                        | `retrieval-eval-finalize`                                                                                                                                                                                                            | Converts completed manual annotations into metric-ready qrels format.                       |
| `retrieval_metrics_summary.json`   | `retrieval-eval-finalize`                                                                                                                                                                                                            | Aggregates retrieval metrics across retrieval modes for thesis reporting.                   |
| `retrieval_metrics_per_query.csv`  | `retrieval-eval-finalize`                                                                                                                                                                                                            | Preserves per-query retrieval results for diagnostic and methodological analysis.           |
| `retrieval_metrics_comparison.tex` | `retrieval-eval-finalize`                                                                                                                                                                                                            | Produces a LaTeX-ready comparison table for inclusion in the thesis document.               |

## 5. RAGAS Metrics Used

The RAGAS metrics in this framework evaluate system behavior and grounding. They do not verify external biomedical correctness.

| Metric Name                         | What It Measures                                                      | Why It Matters                                                                       |
|-------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Answer Relevancy                    | Whether the generated answer addresses the intent of the question.    | Distinguishes relevant answers from evasive, off-topic, or weakly aligned responses. |
| Faithfulness                        | Whether claims in the answer are supported by the retrieved context.  | Detects unsupported statements and hallucination relative to the provided evidence.  |
| Context Precision Without Reference | How much of the retrieved context was actually useful for the answer. | Indicates whether the retrieval stage delivered focused evidence or excessive noise. |

These metrics should therefore be interpreted as measures of system behavior, grounding discipline, and evidence use. They should not be interpreted as proof that a biomedical answer is factually correct in the external world.

## 6. Retrieval Metrics Used

Independent retrieval evaluation uses manually annotated, query-specific qrels rather than weak labels such as `related_to_vte`.

The reported retrieval metrics are:

1. Precision@k
2. Recall@k
3. MRR
4. nDCG@k

Their roles in the evaluation are as follows:

- `Precision@k` measures how many of the top-ranked retrieved candidates are relevant.
- `Recall@k` measures how much of the relevant material for a query is recovered within the evaluated cutoff.
- `MRR` measures how early the first relevant result appears in the ranked list.
- `nDCG@k` measures ranking quality while respecting graded relevance values (`0`, `1`, `2`).

Because the qrels are manually annotated per query, these metrics provide a substantially stronger basis for retrieval analysis than weak supervision derived from document-level labels. They are therefore appropriate for evaluating retrieval behavior, whereas weak labels remain better suited to exploratory audit and overlap checks.

## 7. Advanced and Debugging Commands

The commands below remain available for development, debugging, and partial reruns. They are not the normal thesis benchmark workflow.

### Phase-Oriented Commands

```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase1
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase2_overlap --phase1-pool-jsonl <phase1_pool.jsonl> --tasks-clean-json evaluation_metrics/tasks_clean.json
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase3
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase4 --input-jsonl <phase3_output.jsonl>
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml phase5 --rag-no-hyde <rag_no_hyde.jsonl> --rag-hyde <rag_hyde.jsonl> --llm-only <llm_only.jsonl>
```

### Retrieval and Qrels Utilities

```shell
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml retrieval-export --retrieval-pool-depth 10 --max-queries 50
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml qrels-create-template --direct <retrieval_direct.jsonl> --hyde <retrieval_hyde.jsonl> --pooled-output <pooled_candidates.jsonl> --template-output <qrels_annotation_template.csv>
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml qrels-validate --pooled <pooled_candidates.jsonl> --annotations <qrels_annotation_completed.csv>
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml qrels-generate --pooled <pooled_candidates.jsonl> --annotations <qrels_annotation_completed.csv> --qrels-output <qrels.tsv>
python -m evaluation_metrics.src.eval_cli --config evaluation_metrics/config/eval.yaml retrieval-metrics --direct <retrieval_direct.jsonl> --hyde <retrieval_hyde.jsonl> --qrels <qrels.tsv> --summary-output <retrieval_metrics_summary.json> --per-query-output <retrieval_metrics_per_query.csv> --latex-output <retrieval_metrics_comparison.tex> --k 5 --k 10
```

Example of qrels validation:
```shell
RUN_DIR=evaluation_metrics/runs/20260604_190317_nogit

python -m evaluation_metrics.src.eval_cli \
  qrels-validate \
  --pooled "$RUN_DIR/pooled_candidates.jsonl" \
  --annotations "$RUN_DIR/qrels_annotation_completed.csv"

python -m evaluation_metrics.src.eval_cli \
  qrels-generate \
  --pooled "$RUN_DIR/pooled_candidates.jsonl" \
  --annotations "$RUN_DIR/qrels_annotation_completed.csv" \
  --qrels-output "$RUN_DIR/qrels.tsv"
  
python -m evaluation_metrics.src.eval_cli retrieval-metrics \
  --direct "$RUN_DIR/retrieval_direct.jsonl" \
  --hyde "$RUN_DIR/retrieval_hyde.jsonl" \
  --qrels "$RUN_DIR/qrels.tsv" \
  --summary-output "$RUN_DIR/retrieval_metrics_summary.json" \
  --per-query-output "$RUN_DIR/retrieval_metrics_per_query.csv" \
  --latex-output "$RUN_DIR/retrieval_metrics_comparison.tex" \
  --k 1
```

These commands are useful when a researcher needs to inspect intermediate artifacts, repeat only a specific stage, verify annotation consistency, or troubleshoot a failed run without re-executing the full benchmark.

## 8. Scientific Framing and Limitations

This framework should be interpreted with the following constraints.

1. RAGAS does not verify biomedical truth.
2. Weak labels are not equivalent to query-specific gold relevance.
3. Manual qrels improve retrieval evaluation but remain limited by annotation quality.
4. The framework evaluates evidence use, retrieval behavior, and answer grounding.
5. It does not establish clinical validity.

More specifically, the framework is designed to determine whether the system retrieved plausible supporting evidence, used that evidence coherently, and produced answers that remain aligned with retrieved context. It is not designed to certify that a generated answer is medically correct, safe for clinical use, or valid as biomedical guidance without expert review.

Similarly, weak labels derived from broader dataset annotations may support overlap analysis or exploratory audit, but they should not be treated as interchangeable with query-specific qrels. The qrels workflow is stronger because it asks a researcher to judge relevance at the query-document level, yet it still remains dependent on annotation consistency, annotation expertise, and the quality of the candidate pool that was exposed for judgment.

## 9. Summary

The thesis workflow should begin with `paper`, which is the canonical benchmark entry point for reported evaluation results. Retrieval evaluation should be treated as a separate two-stage workflow based on manually annotated qrels, because answer-level metrics alone cannot isolate retrieval, reranking, context-selection, and generation failures.

Used together, the two workflows provide a structured evaluation of answer behavior, grounding, and retrieval quality. They support thesis reporting and methodological analysis, but they do not by themselves establish biomedical correctness or clinical validity.
