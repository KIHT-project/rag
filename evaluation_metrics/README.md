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