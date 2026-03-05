# AGENTS.md

## Repository intent
This is a monorepo containing:
1. A biomedical RAG platform service, `biomed_knowledge_platform`
2. A PubMed scheduling service, `scheduler_pubmed`
3. An evaluation framework, `evaluation_metrics`
4. End to end BDD tests, `tests/bdd`
5. A thesis workspace under `docs/thesis/latex/src` and thesis planning under `docs/THESIS_PLAN.md`, `docs/high-level-design.pdf`, `docs/low-level-design.pdf` The `low-level-design.pdf` may contain less details for project structure so for project structure read the project code.

Agents must treat production code and thesis writing as separate concerns with strict edit boundaries.

## Canonical thesis source
Canonical thesis source is LaTeX under:
- `docs/thesis/latex/src/`

Planning source is:
- `docs/THESIS_PLAN.md`

Rendered artifacts include:
- PDF output produced by LaTeX build
- DOCX output produced by conversion tooling
- `thesis_v0.1.docx` is a snapshot artifact, not canonical, do not hand edit it unless explicitly instructed.

## High level monorepo map
### biomed_knowledge_platform
FastAPI based RAG core service.
Contains adapters, core domains, services, use cases, API endpoints, and its own tests.

Key locations:
- `biomed_knowledge_platform/src/biomed_platform/api/` HTTP API
- `biomed_knowledge_platform/src/biomed_platform/core/` domains, ports, services, use cases
- `biomed_knowledge_platform/configs/` runtime configs, includes `rag.yaml` and `llm.yaml`

### scheduler_pubmed
FastAPI based scheduler service that queries PubMed and drives incremental ingestion into the RAG service.

Key locations:
- `scheduler_pubmed/src/api/` HTTP API
- `scheduler_pubmed/src/core/` scheduler domains, ports, services, use cases
- `scheduler_pubmed/config/` scheduler configs

### evaluation_metrics
Evaluation CLI and pipelines that generate metrics and run artifacts.
This is the primary source for JSON and CSV outputs used in thesis results.

Key locations:
- `evaluation_metrics/src/phases/` phase pipeline implementation
- `evaluation_metrics/runs/` run outputs and summaries

### tests
System and BDD tests that validate end to end behavior.

Key locations:
- `tests/bdd/features/` Gherkin features
- `tests/bdd/steps/` step definitions
- `tests/bdd/helpers/` HTTP clients and stack orchestration

## Non negotiable rules
1. Do not edit production code unless the task explicitly targets production code.
2. Do not change evaluation outputs to improve results.
3. Do not fabricate citations, DOIs, or bibliographic metadata.
4. Do not introduce numeric claims in the thesis without a traceable source.
5. Prefer small diffs. One concern per change.
6. If a task is ambiguous, stop and request clarification rather than guessing.

## Thesis evidence rules
All factual or numeric claims used in the thesis must be traceable to:
1. Internal artifacts in the repo, for example evaluation outputs under `evaluation_metrics/runs/`, or
2. External sources represented as entries in `docs/thesis/latex/src/references.bib`

When a claim is derived from internal artifacts, the chapter text must include an inline comment near the sentence pointing to the artifact path.
Example pattern:
% evidence: evaluation_metrics/runs/old/.../paper_benchmark_summary.json

## Bibliography rules
- The canonical bibliography file is `docs/thesis/latex/src/references.bib`
- Only the Bibliography Agent may edit that file.
- Citation keys must be stable and descriptive.
- Do not insert raw URLs in thesis prose as a substitute for citations.

## Figures and tables from JSON
Figures and tables used in the thesis should be generated from evaluation artifacts.
If generation tooling does not exist yet, create it under `scripts/` and write outputs into:
- `docs/thesis/latex/src/resources/` for images
- `docs/thesis/latex/src/` for LaTeX table snippets

Generated outputs must be reproducible from inputs.
Do not hand edit generated outputs except to fix formatting issues, and only if regeneration preserves the edits.

## Agent roles and edit scopes
Agents must follow file ownership and may not edit outside their scope.

### Orchestrator
Allowed edits:
- `docs/THESIS_PLAN.md`
- `docs/thesis/latex/src/main.tex`
- `docs/thesis/latex/src/table-of-contents.tex`
- `docs/thesis/latex/src/chapter-*.tex`
- `docs/thesis/latex/src/first-page.tex`
- `docs/thesis/latex/src/literature-review.tex`
- `docs/thesis/latex/src/resources/**`
- `agents/**` if present

Forbidden edits:
- `biomed_knowledge_platform/**`
- `scheduler_pubmed/**`
- `evaluation_metrics/**` except read only
- `tests/**` except read only

Primary duties:
- Maintain outline and section contracts, assign tasks, enforce gates.

### Evidence Curator
Allowed edits:
- `docs/thesis/evidence/**` if created
- `docs/thesis/latex/src/chapter-*.tex` only to add evidence comments, not prose changes
- `scripts/**` only for evidence extraction tooling

Forbidden edits:
- `docs/thesis/latex/src/references.bib`

Primary duties:
- Convert internal evaluation outputs and run summaries into a facts ledger and traceable evidence notes.

### Bibliography Agent
Allowed edits:
- `docs/thesis/latex/src/references.bib`

Forbidden edits:
- Everything else

Primary duties:
- Create and maintain BibTeX entries, ensure consistent keys and required fields.

### Metrics to Figures Agent
Allowed edits:
- `scripts/**` for JSON parsing and figure generation
- `docs/thesis/latex/src/resources/images/**`
- `docs/thesis/latex/src/resources/tables/**` if created

Forbidden edits:
- Thesis prose files except to include generated tables via `\input{}` or figure includes if requested
- Production code

Primary duties:
- Generate plots and LaTeX tables from evaluation JSON and CSV artifacts.

### Chapter Author Agents
Allowed edits:
- Exactly one chapter file under `docs/thesis/latex/src/`, for example `chapter-3.tex`

Forbidden edits:
- Any other chapter file
- `references.bib`

Primary duties:
- Write chapter prose that aligns with the plan, uses traceable evidence, and compiles.

### Technical Accuracy Reviewer
Allowed edits:
- `docs/thesis/reviews/**` if created, or `docs/thesis/latex/src/` as review comments only

Forbidden edits:
- Production code unless explicitly requested

Primary duties:
- Verify architecture descriptions, endpoints, configs, and workflows match actual code.

### Build Agen
Allowed edits:
- Root `Makefile` and `docs/thesis/latex/src/` build scripts if present
- `scripts/**` for build automation

Forbidden edits:
- Thesis content, except build wiring
- Production code

Primary duties:
- Ensure PDF build works, and DOCX conversion is repeatable.

## Default verification commands
Agents should use the closest available build commands.
Preferred:
- `make test` in each service when code changes are requested
- `make pdf` in thesis directory if defined
- Otherwise use `latexmk` invocation used by the thesis folder

If a command is missing, create a minimal one rather than relying on ad hoc local commands.

## Academic integrity
Agents must not fabricate experimental results, datasets, or citations.
Uncertainty must be stated as a limitation or future work, not presented as fact.