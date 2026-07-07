Author
Nikolaos Chaikalis

Supervising Professor
Eleni Kaldoudi

Supervising PhD Candidate
Nikolaos Portokalidis

This software was developed as part of an MSc thesis, Democritus University of Thrace, School of Medicine MSc in Bioinformatics.
Reuse in academic research requires proper attribution.
See CITATION.cff.

## Short Description of the Projects
- `biomed_knowledge_platform`: Main biomedical RAG API and core platform service.
- `evaluation_metrics`: Phase-based evaluation pipeline for retrieval and answer behavior.
- `scheduler_pubmed`: Standalone scheduler service scaffold for PubMed orchestration.
- `llm_server`: Local LLM serving stack with Ollama, Open WebUI, and Traefik.
- `tests`: BDD test harness and commands for running integration scenarios.
- `scripts`: Utility scripts, including local CI simulation with `act`.

## Table of Contents for Other READMEs
1. [Core Service README](biomed_knowledge_platform/README.md)
2. [Evaluation Metrics README](evaluation_metrics/README.md)
3. [PubMed Scheduler README](scheduler_pubmed/README.md)
4. [BDD Tests README](tests/README.md)
5. [Scripts README](scripts/README.md)

## Python Setup
Set up a virtual environment and install dependencies:

```shell
pyenv install 3.14.2
pyenv local 3.14.2
python --version
python -m venv .venv

# macOS
source .venv/bin/activate

# Windows
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

To deactivate the virtual environment:

```shell
deactivate
```
cd ../engineering-wiki

make preflight repo=drone-communication-simulator
make check-profile repo=drone-communication-simulator