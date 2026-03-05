# MSc Thesis Structure  
**Design and Evaluation of a Retrieval-Augmented Biomedical Platform for Thrombosis Risk Factor Extraction: Continuous PubMed Scheduling and Human-in-the-Loop Curation**

---

# 1. Introduction

## 1.1 Problem Statement
Thrombosis remains a major contributor to cardiovascular morbidity and mortality worldwide. Biomedical knowledge related to thrombosis risk factors evolves continuously, yet clinicians and researchers rely heavily on static literature reviews and keyword-based search tools. These approaches lack structured retrieval, contextual grounding, and automated freshness monitoring.

## 1.2 Motivation
Traditional keyword search systems:
- Do not leverage semantic embeddings
- Do not integrate generation models for structured extraction
- Lack continuous ingestion workflows
- Provide limited support for annotation-based validation

This thesis proposes a domain-adapted Retrieval-Augmented Generation platform designed specifically for thrombosis risk factor extraction.

## 1.3 Objectives
- Design a modular biomedical RAG platform
- Implement continuous PubMed ingestion via scheduler
- Integrate human-in-the-loop validation using Label Studio
- Evaluate retrieval, extraction, and operational performance

## 1.4 Contributions
- Domain-adapted biomedical RAG architecture
- DOI-level incremental PubMed scheduling system
- Human-in-the-loop curation integration
- Multi-phase evaluation framework
- Reproducible experiment pipeline

---

# 2. Background and Related Work

## 2.1 Biomedical Information Retrieval
- Evolution from keyword-based retrieval to semantic search
- BM25 limitations in domain-specific literature
- Embedding-based retrieval approaches

## 2.2 Retrieval-Augmented Generation in Biomedical Contexts
- RAG fundamentals
- Biomedical LLM constraints
- Risk of hallucination in clinical domains
- Hybrid retrieval pipelines

## 2.3 Risk Factor Extraction Literature
- Named Entity Recognition approaches
- Relation extraction methods
- Domain adaptation challenges

## 2.4 Knowledge Freshness and Incremental Ingestion
- PubMed APIs and update cycles
- Streaming ingestion models
- Freshness lag metrics in knowledge systems

## 2.5 Human-in-the-Loop NLP Systems
- Annotation reliability
- Inter-annotator agreement
- Feedback loop architectures

## Related Work Matrix
Include comparative table:
- Paper
- Retrieval method
- Dataset
- Metrics
- Human validation
- Limitations

---

# 3. System Architecture

## 3.1 Design Principles
- Modularity
- Reproducibility
- Observability
- Domain specialization

## 3.2 High-Level Architecture

Components:
- `biomed_knowledge_platform`
- `scheduler_pubmed`
- Vector database layer
- Embedding service
- Label Studio integration
- Evaluation module

## 3.3 Document Lifecycle

1. PubMed Query Execution
2. DOI Filtering and Deduplication
3. Metadata Extraction
4. Chunking
5. Embedding Generation
6. Vector Storage
7. Retrieval
8. Answer Generation
9. Human Annotation
10. Feedback Loop Update

## 3.4 Core Services in biomed_knowledge_platform

- Document Service
- Chunk Service
- Retrieval Service
- Answer Generation Endpoint
- Evaluation Service
- Metrics Runner

Include explanation of:
- Repository pattern
- DTO mapping
- Error handling
- Async orchestration
- Config-driven pipeline

## 3.5 Scheduler Architecture (scheduler_pubmed)

Entities:
- QueryModel
- SchedulerRun
- DOIRecord

Workflow:
- Scheduled execution
- PubMed fetch
- DOI comparison
- Incremental ingestion
- Run logging
- Partial/failure state handling

Operational metrics:
- Freshness lag
- Coverage ratio
- Run success rate

## 3.6 OpenAPI Specification

Document endpoints such as:

### /v1/search
- Method: POST
- Input:
  - query: string
  - top_k: int
  - hyde: boolean
- Output:
  - list of chunks
  - similarity scores
  - document metadata

### /v1/ask
- Method: POST
- Input:
  - question: string
  - top_k: int
  - generation parameters
- Output:
  - answer text
  - supporting chunk IDs
  - confidence score

### /v1/documents
- Method: GET
- Output:
  - paginated document list
  - ingestion metadata

### /v1/scheduler/run
- Trigger manual scheduler run
- Output:
  - run ID
  - ingestion summary

Include schema definitions:
- Document
- Chunk
- SchedulerRun
- Annotation

---

# 4. Configuration Management

## 4.1 rag.yaml
- embedding_model
- chunk_size
- overlap
- top_k
- hyde_enabled
- vector_db_config

## 4.2 llm.yaml
- model_name
- temperature
- max_tokens
- system_prompt_template

## 4.3 qdrant.yaml
- host
- port
- collection_name
- distance_metric

## 4.4 scheduler.yaml
- cron_expression
- pubmed_query
- batch_size
- retry_policy

Explain:
- Version locking
- Reproducibility strategy
- Config immutability during experiments

---

# 5. Methodology

## 5.1 Study Design

Experiment 1:
BM25 vs RAG retrieval

Experiment 2:
Static ingestion vs scheduled ingestion

Experiment 3:
Non-curated vs curated extraction

## 5.2 Dataset Protocol

- PubMed thrombosis query definition
- Inclusion criteria
- DOI filtering
- Gold annotation creation using Label Studio
- Annotation schema definition

## 5.3 Reproducibility Protocol

- Fixed config snapshots
- Run logging
- Evaluation scripts version control
- Random seed control

---

# 6. Implementation Details

## 6.1 Embedding Strategy
- Model selection rationale
- Biomedical domain adaptation
- Chunk size tradeoffs

## 6.2 Retrieval Logic
- Vector similarity
- Top-k selection
- HyDE optional query expansion

## 6.3 Error Handling
- PubMed API failures
- Duplicate DOI prevention
- Partial ingestion recovery

## 6.4 Testing Strategy
- Unit tests
- BDD tests
- Integration tests
- CI pipeline
- Coverage tracking

---

# 7. Evaluation

## 7.1 Experimental Setup
- Hardware environment
- Model versions
- Dataset size
- Config parameters

## 7.2 Retrieval Metrics
- Precision@k
- Recall@k
- nDCG

Include mathematical definitions.

## 7.3 Extraction Metrics
- Precision
- Recall
- F1 score

## 7.4 Operational Metrics
- Freshness lag
- Run success rate
- Ingestion latency

## 7.5 Results
- Tables
- Plots
- Comparative graphs

## 7.6 Error Analysis
- Failure examples
- Misclassification patterns
- Retrieval miss cases

---

# 8. Discussion

## 8.1 Hypothesis Validation
Map RQ1–RQ3 to measured results.

## 8.2 Strengths
- Modular architecture
- Continuous ingestion
- Human validation integration

## 8.3 Limitations
- PubMed query bias
- Annotation bias
- Model dependency

## 8.4 Scalability Considerations
- Vector DB scaling
- Scheduler throughput
- Domain transfer potential

## 8.5 Ethical Considerations
- Clinical misinterpretation risks
- Bias in biomedical literature
- Responsible AI use

---

# 9. Conclusion and Future Work

- Summary of validated hypotheses
- Engineering contributions
- Research implications
- Future extensions:
  - Multi-domain biomedical support# MSc Thesis Structure  
**Design and Evaluation of a Retrieval-Augmented Biomedical Platform for Thrombosis Risk Factor Extraction: Continuous PubMed Scheduling and Human-in-the-Loop Curation**

---

# 1. Introduction

## 1.1 Problem Statement
Thrombosis remains a major contributor to cardiovascular morbidity and mortality worldwide. Biomedical knowledge related to thrombosis risk factors evolves continuously, yet clinicians and researchers rely heavily on static literature reviews and keyword-based search tools. These approaches lack structured retrieval, contextual grounding, and automated freshness monitoring.

## 1.2 Motivation
Traditional keyword search systems:
- Do not leverage semantic embeddings
- Do not integrate generation models for structured extraction
- Lack continuous ingestion workflows
- Provide limited support for annotation-based validation

This thesis proposes a domain-adapted Retrieval-Augmented Generation platform designed specifically for thrombosis risk factor extraction.

## 1.3 Objectives
- Design a modular biomedical RAG platform
- Implement continuous PubMed ingestion via scheduler
- Integrate human-in-the-loop validation using Label Studio
- Evaluate retrieval, extraction, and operational performance

## 1.4 Contributions
- Domain-adapted biomedical RAG architecture
- DOI-level incremental PubMed scheduling system
- Human-in-the-loop curation integration
- Multi-phase evaluation framework
- Reproducible experiment pipeline

---

# 2. Background and Related Work

## 2.1 Biomedical Information Retrieval
- Evolution from keyword-based retrieval to semantic search
- BM25 limitations in domain-specific literature
- Embedding-based retrieval approaches

## 2.2 Retrieval-Augmented Generation in Biomedical Contexts
- RAG fundamentals
- Biomedical LLM constraints
- Risk of hallucination in clinical domains
- Hybrid retrieval pipelines

## 2.3 Risk Factor Extraction Literature
- Named Entity Recognition approaches
- Relation extraction methods
- Domain adaptation challenges

## 2.4 Knowledge Freshness and Incremental Ingestion
- PubMed APIs and update cycles
- Streaming ingestion models
- Freshness lag metrics in knowledge systems

## 2.5 Human-in-the-Loop NLP Systems
- Annotation reliability
- Inter-annotator agreement
- Feedback loop architectures

## Related Work Matrix
Include comparative table:
- Paper
- Retrieval method
- Dataset
- Metrics
- Human validation
- Limitations

---

# 3. System Architecture

## 3.1 Design Principles
- Modularity
- Reproducibility
- Observability
- Domain specialization

## 3.2 High-Level Architecture

Components:
- `biomed_knowledge_platform`
- `scheduler_pubmed`
- Vector database layer
- Embedding service
- Label Studio integration
- Evaluation module

## 3.3 Document Lifecycle

1. PubMed Query Execution
2. DOI Filtering and Deduplication
3. Metadata Extraction
4. Chunking
5. Embedding Generation
6. Vector Storage
7. Retrieval
8. Answer Generation
9. Human Annotation
10. Feedback Loop Update

## 3.4 Core Services in biomed_knowledge_platform

- Document Service
- Chunk Service
- Retrieval Service
- Answer Generation Endpoint
- Evaluation Service
- Metrics Runner

Include explanation of:
- Repository pattern
- DTO mapping
- Error handling
- Async orchestration
- Config-driven pipeline

## 3.5 Scheduler Architecture (scheduler_pubmed)

Entities:
- QueryModel
- SchedulerRun
- DOIRecord

Workflow:
- Scheduled execution
- PubMed fetch
- DOI comparison
- Incremental ingestion
- Run logging
- Partial/failure state handling

Operational metrics:
- Freshness lag
- Coverage ratio
- Run success rate

## 3.6 OpenAPI Specification

Document endpoints such as:

### /v1/search
- Method: POST
- Input:
  - query: string
  - top_k: int
  - hyde: boolean
- Output:
  - list of chunks
  - similarity scores
  - document metadata

### /v1/ask
- Method: POST
- Input:
  - question: string
  - top_k: int
  - generation parameters
- Output:
  - answer text
  - supporting chunk IDs
  - confidence score

### /v1/documents
- Method: GET
- Output:
  - paginated document list
  - ingestion metadata

### /v1/scheduler/run
- Trigger manual scheduler run
- Output:
  - run ID
  - ingestion summary

Include schema definitions:
- Document
- Chunk
- SchedulerRun
- Annotation

---

# 4. Configuration Management

## 4.1 rag.yaml
- embedding_model
- chunk_size
- overlap
- top_k
- hyde_enabled
- vector_db_config

## 4.2 llm.yaml
- model_name
- temperature
- max_tokens
- system_prompt_template

## 4.3 qdrant.yaml
- host
- port
- collection_name
- distance_metric

## 4.4 scheduler.yaml
- cron_expression
- pubmed_query
- batch_size
- retry_policy

Explain:
- Version locking
- Reproducibility strategy
- Config immutability during experiments

---

# 5. Methodology

## 5.1 Study Design

Experiment 1:
BM25 vs RAG retrieval

Experiment 2:
Static ingestion vs scheduled ingestion

Experiment 3:
Non-curated vs curated extraction

## 5.2 Dataset Protocol

- PubMed thrombosis query definition
- Inclusion criteria
- DOI filtering
- Gold annotation creation using Label Studio
- Annotation schema definition

## 5.3 Reproducibility Protocol

- Fixed config snapshots
- Run logging
- Evaluation scripts version control
- Random seed control

---

# 6. Implementation Details

## 6.1 Embedding Strategy
- Model selection rationale
- Biomedical domain adaptation
- Chunk size tradeoffs

## 6.2 Retrieval Logic
- Vector similarity
- Top-k selection
- HyDE optional query expansion

## 6.3 Error Handling
- PubMed API failures
- Duplicate DOI prevention
- Partial ingestion recovery

## 6.4 Testing Strategy
- Unit tests
- BDD tests
- Integration tests
- CI pipeline
- Coverage tracking

---

# 7. Evaluation

## 7.1 Experimental Setup
- Hardware environment
- Model versions
- Dataset size
- Config parameters

## 7.2 Retrieval Metrics
- Precision@k
- Recall@k
- nDCG

Include mathematical definitions.

## 7.3 Extraction Metrics
- Precision
- Recall
- F1 score

## 7.4 Operational Metrics
- Freshness lag
- Run success rate
- Ingestion latency

## 7.5 Results
- Tables
- Plots
- Comparative graphs

## 7.6 Error Analysis
- Failure examples
- Misclassification patterns
- Retrieval miss cases

---

# 8. Discussion

## 8.1 Hypothesis Validation
Map RQ1–RQ3 to measured results.

## 8.2 Strengths
- Modular architecture
- Continuous ingestion
- Human validation integration

## 8.3 Limitations
- PubMed query bias
- Annotation bias
- Model dependency

## 8.4 Scalability Considerations
- Vector DB scaling
- Scheduler throughput
- Domain transfer potential

## 8.5 Ethical Considerations
- Clinical misinterpretation risks
- Bias in biomedical literature
- Responsible AI use

---

# 9. Conclusion and Future Work

- Summary of validated hypotheses
- Engineering contributions
- Research implications
- Future extensions:
  - Multi-domain biomedical support
  - Advanced extraction models
  - Improved annotation feedback loops
  - Clinical validation studies

---

# 10. Appendices

## A. Full OpenAPI Schema
Complete endpoint definitions and models.

## B. Configuration Snapshots
rag.yaml
llm.yaml
qdrant.yaml
scheduler.yaml

## C. PubMed Query Strings
Full reproducible queries.

## D. Annotation Schema
Label Studio configuration.

## E. Evaluation Scripts
Command examples and usage.

## F. Prompt Templates
Extraction prompts
System messages
HyDE prompt variations

## G. Ethics and Data Governance Notes
Data sources
Usage boundaries
Reproducibility guarantees

  - Advanced extraction models
  - Improved annotation feedback loops
  - Clinical validation studies

---

# 10. Appendices

## A. Full OpenAPI Schema
Complete endpoint definitions and models.

## B. Configuration Snapshots
rag.yaml
llm.yaml
qdrant.yaml
scheduler.yaml

## C. PubMed Query Strings
Full reproducible queries.

## D. Annotation Schema
Label Studio configuration.

## E. Evaluation Scripts
Command examples and usage.

## F. Prompt Templates
Extraction prompts
System messages
HyDE prompt variations

## G. Ethics and Data Governance Notes
Data sources
Usage boundaries
Reproducibility guarantees
