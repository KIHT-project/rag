# Service X

Intro . . . 

## Table of Contents
1. [Project Structure](#project-structure)
2. [Local Run](#local-run)

## Project Structure
```
WC2025/
├── README.md
├── requirements.txt
├── .env                                  # enviroment variables for the project
├── src/
│   ├── app/ 
│   │   ├──resources/                     # Resources like datasets
│   │   ├──python/
│   │   │    ├── main.py                  # Main orchestration script
│   │   │    ├── api/                     # API
│   │   │    ├── service/                 # Scripts for business logic
│   │   │    ├── utils/                   # Utility classes
├── ├── test/
```

## API

```shell
python -m datamodel_code_generator \
  --input biomed_knowledge_platform/configs/swagger.yaml \
  --input-file-type openapi \
  --output biomed_knowledge_platform/src/biomed_platform/api/models/generated/schemas.py \
  --output-model-type pydantic_v2.BaseModel \
  --use-standard-collections \
  --use-union-operator \
  --use-annotated \
  --disable-timestamp \
  --target-python-version 3.14
```