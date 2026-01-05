# Local CI with act

## Prerequisites
* Docker running
* act installed
  * brew install act
* Permissions:
  * `chmod +x act.sh`

## How to run CI locally

Run all CI simulations (pull request, push, manual), from the root path:
```shell
./scripts/act.sh
```

Run a single event from the root path:
```shell
./scripts/act.sh pull_request
./scripts/act.sh push
./scripts/act.sh workflow_dispatch
```

Run a specific job only from the root path::
```shell
./scripts/act.sh pull_request -j tests-code-style
./scripts/act.sh pull_request -j dockerization
```