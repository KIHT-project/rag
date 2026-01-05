# Local CI with act

## Prerequisites
* Docker running
* act installed
  * brew install act
* Permissions:
  * `chmod +x act.sh`

## How to run CI locally

Run all CI simulations (pull request, push, manual):
```shell
act.sh
```

Run a single event:
```shell
act.sh pull_request
act.sh push
act.sh workflow_dispatch
```

Run a specific job only:
```shell
act.sh pull_request -j test-lint
```