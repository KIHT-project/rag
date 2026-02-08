# Tests

## BDD:
Set up the environment (containers):
```shell
make bdd-up
```

Set up HTTP mode (includes both APIs on ports 8000 and 9000):
```shell
make bdd-http-up
```
Core API: `http://localhost:8000`
Scheduler API: `http://localhost:9000`

Run all BDD suites:
```shell
make bdd
```

Run only core platform BDD:
```shell
make bdd-core
```

Run only scheduler BDD:
```shell
make bdd-scheduler
```

Run HTTP mode BDD (real HTTP against dockerized APIs):
```shell
make bdd-http
```

Close the containers:
```shell
make bdd-down
```
