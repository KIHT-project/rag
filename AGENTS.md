# Repository Adapter for Codex

## Purpose

Provide the repo-local Codex adapter for a repository in the same workspace root as `engineering-wiki`.

This file must stay thin and route Codex to the canonical orchestration file plus the repo's verified local context.

## Repo Name

Write the exact repository name here from the repository metadata or folder name.

If it is not verified, write `Unknown`.

## Repo Purpose

State the repository's verified purpose in one sentence.

If the purpose is not documented yet, write `Unknown`.

## Tech Stack

List only verified technologies.

Examples of allowed entries:

- Java and Spring Boot
- Python and FastAPI
- Helm and Kubernetes
- Liquibase and Postgres

Do not guess. If unknown, write `Unknown`.

## Build Tool Current State

State the build tool actually used today.

Examples:

- Maven
- Gradle
- Unknown

## Build Tool Target State

State the target build tool only if it is explicitly known and justified.

If there is no approved migration, write `Same as current state` or `Unknown`.

## Module Structure

Describe the verified module layout and package boundaries.

Include only facts you can confirm from the repository.

## Database Migration Setup

Describe the verified migration mechanism, changelog location, and master changelog file name.

If there is no verified database migration setup, write `Unknown`.

## Testing Setup

Describe the verified unit, integration, functional, e2e, smoke, and mutation testing setup.

Mark anything unverified as `Unknown`.

## Helm And Kubernetes Setup

Describe the verified deployment and chart layout.

Include chart locations, values files, and any known deployment strategy.

## Repo Specific Deviations From Global Standards

List only explicit, verified deviations.

For each deviation, include:

- what differs
- why it differs
- whether it is current state or target state

If there are no verified deviations, write `None verified`.

## Workspace Resolution

Locate `engineering-wiki` in the workspace root, parent directory, or sibling directory, then load `engineering-wiki/agents/main-agents.md` first.

Then read the relevant standards and repo profile before making changes.

When this adapter is copied into a repository root beside `engineering-wiki`, the canonical links below resolve correctly.

## Canonical Links

Follow the global rules in:

- [`engineering-wiki/agents/main-agents.md`](../engineering-wiki/agents/main-agents.md)
- [`engineering-wiki/standards/solid-and-clean-code.md`](../engineering-wiki/standards/solid-and-clean-code.md)
- [`engineering-wiki/standards/java-spring-boot.md`](../engineering-wiki/standards/java-spring-boot.md)
- [`engineering-wiki/standards/testing-strategy.md`](../engineering-wiki/standards/testing-strategy.md)
- [`engineering-wiki/standards/java-unit-testing-spock-and-mutation.md`](../engineering-wiki/standards/java-unit-testing-spock-and-mutation.md)
- [`engineering-wiki/standards/python-testing.md`](../engineering-wiki/standards/python-testing.md)
- [`engineering-wiki/standards/backend-python-fastapi.md`](../engineering-wiki/standards/backend-python-fastapi.md)
- [`engineering-wiki/standards/functional-e2e-and-smoke-testing.md`](../engineering-wiki/standards/functional-e2e-and-smoke-testing.md)
- [`engineering-wiki/standards/liquibase-postgres.md`](../engineering-wiki/standards/liquibase-postgres.md)
- [`engineering-wiki/standards/helm-and-kubernetes.md`](../engineering-wiki/standards/helm-and-kubernetes.md)
- [`engineering-wiki/standards/yaml-configuration.md`](../engineering-wiki/standards/yaml-configuration.md)
- [`engineering-wiki/standards/local-development.md`](../engineering-wiki/standards/local-development.md)

## Adapter Instructions

1. Locate `engineering-wiki` relative to the current working directory.
2. Read `engineering-wiki/agents/main-agents.md` first.
3. Read the relevant standards before editing code.
4. Read the repo-local profile and any repo-local adapters next.
5. Inspect the repository before proposing a change.
6. Treat repo-local rules as overrides only when they are explicitly justified.
7. Do not invent missing details.
8. Mark unknowns as `Unknown` or `To be confirmed`.
