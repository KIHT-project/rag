@scheduler
Feature: Scheduler run endpoints

  Scenario: List scheduler runs
    Given the scheduler runs API is ready
    When I list scheduler runs
    Then the scheduler runs response status is 200
    And scheduler runs list contains one run

  Scenario: List scheduler runs filtered by status
    Given the scheduler runs API is ready
    When I list scheduler runs with status FAILED
    Then the scheduler runs response status is 200
    And all listed scheduler runs have status FAILED

  Scenario: List scheduler runs filtered by from-to range
    Given the scheduler runs API is ready
    When I list scheduler runs from "2026-02-08T12:00:00Z" to "2026-02-08T14:00:00Z"
    Then the scheduler runs response status is 200
    And scheduler runs are within the requested range

  Scenario: List scheduler runs with invalid from-to range
    Given the scheduler runs API is ready
    When I list scheduler runs from "2026-02-09T14:00:00Z" to "2026-02-08T14:00:00Z"
    Then the scheduler runs response status is 400
    And scheduler runs error code is validation_error

  Scenario: Get scheduler run by id
    Given the scheduler runs API is ready
    And a scheduler run id is available
    When I get the scheduler run by id
    Then the scheduler runs response status is 200
    And the scheduler run id matches the requested id

  Scenario: List scheduler run doi results
    Given the scheduler runs API is ready
    And a scheduler run id is available
    When I list scheduler run dois
    Then the scheduler runs response status is 200
    And scheduler run doi results contain ingested status

  Scenario: Missing scheduler run returns not found
    Given the scheduler runs API is ready
    When I get a missing scheduler run
    Then the scheduler runs response status is 404
