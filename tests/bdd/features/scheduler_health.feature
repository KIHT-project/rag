@scheduler @http
Feature: Scheduler health endpoints

  Scenario: Scheduler health endpoint responds ok
    Given the scheduler API is running
    When I call the scheduler health endpoint
    Then the scheduler response status is 200
    And the scheduler health status is ok
    And the scheduler OpenAPI title is BDD Testing - PubMed Scheduler

  Scenario: Scheduler live endpoint responds alive
    Given the scheduler API is running
    When I call the scheduler live endpoint
    Then the scheduler response status is 200
    And the scheduler live status is alive
    And the scheduler OpenAPI title is BDD Testing - PubMed Scheduler

  Scenario: Scheduler ready endpoint responds ready
    Given the scheduler API is running
    When I call the scheduler ready endpoint
    Then the scheduler response status is 200
    And the scheduler ready status is ready
    And the scheduler OpenAPI title is BDD Testing - PubMed Scheduler
