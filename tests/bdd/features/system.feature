@http
Feature: System endpoints

  Scenario: Health check path
    Given the API is running
    When I call the health endpoint
    Then the response status is 200
    And the response body contains status ok
    And the core OpenAPI title is BDD Testing - Biomedical Knowledge Platform
