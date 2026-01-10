Feature: Readiness endpoint

  Scenario: Readiness check path
    Given the API is running
    When I call the readiness endpoint
    Then the response status is 200
    And the readiness status is ready
    And qdrant is ok and llm is ok
