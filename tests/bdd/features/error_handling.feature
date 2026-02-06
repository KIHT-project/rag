Feature: Error handling

  Scenario: Request validation errors are normalized
    Given a local test app with error handlers
    And an endpoint with strict payload schema
    When I POST invalid payload to the strict endpoint
    Then the response status is 422
    And the error code is validation_error
    And request id is present or none

  Scenario: Unexpected exceptions are normalized
    Given a local test app with error handlers
    And an endpoint that raises an unexpected exception
    When I call the exploding endpoint
    Then the response status is 500
    And the error code is system_error
    And request id is present or none

  Scenario: Not found routes are normalized
    Given a local test app with error handlers
    When I call a missing route on the local app
    Then the response status is 404
    And the error code is not_found
