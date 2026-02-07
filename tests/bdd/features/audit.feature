Feature: API audit persistence

  Scenario: Successful API request is audited in postgres
    When I GET health with request id audit-success-1
    Then audit scenario response status is 200
    And audit request record exists for audit-success-1 with status SUCCESS
    And audit events contain API_REQUEST_RECEIVED and API_RESPONSE_SENT for audit-success-1

  Scenario: Validation error is audited with stacktrace
    Given audit ask request is invalid
    When I POST ask with request id audit-validation-1
    Then audit scenario response status is 422
    And audit request record exists for audit-validation-1 with status ERROR
    And audit error exists for audit-validation-1 with exception class RequestValidationError

  Scenario: Duplicate request ids are audited as separate rows
    When I GET health twice with request id audit-dup-1
    Then audit request row count for audit-dup-1 is 2

  Scenario: Not found route is audited as error
    When I GET missing route with request id audit-404-1
    Then audit scenario response status is 404
    And audit request record exists for audit-404-1 with status ERROR
    And audit error exists for audit-404-1 with exception class HTTPException

  Scenario: System error path is audited with stacktrace
    When I POST search with broken service and request id audit-500-1
    Then audit scenario response status is 500
    And audit request record exists for audit-500-1 with status ERROR
    And audit error exists for audit-500-1 with exception class SystemError

  Scenario: Search request stores request and response raw payloads
    When I POST search with request id audit-search-200-1
    Then audit scenario response status is 200
    And audit request body and response body are persisted for audit-search-200-1
    And audit event sequence is complete for audit-search-200-1

  Scenario: Duplicate request id across endpoints is tracked separately
    When I call health then missing route with request id audit-dup-cross-1
    Then audit request row count for audit-dup-cross-1 is 2
    And audit request rows for audit-dup-cross-1 include paths /health and /missing-audit-path
