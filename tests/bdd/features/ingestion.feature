Feature: Ingestion endpoint

  Scenario: Ingest happy path
    Given ingestion payload is valid
    When I POST ingest
    Then the response status is 202
    And I can poll the job until terminal state
    And request id is present
    And the item status counts are internally consistent

  Scenario: Duplicate DOI in same request
    Given ingestion payload has duplicate doi in same request
    When I POST ingest
    Then the response status is 202
    And I can poll the job until terminal state
    And the job state is succeeded
    And the item status counts are internally consistent

  Scenario: Idempotency happy path
    Given ingestion payload is valid
    And idempotency key is set
    When I POST ingest
    Then the response status is 202
    And I capture the job id
    When I POST ingest again with same idempotency key and same body
    Then the response status is 202
    And the job id is the same as before

  Scenario: Idempotency conflict path
    Given ingestion payload is valid
    And idempotency key is set
    When I POST ingest
    Then the response status is 202
    When I POST ingest again with same idempotency key but different body
    Then the response status is 400
    And the error code is validation_error

  Scenario: Not found path
    When I GET an unknown job id
    Then the response status is 404
    And the error code is not_found
    And request id is present

  Scenario: Schema validation path
    Given ingestion payload is invalid
    When I POST ingest
    Then the response status is 422
    And the error code is validation_error
    And request id is present

  Scenario: Section-aware chunking excludes references
    Given ingestion payload has sectioned content
    When I POST ingest
    Then the response status is 202
    And I can poll the job until terminal state
    And the document content excludes section titles
    And the document sections include expected titles
    And the document content excludes references
