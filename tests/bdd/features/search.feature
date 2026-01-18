Feature: Search endpoint

  Scenario: Search happy path
    Given a document is ingested
    And search request is valid
    When I POST search
    Then the response status is 200
    And request id is present
    And effective embedding model id is present
    And search hits include the ingested doi

  Scenario: Search with filters
    Given a document is ingested
    And search request is valid
    And search filters are set
    When I POST search
    Then the response status is 200
    And request id is present
    And effective embedding model id is present
    And search hits include the ingested doi

  Scenario: Schema validation path
    Given search request is invalid
    When I POST search
    Then the response status is 422
    And the error code is validation_error
    And request id is present

  Scenario: Empty result set
    Given no documents are ingested
    And search request is valid
    When I POST search
    Then the response status is 200
    And request id is present
    And effective embedding model id is present
    And search hits are empty

  Scenario: top_k limits results
    Given two documents are ingested
    And search request matches both documents
    And top_k is set to 1
    When I POST search
    Then the response status is 200
    And request id is present
    And effective embedding model id is present
    And search hits count is 1

  Scenario: Filters exclude non matching documents
    Given two documents with different years are ingested
    And search request matches both documents
    And search filters are set to year 2020 only
    When I POST search
    Then the response status is 200
    And request id is present
    And effective embedding model id is present
    And only year 2020 documents are returned
