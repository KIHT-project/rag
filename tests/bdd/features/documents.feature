Feature: Document delete endpoint

  Scenario: Delete happy path
    Given a document is ingested
    When I DELETE the document by doi
    Then the response status is 204
    And request id is present
    When I DELETE the document by doi again
    Then the response status is 404
    And the error code is not_found

  Scenario: Not found path
    When I DELETE an unknown document
    Then the response status is 404
    And the error code is not_found
    And request id is present

  Scenario: Invalid DOI path
    When I DELETE a document with invalid doi
    Then the response status is 400
    And the error code is validation_error

  Scenario: Get document by DOI
    Given a document is ingested
    When I GET the document by doi
    Then the response status is 200
    And request id is present
    And the document doi matches

  Scenario: List DOIs simple
    Given a document is ingested
    When I GET the doi list
    Then the response status is 200
    And request id is present
    And the doi list includes the ingested doi

  Scenario: List DOIs with document info
    Given a document is ingested
    When I GET the doi list with document info
    Then the response status is 200
    And request id is present
    And the document list includes the ingested doi
