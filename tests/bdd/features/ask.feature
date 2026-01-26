Feature: Ask endpoint

  Scenario: Ask happy path
    Given a document is ingested
    And ask request is valid
    When I POST ask
    Then the response status is 200
    And request id is present
    And effective hyde enabled is false
    And answer summary is present
    And citations include the ingested doi

  Scenario: Ask with HyDE enabled
    Given a document is ingested
    And ask request is valid
    When I POST ask with hyde enabled
    Then the response status is 200
    And effective hyde enabled is true

  Scenario: Schema validation path
    Given ask request is invalid
    When I POST ask
    Then the response status is 422

  Scenario: Filters exclude non matching documents
    Given two documents are ingested
    And ask request is valid with filters year_min 2021
    When I POST ask
    Then the response status is 200
    And citations include the expected doi
    And citations do not include the excluded doi