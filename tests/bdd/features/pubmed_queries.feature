@scheduler
Feature: Scheduler PubMed query endpoints

  Scenario: Create and get PubMed query
    Given the scheduler query API is ready
    When I create a scheduler PubMed query
    Then the scheduler query response status is 201
    And the created scheduler query has an id
    When I get the created scheduler PubMed query by id
    Then the scheduler query response status is 200
    And the fetched scheduler query id matches the created id

  Scenario: Update query and toggle enabled status
    Given the scheduler query API is ready
    And a scheduler PubMed query exists
    When I update the scheduler PubMed query
    Then the scheduler query response status is 200
    And the scheduler query description is updated
    When I disable the scheduler PubMed query
    Then the scheduler query response status is 200
    And the scheduler query is disabled
    When I enable the scheduler PubMed query
    Then the scheduler query response status is 200
    And the scheduler query is enabled

  Scenario: List queries filtered by enabled
    Given the scheduler query API is ready
    And I have both enabled and disabled scheduler queries
    When I list scheduler PubMed queries with enabled true
    Then the scheduler query response status is 200
    And all listed scheduler queries are enabled
    When I list scheduler PubMed queries with enabled false
    Then the scheduler query response status is 200
    And all listed scheduler queries are disabled
