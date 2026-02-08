@scheduler
Feature: Scheduler control endpoints

  Scenario: Trigger scheduler run returns accepted response
    Given the scheduler control API is ready
    When I trigger the scheduler run
    Then the scheduler control response status is 202
    And the scheduler run status is RUNNING
    And the scheduler run id exists

  Scenario: Trigger scheduler run with reldate override returns accepted response
    Given the scheduler control API is ready
    When I trigger the scheduler run with 365 days lookback
    Then the scheduler control response status is 202
    And the scheduler run status is RUNNING
    And the scheduler run id exists

  Scenario: Scheduler status endpoint returns configured status
    Given the scheduler control API is ready
    When I get scheduler control status
    Then the scheduler control response status is 200
    And scheduler status contains utc schedule
    And scheduler status enabled flag is true
