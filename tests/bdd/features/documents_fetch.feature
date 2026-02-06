Feature: Document fetch endpoint

  Scenario: Fetch by DOI with ingest
    Given PubMed has a document for doi
    When I POST fetch by doi
    Then the response status is 200
    And request id is present
    And content text source is abstract
    And full text available is false
    And ingest job id is present

  Scenario: Fetch by PMID without ingest
    Given PubMed has a document for pmid
    When I POST fetch by pmid without ingest
    Then the response status is 200
    And request id is present
    And content text source is pmc
    And full text available is true

  Scenario: Fetch by PMID with namespaced PMC without ingest
    Given PubMed has a namespaced PMC document for pmid
    When I POST fetch by pmid without ingest
    Then the response status is 200
    And request id is present
    And content text source is pmc
    And full text available is true

  Scenario: Fetch batch accepted
    Given PubMed has a document for doi
    When I POST fetch batch
    Then the response status is 202
    And request id is present
    And batch job id is present
