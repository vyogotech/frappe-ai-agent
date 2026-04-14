Feature: Chat Flow
  As a Frappe Copilot user
  I want to ask questions about my ERPNext data
  So that I can get structured visual answers

  Scenario: User sends a simple text question
    Given the agent is running with mock LLM
    And the mock LLM returns "Hello! How can I help?"
    When I send a chat message "Hi there"
    Then I receive an ack event
    And I receive a content_block event with type "text"
    And I receive a done event

  Scenario: User asks a data question requiring tools
    Given the agent is running with mock LLM
    And the mock LLM calls tool "list_documents" with doctype "Customer"
    And the tool returns 3 customers
    And the mock LLM returns a table block
    When I send a chat message "Show me my customers"
    Then I receive an ack event
    And I receive a tool_start event for "list_documents"
    And I receive a tool_end event with success
    And I receive a content_block event with type "table"
    And I receive a done event
