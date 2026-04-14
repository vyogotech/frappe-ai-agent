Feature: Error Handling
  As a Frappe Copilot user
  I want graceful error handling
  So that I always get a useful response

  Scenario: MCP server is unreachable
    Given the agent is running with mock LLM
    And the MCP server is down
    When I send a chat message "Show me customers"
    Then I receive an error event with code "AGENT_ERROR"
    And I receive a done event

  Scenario: Rate limit exceeded
    Given the agent is running with a rate limit of 2 per minute
    When I send 3 chat messages within 1 second
    Then the third message receives a "RATE_LIMITED" error

  Scenario: Invalid JWT token
    Given the agent is running
    When I connect with an invalid token
    Then the WebSocket is closed with code 4001
