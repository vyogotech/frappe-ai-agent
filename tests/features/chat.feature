Feature: Chat with the agent

  The agent accepts SSE POST requests at /api/v1/chat. It requires a
  Frappe `sid` cookie and forwards that sid to downstream tool calls.
  Tool permission errors become LLM-readable tool-result messages, not
  SSE errors. Upstream LLM or MCP failures become a single SSE error
  event and the stream closes cleanly.

  Background:
    Given a running agent app with a stubbed chat service

  Scenario: Missing sid is rejected with 401
    When I POST "hello" to /api/v1/chat without a sid cookie
    Then I receive HTTP 401
    And no SSE stream is opened

  Scenario: User with permission gets an answer
    Given the stub chat service will yield status, a tool_call, content, and done
    When I POST "show me unpaid invoices" to /api/v1/chat with sid "valid-sid"
    Then I receive HTTP 200
    And the response content-type is "text/event-stream"
    And the stream contains a status event
    And the stream contains a tool_call event
    And the stream contains at least one content event
    And the stream ends with a done event
    And the stream does not contain an error event

  Scenario: User without permission is told gracefully
    Given the stub chat service will yield a content event explaining access was denied and then done
    When I POST "show me unpaid invoices" to /api/v1/chat with sid "limited-sid"
    Then I receive HTTP 200
    And the stream contains a content event mentioning access denied
    And the stream ends with a done event
    And the stream does not contain an error event

  Scenario: Ollama unavailable produces one SSE error event
    Given the stub chat service will yield an error event and then done
    When I POST "hi" to /api/v1/chat with sid "valid-sid"
    Then I receive HTTP 200
    And the stream contains exactly one error event
    And the error message mentions unavailable
    And the stream ends with a done event
