# Agent Registry & Intelligent Workflow Orchestration

An extensible, multi-agent orchestration platform that interprets organizational requests, identifies the relevant business process, decomposes it into actionable tasks, and dynamically delegates each task to the most suitable specialized agent.

The project combines semantic retrieval, LLM-based task planning, agent selection, and tool-enabled execution to coordinate employee lifecycle, department administration, and organizational notifications through a registry-driven architecture.

## Overview

Traditional automation often relies on fixed workflows and hard-coded routing rules. This project explores a more flexible approach: requests are interpreted in context, matched against documented organizational processes, split into ordered subtasks, and routed to agents according to their registered capabilities.

The registry acts as the discovery layer for specialized agents. Each agent is described by metadata such as its name, responsibilities, skill set, available tools, and service endpoint. Semantic similarity helps identify relevant process and agent candidates, while an LLM performs structured task decomposition and agent selection.

## Key Capabilities

- **Agent registry and capability discovery** — Maintains agent metadata, descriptions, skill sets, tool names, and service endpoints in JSON configuration.
- **Semantic process retrieval** — Uses Sentence Transformers embeddings to identify the organizational process most relevant to an incoming request.
- **LLM-based task decomposition** — Converts a request and its matching process definition into ordered, actionable subtasks.
- **Intelligent agent selection** — Combines semantic similarity with structured LLM selection to map each subtask to a suitable specialized agent.
- **Distributed agent execution** — Delegates subtasks to agent service endpoints through HTTP APIs.
- **MCP-based tool integration** — Exposes employee and notification capabilities through dedicated Model Context Protocol servers.
- **A2A agent metadata** — Includes agent descriptions and endpoint information intended for Agent-to-Agent integration.
- **Human-in-the-loop routing concepts** — Includes routing logic for execution, clarification/approval, and rejection decisions.
- **Organizational process coordination** — Models cross-functional activities involving employees, departments, HR, IT Support, and assignee teams.

## Architecture

At a high level, the primary orchestration flow is:

1. **Request intake** — The FastAPI service receives an organizational request.
2. **Process matching** — The request is embedded using `all-MiniLM-L6-v2` and compared with precomputed organizational-process embeddings.
3. **Task planning** — The selected process definition and user request are passed to an LLM task-splitting agent, which returns structured, ordered subtasks.
4. **Agent discovery** — Each subtask is embedded and compared with registered agent embeddings to retrieve relevant agent metadata.
5. **Agent selection** — An LLM selects the most appropriate registered agent based on its responsibilities, skills, and tools.
6. **Task delegation** — The orchestrator sends the subtask context to the selected agent's configured HTTP endpoint.
7. **Result collection** — Responses from the delegated agents are collected and returned by the orchestration API.

```text
                         User Request
                              |
                              v
                    FastAPI Orchestrator
                              |
                              v
                  Semantic Process Retrieval
                              |
                              v
                 Organizational Process Match
                              |
                              v
                    LLM Task Decomposition
                              |
                              v
                    Ordered Subtasks
                              |
                              v
                    Semantic Agent Search
                              |
                              v
                     LLM Agent Selection
                              |
                              v
                    Agent Registry Metadata
                              |
                              v
                    HTTP Task Delegation
                       /      |       \
                      v       v        v
                Employee  Department  Notification
                  Agent      Agent       Agent
                      \       |        /
                       v      v       v
                    Aggregated Responses
                              |
                              v
                         API Response
```

> The repository contains multiple implementation paths and supporting experiments, including the semantic orchestration flow, MCP-backed employee and notification agents, and a separate smart-router implementation. The diagram describes the primary flow implemented in `main.py`; it should not be interpreted as a claim that every component is wired into one production workflow.

## Specialized Agents

### Employee Agent

Handles employee-related operations, including:

- Employee onboarding
- Employee deboarding
- Employee information retrieval
- Employee detail validation

The employee MCP server exposes employee capabilities as MCP tools.

### Department Agent

Designed for department administration, including:

- Department creation
- Department retrieval
- Department updates
- Department removal

The department agent is represented in the registry metadata. Its runtime implementation and endpoint configuration should be aligned with the deployment setup.

### Notification Agent

Provides organizational communication capabilities, including:

- Sending notifications
- Broadcasting notifications to multiple recipients
- Scheduling notifications
- Scheduling meeting notifications

The notification MCP server exposes these operations as MCP tools.

## Organizational Processes

The process definitions in `organization_process.txt` cover:

1. Employee onboarding
2. Employee deboarding
3. Department onboarding
4. Department dismantling
5. Employee department transfer
6. Employee detail update
7. Department detail update

These process definitions provide the business context used during semantic retrieval and task decomposition. They describe cross-functional steps such as validation, approvals, HR and IT notifications, employee movement, and knowledge-transfer coordination.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.12+ |
| API layer | FastAPI, Uvicorn |
| Agent framework | OpenAI Agents SDK |
| Model integration | LLMs configured through the Agents SDK |
| Semantic retrieval | Sentence Transformers, `all-MiniLM-L6-v2` |
| Similarity search | NumPy, PyTorch, Sentence Transformers utilities |
| Task planning | Pydantic structured outputs, LLM-based task decomposition |
| Agent/tool interoperability | Model Context Protocol (MCP), A2A SDK dependencies and agent metadata |
| Service communication | HTTP, Requests |
| Configuration | JSON, environment variables, python-dotenv |

## Repository Structure

```text
agent_registry-main/
├── main.py                         # Primary semantic orchestration API
├── main_v2.py                      # Alternative orchestration implementation
├── agents_api.py                   # FastAPI application hosting employee,
│                                   # department, and notification agents
├── agents.json                     # Agent registry metadata
├── a2a_agents_info.json             # A2A-oriented agent descriptions/endpoints
├── task_spliter.py                 # LLM-based task decomposition
├── init.py                         # Agent and process embedding utilities
├── app_util.py                     # Request routing and human-feedback utilities
├── employee_mcp_server.py          # Employee MCP tools
├── notification_mcp_server.py      # Notification MCP tools
├── organization_process.txt        # Organizational process definitions
├── agents_embedding.npy            # Precomputed agent embeddings
├── organization_process_embed.npy  # Precomputed process embeddings
├── sql_init.py                     # SQL initialization utilities
├── smart router architecture/      # Separate routing design and models
├── pyproject.toml                  # Project configuration and dependencies
└── requirements.txt                # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.12 or later
- An LLM API key supported by the configured Agents SDK model
- `uv` (recommended) or `pip`

### 1. Clone the repository

```bash
git clone <repository-url>
cd agent_registry-main
```

### 2. Configure environment variables

Create a `.env` file in the project root and configure the credentials required by your selected model provider.

```env
OPENAI_API_KEY=your_api_key
```

If you use a non-OpenAI model through the alternative model integration, configure the corresponding base URL and API key as required by that implementation.

### 3. Install dependencies

Using `uv`:

```bash
uv sync
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

### 4. Start the agent service

In one terminal, start the FastAPI application that hosts the agent endpoints:

```bash
python agents_api.py
```

The service is configured to listen on:

```text
http://localhost:8011
```

### 5. Start the orchestrator

In another terminal, start the primary orchestration API:

```bash
python main.py
```

The orchestrator is configured to listen on:

```text
http://localhost:8000
```

### 6. Submit a request

The primary orchestration endpoint is:

```http
GET /agent?query=<your-organizational-request>
```

Example:

```bash
curl --get "http://localhost:8000/agent" \
  --data-urlencode "query=Onboard a new employee named John Smith in the Engineering department"
```

The orchestrator attempts to match the request to an organizational process, decompose it into subtasks, select agents, and delegate the resulting work.

> Run the services and endpoints that correspond to the implementation you intend to test. The repository includes alternative and experimental components; their availability does not imply that all routes or agents are integrated into the primary flow.

## Agent Registry Configuration

Agent definitions are maintained in `agents.json`. Each entry describes the agent and the information used by the orchestrator for discovery and delegation.

Example structure:

```json
{
  "agent_name": "employee_agent",
  "description": "Specialized agent for employee lifecycle operations.",
  "skill_set": [
    "Employee management",
    "Employee onboarding",
    "Employee deboarding"
  ],
  "endpoint": "http://localhost:8011/employee",
  "tools": [
    "get_employees",
    "onboard_new_employee",
    "deboard_employee"
  ]
}
```

To add a new specialized agent:

1. Implement the agent service and its required tools.
2. Expose the service through an endpoint reachable by the orchestrator.
3. Add the agent's name, description, skill set, endpoint, and tool names to `agents.json`.
4. Regenerate agent embeddings using the project's embedding utility when the registry content changes.
5. Verify that the selected agent name maps to the correct registry entry and endpoint.

Keep the registry metadata consistent with the actual tools and API routes exposed by each agent service.

## MCP Tool Servers

The employee and notification MCP servers use the stdio transport and are configured in `agents_api.py`.

- `employee_mcp_server.py` — employee onboarding, deboarding, retrieval, and validation tools.
- `notification_mcp_server.py` — notification, broadcast, scheduling, and meeting-notification tools.

The MCP servers are started by the agent application's lifespan configuration. Start the agent application with the required Python environment and working directory so the configured server commands can resolve the modules.

## Human-in-the-Loop Routing

The repository also contains routing utilities and a separate smart-router implementation that define three routing outcomes:

| Route | Intended behavior |
|---|---|
| `EXECUTE` | The request is sufficiently clear and can proceed. |
| `HUMAN_IN_LOOP_FLOW` | Clarification, missing information, or human approval is required. |
| `REJECT` | The request is outside the supported scope or should not be processed. |

This design is intended to prevent the system from guessing critical business information and to provide a controlled path for requests that require human input. Review the relevant router implementation before relying on it for approval enforcement in a live environment.

## Configuration and Operational Notes

- Keep API keys and credentials in environment variables; do not commit `.env` files.
- Ensure the endpoints in `agents.json` match the host and port where the agent services are running.
- Regenerate `.npy` embedding files whenever agent metadata or organizational process definitions change.
- Ensure the registry, process definitions, tool descriptions, and actual service implementations remain synchronized.
- The sample MCP tools currently include demonstration-style behavior; connect them to the intended persistence and notification systems before using them for real organizational operations.
- Add authentication, authorization, request validation, audit logging, retries, timeout policies, and operational monitoring before exposing services beyond a trusted development environment.
- Validate approval requirements in the execution layer, not only in LLM instructions or routing prompts.

## Current Scope

This repository is a development implementation and architectural foundation for registry-driven, multi-agent organizational workflow automation. It demonstrates semantic process matching, structured task decomposition, capability-based agent selection, HTTP delegation, MCP tool integration, and human-in-the-loop routing concepts.

It should not be considered production-ready without further engineering, including persistent business-system integrations, robust workflow state management, enforceable authorization and approval controls, service security, observability, and end-to-end testing.

## License

Add the license and usage terms applicable to this repository.
