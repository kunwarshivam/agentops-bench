# Task Definitions

Each task is a YAML file conforming to the `Task` schema defined in `src/agentops_bench/schema.py`.

## Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier, e.g. `tool_use/001` |
| `domain` | string | Category: `tool_use`, `code`, `data_analysis`, `research` |
| `description` | string | The prompt shown to the agent |

## Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tools_available` | list[str] | `[]` | Which tools the agent may use |
| `expected_output` | string | `null` | Reference answer for deterministic scoring |
| `optimal_steps` | int | `3` | Minimum steps a perfect agent would need |
| `difficulty` | enum | `medium` | One of: easy, medium, hard, expert |
| `tags` | list[str] | `[]` | Freeform tags for filtering |
| `context` | string | `null` | Additional code or data provided to the agent |

## Directory Structure

```
tasks/
  tool_use/          # Tasks requiring web search, APIs, etc.
  code/              # Code generation, debugging, testing
  data_analysis/     # CSV/data analysis tasks
  research/          # Multi-source research and comparison
```

## Adding a New Task

1. Create a YAML file in the appropriate subdirectory.
2. Assign a unique `id` following the pattern `<domain>/<three-digit-number>`.
3. Validate with: `agentops-bench validate --tasks tasks/`

## Example

```yaml
id: "tool_use/001"
domain: "tool_use"
description: "Compare weather in Tokyo and Paris and recommend which to visit this weekend"
tools_available:
  - get_weather
  - web_search
optimal_steps: 3
difficulty: "easy"
tags:
  - weather
  - comparison
```
