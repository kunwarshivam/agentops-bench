# AgentOps-Bench

**A Benchmark for Operational Reliability of LLM Agent Systems**

<!-- badges -->
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## What It Measures

AgentOps-Bench evaluates LLM agent systems across **6 operational axes**:

| Axis | Description |
|------|-------------|
| **Completion** | Does the agent produce a correct final answer for the given task? |
| **Cost** | How many tokens and dollars does the agent consume to reach its answer? |
| **Efficiency** | Does the agent reach the answer in a reasonable number of steps without redundancy? |
| **Reliability** | Does the agent produce consistent results across repeated runs? |
| **Recovery** | Can the agent still succeed when tools fail, return errors, or time out? |
| **Safety** | Does the agent resist adversarial prompt injections embedded in tool outputs? |

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Configure API Keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### Run

```bash
# Run the full benchmark suite
agentops-bench run --tasks tasks/ --agents claude,gpt4o --conditions clean,noisy,adversarial --runs 8

# Generate a summary report
agentops-bench report --results results/

# Validate task definitions
agentops-bench validate --tasks tasks/
```

## Architecture

```
                         +------------------+
                         |   CLI (click)    |
                         +--------+---------+
                                  |
                         +--------v---------+
                         | BenchmarkRunner   |
                         +--------+---------+
                                  |
              +-------------------+-------------------+
              |                                       |
    +---------v----------+               +------------v-----------+
    |   AgentAdapter     |               | InstrumentedToolServer |
    | (Anthropic/OpenAI) |<------------->|  (failure injection,   |
    +--------------------+  tool calls   |   prompt injection)    |
              |                          +------------------------+
              |
    +---------v----------+
    |   Scoring Suite    |
    | completion | cost  |
    | efficiency | rel.  |
    | recovery   | safety|
    +--------------------+
              |
    +---------v----------+
    |  BenchmarkReport   |
    |  (JSON + tables)   |
    +--------------------+
```

## Task Format

Tasks are defined as YAML files. See `tasks/README.md` for the full specification.

## Citation

```bibtex
@inproceedings{agentopsbench2026,
  title={AgentOps-Bench: A Benchmark for Operational Reliability of LLM Agent Systems},
  author={AgentOps-Bench Contributors},
  year={2026},
  note={Preprint}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
