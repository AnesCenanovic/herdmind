# herdmind

Emergent behaviour in ecosystem simulation via AI-driven decision models (MPVI semester project)

## Overview

**herdmind** is an ecosystem simulation platform designed to study how complex, collective behaviors (emergent behavior) can arise from simple, individualized artificial agents. The project focuses on bridging the gap between high-fidelity multi-agent systems and real-time simulation performance.

### Core Concepts

*   **Emergent Behavior:** Collective phenomena such as flocking, herding, and resource-driven migration emerge naturally from local interactions without explicit global coordination.
*   **Ecosystem Simulation:** A dynamic environment where agents interact with terrain, obstacles, and fluctuating resources (e.g., food, water) to optimize their survival.
*   **Agent Decision-Making Architecture (Under Evaluation):** We are currently exploring two potential approaches for powering agent intelligence in real-time:
    1.  **Model Distillation:** Training a lightweight "student" model to mimic complex decision-making policies. This uses a fraction of the memory and processing power, allowing the simulation to easily scale to thousands of concurrent agents locally.
    2.  **Real-Time API Inference:** Directly querying an external AI model via API calls during the simulation to drive agent behavior, leveraging pre-trained intelligence without the need for local model training.

## Development

```
uv sync            # install
uv run pytest -m "not slow" -q
uv run ruff check . && uv run ruff format --check .
```

Work on a branch and open a PR; `main` is protected and PRs are squash-merged. See `.claude/CLAUDE.md` for the project rules.
