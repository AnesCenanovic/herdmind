# herdmind
Emergent behaviour in ecosystem simulation via a distilled decision model (MPVI semester project)

## Overview

**herdmind** is an ecosystem simulation platform designed to study how complex, collective behaviors (emergent behavior) can arise from simple, individualized artificial agents. The project focuses on bridging the gap between high-fidelity multi-agent systems and computational efficiency through policy distillation.

### Core Concepts

*   **Emergent Behavior:** Collective phenomena such as flocking, herding, and resource-driven migration emerge naturally from local interactions without explicit global coordination.
*   **Ecosystem Simulation:** A dynamic environment where agents interact with terrain, obstacles, and fluctuating resources (e.g., food, water) to optimize their survival.
*   **Distilled Decision Model:** Instead of running heavy, computationally expensive reinforcement learning (RL) deep networks or complex cognitive architectures for every agent in real-time, we train a lightweight "distilled" model. This student model mimics the complex decision-making policy while using a fraction of the memory and processing power, allowing the simulation to scale to thousands of concurrent agents.

### Project Poster

Our academic project poster is available here:

[View Project Poster](deliverables/poster.pdf)


## Development

```
uv sync            # install
uv run pytest -m "not slow" -q
uv run ruff check . && uv run ruff format --check .
```

Work on a branch and open a PR; `main` is protected and PRs are squash-merged. See `.claude/CLAUDE.md` for the project rules.
