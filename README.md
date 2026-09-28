# Researcher Crew Agent

> A multi-agent AI research system built with **CrewAI**, **LiteLLM**, and **Groq** to automate research and transform findings into structured reports.

---

## Overview

**Researcher Crew Agent** is a multi-agent system designed to automate the research-to-report workflow.

Instead of relying on a single LLM call, the system separates the workflow into specialized agents with different responsibilities:

1. **Researcher Agent** — investigates the requested topic and gathers relevant findings.
2. **Reporting Analyst** — analyzes the research output and transforms it into a structured report.

The agents operate through a **sequential CrewAI workflow**, where the output of the Researcher becomes the context for the Reporting Analyst.

```text
                         User Input
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Researcher Agent  │
                  │                     │
                  │ Research & Analysis │
                  └──────────┬──────────┘
                             │
                             ▼
                    Research Findings
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Reporting Analyst   │
                  │                     │
                  │ Analyze & Structure │
                  └──────────┬──────────┘
                             │
                             ▼
                       Final Report
                         report.md
```

---

## Key Features

* Multi-agent architecture using **CrewAI**
* Specialized agents with separate roles and goals
* Sequential agent orchestration
* LLM integration through **LiteLLM**
* Groq API integration
* **GPT-OSS 120B** model support
* YAML-based agent and task configuration
* Markdown report generation
* CrewAI training support
* Task replay support
* Trigger-based execution
* `uv` dependency and environment management
* Intel macOS-compatible ONNX Runtime configuration

---

## Tech Stack

| Technology       | Purpose                             |
| ---------------- | ----------------------------------- |
| **Python**       | Core programming language           |
| **CrewAI**       | Multi-agent orchestration           |
| **LiteLLM**      | LLM provider abstraction            |
| **Groq**         | LLM inference provider              |
| **GPT-OSS 120B** | Language model                      |
| **Pydantic**     | Data validation                     |
| **uv**           | Dependency & environment management |
| **YAML**         | Agent & task configuration          |
| **Markdown**     | Report output                       |

---

## Agent Architecture

### 1. Researcher Agent

The Researcher is responsible for investigating the requested topic.

**Responsibilities:**

* Analyze the research topic
* Identify relevant information
* Explore recent developments
* Produce research findings for downstream processing

Example configuration:

```yaml
researcher:
  role: >
    {topic} Senior Data Researcher

  goal: >
    Uncover cutting-edge developments in {topic}

  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.

  llm: groq/openai/gpt-oss-120b
```

### 2. Reporting Analyst

The Reporting Analyst receives the Researcher's output and converts it into a structured report.

**Responsibilities:**

* Analyze research findings
* Organize the information
* Produce a clear report
* Save the final output as Markdown

```yaml
reporting_analyst:
  role: >
    {topic} Reporting Analyst

  goal: >
    Create detailed reports based on {topic} data analysis and research findings

  llm: groq/openai/gpt-oss-120b
```

---

## Workflow

The crew uses a **sequential process**:

```python
Crew(
    agents=self.agents,
    tasks=self.tasks,
    process=Process.sequential,
    verbose=True,
)
```

Execution flow:

```text
1. User provides a topic
          ↓
2. Researcher Agent starts
          ↓
3. Research findings are generated
          ↓
4. Findings are passed to Reporting Analyst
          ↓
5. Reporting Analyst generates final report
          ↓
6. report.md is created
```

This architecture keeps the responsibilities of research and reporting separate, making the workflow easier to extend with additional agents or tasks.

---

## Project Structure

```text
Researcher_crew_agent/
│
└── researcher/
    │
    ├── src/
    │   └── researcher/
    │       │
    │       ├── config/
    │       │   ├── agents.yaml
    │       │   └── tasks.yaml
    │       │
    │       ├── crew.py
    │       └── main.py
    │
    ├── tests/
    │
    ├── .env
    ├── .gitignore
    ├── pyproject.toml
    ├── uv.lock
    ├── README.md
    └── report.md
```

---

## Installation

### Prerequisites

* Python `>=3.10,<3.14`
* `uv`
* Groq API key

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Researcher_crew_agent/researcher
```

Install dependencies:

```bash
uv sync
```

---

## Environment Configuration

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Make sure `.env` is excluded from Git:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

**Never commit your API key to the repository.**

---

## Dependencies

The project intentionally keeps its direct dependency set small:

```toml
dependencies = [
    "crewai==1.9.3",
    "litellm==1.75.3",
    "onnxruntime==1.23.2",
    "tiktoken==0.14.0",
]
```

### Why is ONNX Runtime pinned?

`onnxruntime==1.23.2` is pinned to maintain compatibility with **Intel-based macOS** environments.

This avoids dependency resolution issues caused by newer ONNX Runtime releases that may not provide compatible Intel macOS wheels.

---

## LLM Configuration

The project uses Groq through LiteLLM.

```yaml
llm: groq/openai/gpt-oss-120b
```

The structure is:

```text
groq/
   │
   └── openai/gpt-oss-120b
       │
       └── Groq model ID
```

This allows the CrewAI agents to communicate with Groq without directly coupling the application architecture to a single LLM SDK.

---

## Running the Project

Run the crew:

```bash
uv run crewai run
```

Or:

```bash
uv run run_crew
```

The default research topic is:

```python
inputs = {
    "topic": "Large Language Models",
    "current_year": str(datetime.now().year)
}
```

---

## Output

After execution, the Reporting Analyst writes the final report to:

```text
report.md
```

Example:

```text
Researcher Agent
       │
       ▼
Research Findings
       │
       ▼
Reporting Analyst
       │
       ▼
┌──────────────────┐
│     report.md    │
└──────────────────┘
```

---

## CrewAI Capabilities

The project also exposes several CrewAI capabilities.

### Training

```bash
uv run train <iterations> <filename>
```

Example:

```bash
uv run train 5 training_data.pkl
```

### Replay

Replay a previously executed task:

```bash
uv run replay <task_id>
```

### Testing

```bash
uv run test <iterations> <eval_llm>
```

### Trigger Execution

Execute the crew using a JSON payload:

```bash
uv run run_with_trigger '{"topic":"Generative AI"}'
```

---

## Design Decisions

### Why Multi-Agent?

Research and reporting are different responsibilities.

Separating them allows each agent to focus on a specific task instead of relying on one large prompt to perform the complete workflow.

### Why LiteLLM?

LiteLLM provides a unified interface for working with different LLM providers.

This keeps the agent configuration flexible and makes it easier to change providers or models without redesigning the overall CrewAI architecture.

### Why Sequential Processing?

The reporting stage depends on the research stage.

Therefore:

```text
Research → Analysis → Report
```

is a natural fit for a sequential workflow.

---

## Current Scope

The current version focuses on the core multi-agent workflow:

* Agent definition
* Task definition
* Sequential orchestration
* LLM integration
* Research generation
* Report generation
* Training and replay interfaces

### Future Improvements

Potential extensions include:

* Web search tools for real-time research
* Source citation and verification
* Research quality evaluation
* Additional specialized research agents
* Parallel research tasks
* Human-in-the-loop review
* Structured JSON outputs
* Observability and tracing
* RAG integration
* Automated report evaluation

---

## Security

API credentials should always be stored in environment variables.

```env
GROQ_API_KEY=your_groq_api_key
```

Never hard-code credentials inside:

* Python source files
* YAML configuration
* Git commits
* README files

---

## Learning Objectives

This project was built to explore practical concepts in **Agentic AI**, including:

* Multi-agent systems
* Agent roles and responsibilities
* Task orchestration
* Sequential workflows
* LLM provider abstraction
* Prompt-driven agent behavior
* Agent training and replay
* AI workflow design

---

## Author

**Ahmed Taha**

AI Engineer | LLM Engineer

Focused on building practical AI systems with:

```text
LLMs • RAG • AI Agents • LangGraph • FastAPI • Python
```

---

## License

This project is intended for educational and experimental purposes.
