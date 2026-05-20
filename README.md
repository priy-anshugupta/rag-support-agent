# 🧑‍💻 Multi-Ecosystem AI Support Agent

> [!NOTE]  
> A terminal-based AI agent that triages real support tickets across three product ecosystems—**HackerRank**, **Claude**, and **Visa**—using only a local support corpus.

This project uses Retrieval-Augmented Generation (RAG) and intelligent routing to act as a level 1 support agent, determining the correct product area, formulating grounded responses, and escalating issues when necessary.

---

## 📑 Contents

- [Repository layout](#-repository-layout)
- [Agent Capabilities](#-agent-capabilities)
- [Project Configuration](#-project-configuration)
- [Quickstart](#-quickstart)

---

## 📁 Repository layout

```text
.
├── README.md                       # You are here
├── code/                           # ← Agent implementation
│   └── main.py                     #   Entry point
├── data/                           # Local-only support corpus (no network needed)
│   ├── hackerrank/                 #   HackerRank help center
│   ├── claude/                     #   Claude Help Center export
│   └── visa/                       #   Visa consumer + small-business support
└── support_tickets/
    ├── sample_support_tickets.csv  # Inputs + expected outputs (for dev/evaluation)
    ├── support_tickets.csv         # Inputs only (run the agent on these)
    └── output.csv                  # Agent's generated predictions
```

---

## 🛠️ Agent Capabilities

The agent processes incoming rows in `support_tickets/support_tickets.csv` and outputs structured predictions containing:

| Column | Description |
| :--- | :--- |
| `status` | `replied` or `escalated` |
| `product_area` | The most relevant support category / domain area |
| `response` | User-facing answer grounded seamlessly in the provided corpus |
| `justification`| Concise explanation of the routing/answering decision |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

### Core Requirements

- **Local Execution**: Runs via the terminal.
- **Strict Grounding**: Uses **only the provided support corpus** (no live web calls for ground-truth answers).
- **Smart Escalation**: **Escalates** high-risk, sensitive, or unsupported cases instead of guessing.
- **Anti-Hallucination**: Avoids hallucinated policies or unsupported claims.

> [!TIP]  
> The agent is designed to work agnostically with various approaches — RAG, vector DBs, tool use, structured output, agent frameworks, or classical ML.

---

## 💻 Project Configuration

All agent modules and logic are located in [`code/`](./code/).

### 📝 Conventions

- Check the **README inside `code/`** for instructions on installing dependencies and running the agent.
- Read secrets **from environment variables only** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …). Rename `.env.example` → `.env` to configure your keys. **Never hardcode keys.**
- The application executes deterministically where possible.
- Predictions and responses are written to `support_tickets/output.csv`.

---

## 🚀 Quickstart

Clone this repository and navigate to the directory:

```bash
git clone <your-repository-url>
cd support-agent
```

_You are free to use any language or runtime. Recommended environments are **Python**, **JavaScript**, or **TypeScript**._