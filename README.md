# Agentic Personal Assistant (LangGraph Supervisor)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A production-ready implementation of a **Hierarchical Multi-Agent System**. This project uses a Supervisor Agent to orchestrate specialized Sub-Agents (Calendar, Mail, and SQL) to perform complex, multi-step tasks via natural language.

---

## 🚀 Key Features

*   **Hierarchical Orchestration:** Uses a **Supervisor Node** to route tasks to specific workers (`CalendarAgent`, `MailAgent`, `SQLAgent`) based on intent.
*   **Natural Language to SQL:** The SQL Agent safely inspects database schemas and constructs syntactically correct queries to retrieve dynamic data (Team rosters, Availability).
*   **Transparent Reasoning:** A custom recursive UI renderer visualizes the entire "Chain of Thought" (CoT) and tool execution tree in real-time.
*   **Production Observability:**
    *   **Custom Callbacks:** Implemented `BaseCallbackHandler` to track token usage and cost *per specific agent* (not just global).
    *   **Distributed Logging:** Integrated with **Redis** and **BetterStack** for external log monitoring.
*   **Persistent State:** Supports conversation checkpoints (SQLite/Memory) allowing for context-aware multi-turn conversations.

## 🏗️ Architecture

The system follows the **Supervisor-Worker** pattern:

1.  **User Request:** "Schedule a meeting with the Design Team on Monday."
2.  **Supervisor:** Analyzes request -> Decomposes task.
    *   *Step 1:* Asks **SQL Agent** for "Design Team" members and emails.
    *   *Step 2:* Asks **Calendar Agent** to check availability for those emails.
    *   *Step 3:* Instructs **Calendar Agent** to book the slot.
    *   *Step 4:* Instructs **Mail Agent** to send notifications.
3.  **Response:** Final confirmation summary.

## 🛠️ Tech Stack

*   **LLM:** Google Gemini 1.5 Flash (via `langchain-google-genai`)
*   **Orchestration:** LangGraph (StateGraph, ReAct Agents)
*   **Database:** SQLite (Relational data for Staff/Availability)
*   **Cache/Queues:** Redis/Valkey
*   **Interface:** Streamlit (with Custom Component rendering)
*   **DevOps:** Hosted on Render, Environment management via `.env`

## 🧪 Usage & Testing

### Installation
```bash
git clone https://github.com/danielecelsa/mail-calendar-assistant.git
cd mail-calendar-assistant
pip install -r requirements.txt
```

### Environment Setup
Create a .env file:
```bash
GOOGLE_API_KEY=your_key_here
REDIS_URL=your_redis_url (optional)
```

### Running the App
```bash
streamlit run app.py
```

## Interesting Prompts to Try

### Shared Availability (Complex Logic):
*"Find a time when both Marco Rossi and the Developer team are available on Friday."*
(Requires SQL Agent to resolve 'Developer team' to names, then check intersection of availability)
### Full Workflow:
*"Schedule a meeting with Anna Garau next Monday at 10am and send her an email confirmation."*
(Triggers Supervisor -> Calendar -> Mail)

## 🧠 Engineering Highlights
*   **Custom Token Tracking:** Unlike standard LangChain callbacks, MultiAgentUsageHandler (in helpers.py) maps execution runs to specific Agent IDs, allowing for granular cost analysis in a multi-agent environment.
*   **Recursive UI Rendering:** render_workflow_node recursively unpacks the LangGraph execution trace to show the user exactly which tool was called and why.

---

## 👨‍💻 Author
Daniele Celsa

*   [Portfolio Website](https://danielecelsa.com)
*   [LinkedIn](https://diretta.it)