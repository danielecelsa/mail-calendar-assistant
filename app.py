# Agentic Personal Assistant with Calendar and Email Management
# Uses LangGraph ReAct framework with Google Gemini LLM.
# Capable of scheduling events and sending emails via natural language.

print("=== PRINT FROM PROCESS START ===")

# ------------------------------
# Imports
# ------------------------------
import os
import threading
from pathlib import Path
import uuid
import queue
import re
import datetime
from datetime import timezone
import json
import time
import valkey
from time import perf_counter
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from logging_config import (
    get_logger,
    get_redis_client
)

from helpers import (
    compute_cost,
    get_user_info,
    MultiAgentUsageHandler,
    MultiAgentWorkflowHandler,
)

from async_bg import collect_events_from_agent

from prompts import (
    CALENDAR_AGENT_PROMPT,
    MAIL_AGENT_PROMPT,
    SQL_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
)

# LangGraph / LangChain Core
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.callbacks import Callbacks

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Load environment variables from .env file if not in a rendering environment
if os.getenv("RENDER") != "true":
    load_dotenv()


# ------------------------------
# Configuration
# ------------------------------
MODEL = os.environ.get("GENAI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_API_KEY_2 = os.environ.get("GOOGLE_API_KEY_2") # paid tier key (if any)

COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_INPUT", "0.002"))
COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_OUTPUT", "0.002"))

DB_PATH = Path("agent_test.db")
DB_URI = f"sqlite:///{DB_PATH}"

AGENT_CONFIG = {
    "supervisor": {"temp": 0.2, "tier": "free"},
    "calendar":   {"temp": 0.0, "tier": "paid"},
    "sql":        {"temp": 0.0, "tier": "paid"}, 
    "mail":       {"temp": 0.2, "tier": "free"},
}

# ------------------------------
# LOGGING SETUP
# ------------------------------
logger_local = get_logger("local")
logger_betterstack = get_logger("betterstack")
logger_redis = get_logger("redis")
logger_all = get_logger("all") 

@st.cache_resource
def get_kv_client():
    kv_client = get_redis_client() 
    return kv_client

kv_client = get_kv_client() # Valkey client (Redis-compatible) for temporary queues

# --------------------------------------
# Streamlit session state initialization
# --------------------------------------
if "conversation_thread_id" not in st.session_state:
    st.session_state.conversation_thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am your personal assistant: I can schedule your Meetings and send Emails!  \nHow can I help you today?")]
if "latency" not in st.session_state:
    st.session_state.latency = 0.0
if "to_cut" not in st.session_state:
    st.session_state.to_cut = 0

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_input_tokens" not in st.session_state: # to track total input tokens cost
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state: # to track total output tokens cost
    st.session_state.total_output_tokens = 0

st.session_state.Supervisor_agent_history = []
st.session_state.SQL_agent_history = []
st.session_state.Calendar_agent_history = []
st.session_state.Mail_agent_history = []


st.session_state.Supervisor_last_tokens = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
}
st.session_state.SQL_last_tokens = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
}
st.session_state.Mail_last_tokens = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
}
st.session_state.Calendar_last_tokens = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
}
st.session_state.input_tokens_last = 0
st.session_state.output_tokens_last = 0
st.session_state.total_tokens_last = 0

if "usd" not in st.session_state:
    st.session_state.usd = 0.0
if "usd_last" not in st.session_state:
    st.session_state.usd_last = 0.0

if "workflow_tree" not in st.session_state:
    st.session_state.workflow_tree = []

if 'tour_completed' not in st.session_state:
    st.session_state['tour_completed'] = False

# ------------------------------
# Helpers
# ------------------------------

def get_api_key_by_tier(tier: str):
    if tier == "paid" and os.environ.get("GOOGLE_API_KEY_2"):
        return os.environ.get("GOOGLE_API_KEY_2")
    return os.environ.get("GOOGLE_API_KEY")

def update_workflow(multi_agent_workflow_handler: MultiAgentWorkflowHandler):
    """Save the execution tree in session_state."""
    tree = multi_agent_workflow_handler.get_workflow_tree()
    logger_local.info(f"FINAL WORKFLOW TREE: {tree}")
    st.session_state.workflow_tree = tree # New state variable

def update_usage_token(multi_agent_handler: MultiAgentUsageHandler):
    """Update the token usage in st.session_state based on MultiAgentUsageHandler data."""
    
    # Extract aggregated data at the end of everything
    final_usage_data = multi_agent_handler.get_final_usage()
    logger_local.info(f"FINAL COLLECTED DATA: {final_usage_data}")

    # Populate Streamlit state with usage data
    # Reset "last" counters
    st.session_state.input_tokens_last = 0
    st.session_state.output_tokens_last = 0
    st.session_state.total_tokens_last = 0
    
    # Update values for each agent
    for agent_name, usage in final_usage_data.items():
        key_tokens = f"{agent_name}_last_tokens"
        st.session_state[key_tokens] = usage # E.g., st.session_state['Supervisor_last_tokens'] = usage
        
        # Update "last" and "session" totals
        st.session_state.input_tokens_last += usage.get("input_tokens", 0)
        st.session_state.output_tokens_last += usage.get("output_tokens", 0)
        st.session_state.total_tokens_last += usage.get("total_tokens", 0)
        
        st.session_state.total_input_tokens += usage.get("input_tokens", 0)
        st.session_state.total_output_tokens += usage.get("output_tokens", 0)
        st.session_state.total_tokens += usage.get("total_tokens", 0)

def render_workflow_node(node, level=0, parent_name=""):
    """
    Recursively renders the execution tree.
    Applies an 'Unwrapping' logic to remove empty container nodes
    that LangChain generates excessively.
    """
    
    node_name = node['name']
    node_type = node['type']
    children = node['children']

    DISPLAY_NAMES = {
        "SQL": "SQL Agent",
        "Calendar": "Calendar Agent",
        "Mail": "Mail Agent",
        "Supervisor": "Supervisor Agent",
        "check_staff_info": "check_staff_info (SQL Agent)",
        "manage_mail": "manage_mail (Mail Agent)",
        "schedule_event": "schedule_event (Calendar Agent)",

    }
    pretty_name = DISPLAY_NAMES.get(node_name, node_name)
    
    # 1. CLEANING LOGS
    # Remove duplicate logs or redundant "Request" entries that appear in every child node
    clean_logs = []
    for log in node['logs']:
        # Ignore Requests if we are not at the root level (avoids infinite repetitions)
        if "**Request**" in log and level > 0:
            continue
        clean_logs.append(log)
    
    # 2. CONTENT ANALYSIS
    has_logs = len(clean_logs) > 0
    has_children = len(children) > 0
    
    # Debug print to understand what's happening in the terminal
    logger_local.info(f"[UI DEBUG] Analyzing Node: '{node_name}' (Type: {node_type}, Level: {level})")
    logger_local.info(f"           -> Has Logs: {has_logs} ({len(clean_logs)} lines)")
    logger_local.info(f"           -> Has Children: {has_children} ({len(children)} children)")

    # 3. UNWRAPPING LOGIC (The heart of the solution)
    # If an Agent node has NO logs of its own AND has children, 
    # instead of drawing an empty Expander, we pass directly to the children.
    # This removes the chain Supervisor -> Supervisor -> Supervisor
    should_unwrap = (node_type == 'agent' and not has_logs and has_children)
    
    # Exception: If the child is a Tool or a different Agent, we might want to keep 
    # this container for context, BUT if it's empty, it's better to remove it.
    # If the node is empty and has no children, we don't show it at all.
    if not has_logs and not has_children:
        logger_local.info(f"           -> ACTION: SKIPPING (Empty Node)")
        return

    if should_unwrap:
        logger_local.info(f"           -> ACTION: UNWRAPPING (Pass-through to children)")
        # We don't draw an expander here, we recursively call the children
        # keeping the same visual 'level'
        for child in children:
            render_workflow_node(child, level, parent_name=node_name)
        return

    # 4. ACTUAL RENDERING
    logger_local.info(f"           -> ACTION: RENDERING EXPANDER")
    
    # Choose the icon
    icon = "🛠️" if node_type == 'tool' else "🤖"
    
    # Build the label. 
    # If we come from an unwrap (parent_name == node_name), it's a continuation.
    label = f"{icon} {pretty_name}"
    
    # Different color or style for Tools to distinguish them
    if node_type == 'tool':
        label = f"🛠️ Call Tool: {pretty_name}"

    # Default expanded: True only for level 0 (Root)
    is_expanded = (level == 0)

    with st.expander(label, expanded=is_expanded):
        # Print the clean logs
        for log in clean_logs:
            st.markdown(log)
            # Add a divider only if it's not the last log
            if log != clean_logs[-1]:
                st.markdown("---")
        
        # If there are logs and also children, add a visual separator
        if clean_logs and children:
            st.markdown("---")

        # Render the children (increasing the level for visual indentation if necessary,
        # but st.expander automatically handles nesting)
        for child in children:
            render_workflow_node(child, level + 1, parent_name=node_name)

@st.cache_resource
def get_db():
    """Get the SQLDatabase instance connected to the Chinook sample database."""

    db = SQLDatabase.from_uri(DB_URI)

    logger_local.info("Connected to database at %s", DB_URI)
    logger_local.info(f"Dialect: {db.dialect}")
    logger_local.info(f"Available tables: {db.get_usable_table_names()}")
    logger_local.info(f'Sample output: {db.run("SELECT * FROM staff LIMIT 5;")}')

    return db

# ------------------------------
# LLM
# ------------------------------
@st.cache_resource
def get_llm(agent_name: str):
    """Factory to get the right LLM based on the agent's profile."""
    
    config = AGENT_CONFIG.get(agent_name, {"temp": 0.1, "tier": "free"})
    
    api_key = get_api_key_by_tier(config["tier"])
    temp = config["temp"]

    print("Creating LLM with temperature:", temp)
    print("Using API Key:", "PAID TIER KEY" if config["tier"] == "paid" else "FREE TIER KEY")
    print("Agent Name:", agent_name)

    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            google_api_key=api_key,
            temperature=temp,
            safety_settings=None,
            transport="rest" # if not working, just use python 3.11 (not 3.13)
            )
    except Exception as e:
        logger_all.exception("Could not initialize LLM: %s", e)
        llm = None

    return llm  


# ------------------------------
# Tools - for Sub-Agents
# ------------------------------
@tool
def create_calendar_event(
    title: str,
    date: str,         # ISO format: "2024-01-15"
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    attendees: list[str],  # email addresses
    ) -> str:
    """Create a calendar event. Requires exact ISO datetime format, start time, title and attendees."""
    
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.

    if title and date and start_time and attendees:
        logger_all.info("create_calendar_event Created event: %s on %s at %s with attendees %s", title, date, start_time, attendees)
        pass  # Assume event created successfully
    else:
        logger_all.error("create_calendar_event Failed to create event: missing information.")
        raise ValueError("Missing required event information.")
    
    return f"Event created: {title} on {date} from {start_time} with {len(attendees)} attendees"

@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    ) -> str:
    """Send an email via email API. Requires properly formatted email addresses (like a@b), subject and body."""
    
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    
    if to and subject and body:
        pass  # Assume email sent successfully
    else:
        logger_all.error("send_email Failed to send email: missing information.")
        raise ValueError("Missing required email information.")  
    
    logger_all.info("send_email Sent email to: %s with subject: %s", to, subject)

    return f"Email sent to {', '.join(to)} - Subject: {subject} - Body length: {len(body)} characters"

# ------------------------------
# SQL Database Tools - we don't need @tool here cause SQLDatabaseToolkit.get_tools() does it
# ------------------------------
@st.cache_resource
def get_sql_tools():
    """Get the list of tools for the SQL agent."""
    toolkit = SQLDatabaseToolkit(db=get_db(), llm=get_llm("sql"))

    tools = toolkit.get_tools()

    for tool in tools:
        logger_local.info(f"SQL Tool loaded: {tool.name} - {tool.description}")

    return tools

# ------------------------------
# Build SQL Sub-Agent
# ------------------------------
@st.cache_resource
def get_sql_agent():
    """Get the SQL agent instance."""

    formatted_sql_prompt = SQL_AGENT_PROMPT.format(
        dialect=get_db().dialect,
        top_k=5,
    )
    
    sql_agent = create_react_agent(
        model=get_llm("sql"),
        tools=get_sql_tools(),
        prompt=SystemMessage(content=formatted_sql_prompt)
    )

    return sql_agent


# ------------------------------
# SQL Sub-agent as Tool
# ------------------------------
@tool
def check_staff_info(request: str, callbacks: Callbacks = None) -> str:
    """
    Check info about the staff.

    Use this to check staff emails, availability, team membership, etc.

    The request should specify what info is needed about which staff member(s) or team.
    If the request is ambiguous (e.g., only first name provided), the agent will ask for clarification.

    The database contains tables like 'staff' (with columns like name, surname, full_name_norm, email, team) and 'availability' (with columns like day of the week, time, and one column per staff member indicating availability - True/False - at that time/day).

    Input: Natural language request about staff info (e.g., 'Is Marco available
    Monday at 09:00?',  'What are the emails of the developer team members?')
    """

    sub_agent_config = {
        "callbacks": callbacks,
        "metadata": {"agent_name": "SQL"} # Tag for the handler
    }

    result = get_sql_agent().invoke({
        "messages": [{"role": "user", "content": request}]
        },
        config=sub_agent_config
    )


    logger_local.info("SQL_AGENT RESULT: %s", result)

    return result["messages"][-1].text


# ------------------------------
# Build Calendar/Mail Sub-Agents
# ------------------------------
@st.cache_resource
def get_calendar_agent():
    """Get the Calendar agent instance."""

    calendar_agent = create_react_agent(
        model=get_llm("calendar"),
        tools=[create_calendar_event, check_staff_info],
        prompt=SystemMessage(content=CALENDAR_AGENT_PROMPT)
        )
    
    return calendar_agent


@st.cache_resource
def get_mail_agent():
    """Get the Mail agent instance."""

    mail_agent = create_react_agent(
        model=get_llm("mail"),
        tools=[send_email, check_staff_info],
        prompt=SystemMessage(content=MAIL_AGENT_PROMPT)
        )
    
    return mail_agent


# ------------------------------
# Calendar/Mail Sub-agents as Tools
# ------------------------------
@tool
def schedule_event(request: str, callbacks: Callbacks = None) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create calendar events.

    Provide as much details as possible in the request. 
    For example, for scheduling events include title, date, start time, and attendees, if possible.
    Some information may be missing, in which case the agent will ask for clarification or create generic titles if needed.
    But some others are essential to proceed, like date/start time and attendees.

    Input: Natural language request about scheduling calendar events (e.g., 'Schedule a meeting with Marco next Tuesday at 2pm').
    """

    sub_agent_config = {
        "callbacks": callbacks,
        "metadata": {"agent_name": "Calendar"} # Tag for the handler
    }

    result = get_calendar_agent().invoke({
        "messages": [{"role": "user", "content": request}]
        },
        config=sub_agent_config
    )

    logger_local.info("CALENDAR_AGENT RESULT: %s", result)
    
    return result["messages"][-1].text


@tool
def manage_mail(request: str, callbacks: Callbacks = None) -> str:
    """Send emails using natural language.

    Use this when the user wants to send emails.
    Emails can be about any topic, including notifications about scheduled events.

    Provide as much details as possible in the request. 
    For example, include recipients, subject, and body content, if possible.
    Some information may be missing, in which case the agent will ask for clarification or create generic subject or body if context is clear enough.
    But some others are essential to proceed, like recipients and content.

    Input: Natural language request about sending email (e.g., 'Send an email to the design team about the project update').
    """

    sub_agent_config = {
        "callbacks": callbacks,
        "metadata": {"agent_name": "Mail"} # Tag for the handler
    }

    result = get_mail_agent().invoke({
        "messages": [{"role": "user", "content": request}]
        },
        config=sub_agent_config
    )

    logger_local.info("MAIL_AGENT RESULT: %s", result)
    
    return result["messages"][-1].text


# ------------------------------
# Supervisor Prompt
# ------------------------------
@st.cache_resource
def get_supervisor_prompt():
    """Get the prompt template for the supervisor agent."""
    
    system = SystemMessagePromptTemplate.from_template(SUPERVISOR_PROMPT)

    hist = MessagesPlaceholder(variable_name="messages")
    prompt = ChatPromptTemplate.from_messages([system, hist])
    
    return prompt


# ------------------------------
# Supervisor Checkpointer
# ------------------------------
@st.cache_resource
def get_checkpointer():
    """Get a checkpointer for the supervisor agent."""

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver as SqliteCheckpointer
        checkpointer = SqliteCheckpointer.from_conn_string(os.environ.get("CHECKPOINT_DB", "./langgraph_state.sqlite"))
    except Exception:
        try:
            from langgraph.checkpoint.memory import InMemorySaver as InMemoryCheckpointer
            checkpointer = InMemoryCheckpointer()
        except Exception:
            checkpointer = None
    print("Using checkpointer:", type(checkpointer))
    print(checkpointer)
    return checkpointer


# ------------------------------
# Build Supervisor
# ------------------------------
def build_supervisor():
    """Build the supervisor agent coordinating calendar and email sub-agents."""

    supervisor = create_react_agent(
        model=get_llm("supervisor"),
        tools=[schedule_event, manage_mail, check_staff_info],
        prompt=get_supervisor_prompt(),
        checkpointer=get_checkpointer(),
    )

    return supervisor


supervisor = build_supervisor()


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Enterprise Multi-Agent Orchestrator", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise Multi-Agent Orchestrator", anchor=False)
body="""
**Scheduling & Email Assistant - powered by LangGraph**

This application demonstrates a **Hierarchical Multi-Agent System** capable of handling complex execution workflows. 
Unlike standard chatbots, this system uses a **Supervisor-Workers** architecture to decompose vague user requests into precise, executable actions
performed by specialized **Sub-Agents** (Calendar, Mail, SQL-RAG)

**Key Features:**
*   **🧠 Hierarchical Planning:** Uses a **Supervisor-Worker** pattern to prevent context pollution between agents.
*   **🔄 Self-Correcting Logic:** Agents utilize the **ReAct pattern** (Reason + Act) to evaluate tool outputs and retry steps if parameters are missing or incorrect.
*   **🛡️ Deterministic Data Retrieval:** A specialized SQL Agent performs **dynamic schema introspection** to execute safe, read-only queries. This ensures 100% syntactically correct SQL generation and eliminates hallucinations during entity resolution.
*   **📊 Production Observability:** Includes custom-built middleware for real-time **Token Tracking**, **Cost Estimation**, and **Decision Tree Tracing**.
"""
with st.expander('About this demo (Read me)', expanded=False):
    st.markdown(body)

# ------------------------------
# Chat submission
# ------------------------------
user_query = st.chat_input("Type your message here...")
if user_query:

    # Get user info as soon as they submit a query
    user_details = get_user_info(logger_all)
    logger_all.info(
        f"New query received from IP: {user_details['ip']} "
        f"with User-Agent: {user_details['user_agent']}"
    )

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    multi_agent_handler = MultiAgentUsageHandler()
    multi_agent_workflow_handler = MultiAgentWorkflowHandler()

    thread_id = st.session_state.conversation_thread_id
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [multi_agent_handler, multi_agent_workflow_handler], # Pass our own handler
        "metadata": {"agent_name": "Supervisor"} # Label the supervisor as well
    }
    inputs = {"messages": st.session_state.chat_history}

    # using invoke to obtain final structured response (astream requires custom handling)
    start = perf_counter()
    with st.spinner("Thinking..."):
        events = collect_events_from_agent(supervisor, inputs, config=config)
 
    update_usage_token(multi_agent_handler)
    update_workflow(multi_agent_workflow_handler)

    st.session_state.latency = perf_counter() - start

    # Compute cost
    try:
        st.session_state.usd = compute_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger_all.exception("Error computing total cost: %s", e)
        st.session_state.usd = 0.0
    
    try:
        st.session_state.usd_last = compute_cost(st.session_state.input_tokens_last, st.session_state.output_tokens_last, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        logger_all.exception("Error computing last cost: %s", e)
        st.session_state.usd_last = 0.0

    logger_local.info("SUPERVISOR EVENT: %s", events)

    # 3. Extract final text from the EVENTS list
    # We loop through events to find the last message from the AI
    final_text = "I'm sorry, I couldn't generate a response."
    
    for event in reversed(events):
        # 'messages' key usually appears in updates from the agent node
        if "messages" in event:
            # It might be a list of messages or a single message depending on LangGraph version
            msgs = event["messages"]
            if isinstance(msgs, list):
                last_msg = msgs[-1]
            else:
                last_msg = msgs
            
            # Check if it is an AI Message with content
            if hasattr(last_msg, "content") and last_msg.content:
                final_text = last_msg.content
                break
        # Sometimes it appears under 'agent' key depending on graph structure
        elif "agent" in event and "messages" in event["agent"]:
             msgs = event["agent"]["messages"]
             last_msg = msgs[-1]
             if hasattr(last_msg, "content") and last_msg.content:
                final_text = last_msg.content
                break

    st.session_state.chat_history.append(AIMessage(content=final_text))

    # --- LOGGING ---
    logger_all.info("Latency for full response: %.2f seconds", st.session_state.latency)
    logger_all.info("Last interaction tokens: %d (input), %d (output), %d (total)", st.session_state.input_tokens_last, st.session_state.output_tokens_last, st.session_state.total_tokens_last)
    logger_all.info("Estimated last interaction cost: $%.5f", st.session_state.usd_last)
    logger_all.info("Total tokens so far: %d (input), %d (output), %d (total)", st.session_state.total_input_tokens, st.session_state.total_output_tokens, st.session_state.total_tokens)
    logger_all.info("Estimated total cost so far: $%.5f", st.session_state.usd)

    # Only for Supervisor
    if st.session_state.Supervisor_last_tokens['total_tokens']:
        logger_all.info("Supervisor last interaction tokens: %d (total)", 
            st.session_state.Supervisor_last_tokens['total_tokens']
        )
    if st.session_state.Mail_last_tokens['total_tokens']:
        logger_all.info("Mail Agent last interaction tokens: %d (total)", 
            st.session_state.Mail_last_tokens['total_tokens']
        )
    if st.session_state.Calendar_last_tokens['total_tokens']:
        logger_all.info("Calendar Agent last interaction tokens: %d (total)", 
            st.session_state.Calendar_last_tokens['total_tokens']
        )
    if st.session_state.SQL_last_tokens['total_tokens']:
        logger_all.info("SQL Agent last interaction tokens: %d (total)", 
            st.session_state.SQL_last_tokens['total_tokens']
        )

    data_dict={
        "ts": datetime.datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "messages": [{"role": m.__class__.__name__, "content": m.content} for m in st.session_state.chat_history],
    }
    logger_all.info(json.dumps(data_dict, indent=2, ensure_ascii=False))

    # TO TEST:
    logger_local.info(f"UW-TEST - SUPERVISOR HISTORY: {st.session_state.Supervisor_agent_history}")
    logger_local.info(f"UW-TEST - MAIL AGENT HISTORY: {st.session_state.Mail_agent_history}")
    logger_local.info(f"UW-TEST - CALENDAR AGENT HISTORY: {st.session_state.Calendar_agent_history}")
    logger_local.info(f"UW-TEST - SQL AGENT HISTORY: {st.session_state.SQL_agent_history}")
    logger_local.info(f"UW3-TEST - WORKFLOW TREE: {st.session_state.workflow_tree}")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("⚙️ System Architecture", divider="rainbow")
    st.info("This PoC demonstrates **Hierarchical Multi-Agent Orchestration** with **Granular Cost Tracking**.")

    st.markdown(" ")
    
    # ------------------------------
    # Info Section
    # ------------------------------
    with st.expander("🛠️ Architecture & Tech Stack"):
            st.markdown("""
            **Core Orchestration:**
            - `LangGraph`: Cyclic state management implementation (Supervisor-Worker pattern).
            - `LangChain`: Tool binding and ReAct loops.
            
            **Advanced Implementation:**
            - ***Custom Callback Handlers:*** Built from scratch to intercept real-time token usage and reconstruct the hierarchical execution tree for the UI.
            - ***NL-to-SQL Engine:*** A secure, read-only SQL agent that performs dynamic schema introspection to answer data-driven questions
            - ***Non-blocking Async Event Loop:*** Implements a background loop to handle agent streams concurrently, ensuring the Streamlit UI remains responsive during long chains of thought.
                        
            **Observability & FinOps:** 
            - ***Reasoning Trace:*** Real-time "Chain of Thought" visualization.
            - ***Live Metrics:*** Real-time Token Usage and USD Cost **per-agent**. Allows for analysis of **Unit Economics** per interaction.
            - ***Distributed Logging:*** Structured logs (Redis + BetterStack) for remote monitoring.
            """)
            with st.expander("Architecture Overview - :orange[show diagram]"):
                st.caption("This workflow allows the Supervisor to delegate tasks to Sub-Agents, which can act as specialized tools.")
                st.image("images/supervisor_subagents.png", caption="Supervisor + Sub-Agents workflow")
                with st.expander("💡 Technical Implementation Details"):
                    st.caption("""
                                *   **Sandboxed Execution Environment:** 
                                
                                    To ensure security and portability for this public demo, external API calls (Email/Calendar) are sandboxed. The system validates inputs and constructs the full payload, simulating execution without triggering real-world side effects.
                               
                                *   **SQL Schema Introspection:** 
                                    
                                    The `check_staff_info` tool is not a vector search. It is a **deterministic SQL Agent** leveraging the `SQLDatabaseToolkit`. It dynamically reads the SQLite schema to construct syntactically correct queries, ensuring 100% accurate data retrieval for rosters and availability.
                                
                                *   **Shared Dependency Pattern:** 
                                    
                                    Note that the SQL Agent acts as a **Shared Service**. Both the *Calendar Agent*, the *Mail Agent* and the *Supervisor* can independently query it to resolve team members or email addresses before proceeding with their tasks.
                               """)
            with st.expander("SQLite DB - :orange[check data]"):
                st.caption("The SQL Agent queries these tables to resolve staff entities.")
                st.image("images/db_team.png", caption="Team Membership & Emails")
                st.image("images/db_availability.png", caption="Weekly Staff Availability")
    
    with st.expander("🧪 How to Test (Scenarios)"):
        st.caption(("Try these inputs to see the specific architectural patterns in action:"))
        st.markdown("**1. RAG & Entity Resolution (SQL Agent)**:")
        st.write("##### *The agent must query the DB to retrieve data before answering*")
        st.markdown("""
                    > *Is Daniele Celsa available on Monday at 10?*

                    > *What are the emails of the developer team members?*

                    > *At what times is Anna Garau available all the week? I need her availability for all days*
                    """)
        st.markdown("**2. Cross-Agent Logic (SQL + Mail)**:")
        st.write("##### *Requires fetching data (SQL) and passing it to a different context (Mail)*")
        st.markdown("""
                    > *Send an email to developer team saying I won't be at office today.*
                    
                    > *Check if Marco Rossi is available on Friday at 10, if yes send him an email saying that he is doing a great job*
                    """
                    )
        st.caption("👉 Reply with: *What about 11:00?*")
        st.markdown("**3. Complex Orchestration (SQL + Calendar + Mail)**:")
        st.write("##### *The Supervisor must delegate to the Calendar agent (which calls the SQL agent), wait for confirmation, and then delegate to the Mail agent*")
        st.markdown("""
                    > *Create a meeting with Design team on Monday at 17, then notify them with an email*
                    
                    > *Create a meeting with Developer team on Friday at 10, then notify them with an email*
                    """
                    )
        st.caption("👉 Reply with: *Ok, when are all Developers available on Friday?*")
        st.caption("👉 Then: *Ok let's go with 16:00*")
  
        with st.expander("👀 What to watch"):
                    st.markdown("**1. Orchestration & Reasoning Trace**:")
                    st.caption("👇 *Check the **🧠 Agents' Reasoning Steps** section below*")
                                
                    st.markdown("""
                                Observe the architectural flow:
                                - The **Supervisor** delegating tasks to specific sub-agents.
                                - The **SQL Agent** performing schema inspection before query generation.
                                - The explicit **Chain of Thought** validation loops.
                                """)

                    st.markdown("**2. Cost Attribution & FinOps**:")
                    st.caption("👇 *Check the **📊 Live Metrics** section below*")
                                
                    st.markdown("""
                                Resource tracking goes beyond total costs. Notice the **per-agent granularity**, which allows for specific ROI analysis of the Supervisor vs. Sub-Agents.
                                """)

    st.markdown("---")    
    
    # ------------------------------
    # Metrics Section
    # ------------------------------
    st.subheader("📊 Live Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=":blue[Latency]", value=f"{st.session_state.latency:.2f}s", help="Time taken by all the agents to produce the LAST response.")
    with col2:
        st.metric(label=":blue[Session Cost]", value=f"${st.session_state.usd:.4f}", help="Estimated cost for the entire session, calculated using %.4f per 1K input tokens and %.4f per 1K output tokens" % (COST_PER_1K_INPUT, COST_PER_1K_OUTPUT))
    st.caption("💡 *Metrics update in real-time to track API costs.*")
    st.caption(f"Token Usage: :blue[{st.session_state.total_tokens}] total tokens used.", help="Total number of tokens consumed in the entire session - included the tokens used by tools.")
    
    with st.expander(":blue[🔎 Token Usage Breakdown]"):
        st.caption(f"Total Input Tokens: :blue[{st.session_state.total_input_tokens}] | Total Output Tokens: :blue[{st.session_state.total_output_tokens}]")
        st.caption(f"Last Input Tokens: :blue[{st.session_state.input_tokens_last}] | Last Output Tokens: :blue[{st.session_state.output_tokens_last}]")
        st.markdown(f"#### :blue[Last Interaction Tokens:] {st.session_state.total_tokens_last}", help="Number of tokens consumed in the last interaction - included the tokens used by tools.")
        col1, col2 = st.columns(2)
        col1.metric("Supervisor", f"{st.session_state.Supervisor_last_tokens['total_tokens']}")
        col2.metric("Mail Agent", f"{st.session_state.Mail_last_tokens['total_tokens']}")
        col3, col4 = st.columns(2)
        col3.metric("Calendar Agent", f"{st.session_state.Calendar_last_tokens['total_tokens']}")
        col4.metric("SQL Agent", f"{st.session_state.SQL_last_tokens['total_tokens']}")
        st.metric("Last Est. Cost (USD)", f"${st.session_state.usd_last:.4f}", help="Estimated cost for the last interaction, calculated using %.4f per 1K input tokens and %.4f per 1K output tokens" % (COST_PER_1K_INPUT, COST_PER_1K_OUTPUT))
    

    st.markdown("---")

    # ------------------------------
    # Reasoning Steps Section
    # ------------------------------
    st.subheader("🧠 Agents' Reasoning Steps")
    if not st.session_state.get("workflow_tree"):
        st.caption("Start a chat to see the agent's decision tree.")
    else:
        for root_node in st.session_state.workflow_tree:
            render_workflow_node(root_node, level=0)

    st.markdown("---")
    st.markdown("[View Source Code](https://github.com/danielecelsa/mail-calendar-assistant) • Developed by **[Daniele Celsa](https://danielecelsa.github.io/portfolio/)**")

# ------------------------------
# Render chat
# ------------------------------ 
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
    else:
        content = getattr(msg, "content", None) or str(msg)
        with st.chat_message("assistant"):
            st.write(content)

# ------------------------------
# INTERACTIVE TUTORIAL (Driver.js)
# ------------------------------

def get_tour_script(run_id):
    return f"""
    <!-- Run ID: {run_id} -->
    <script>
        function injectAndRunTour() {{
            const parentDoc = window.parent.document;
            const parentWin = window.parent;

            // --- 1. CSS Injection ---
            if (!parentDoc.getElementById('driver-js-css')) {{
                const link = parentDoc.createElement('link');
                link.id = 'driver-js-css';
                link.rel = 'stylesheet';
                link.href = 'https://cdn.jsdelivr.net/npm/driver.js@1.0.1/dist/driver.css';
                parentDoc.head.appendChild(link);
            }}

            // --- CUSTOM STYLING (Dark Mode Theme FIXED) ---
            const customStyle = `
                /* Main Popover Box */
                .driver-popover {{
                    background-color: #1e1e1e !important;
                    color: #ffffff !important;
                    border-radius: 12px;
                    /* Remove the solid border to avoid graphical conflicts with the arrow */
                    border: none; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
                    font-family: sans-serif;
                }}
                
                /* Title */
                .driver-popover-title {{
                    font-size: 18px;
                    font-weight: 600;
                    color: #61dafb; /* React Blue */
                    margin-bottom: 8px;
                }}
                
                /* Description */
                .driver-popover-description {{
                    font-size: 14px;
                    line-height: 1.6;
                    color: #e0e0e0;
                }}
                
                /* Buttons */
                .driver-popover-footer button {{
                    background-color: #333;
                    color: white;
                    border: 1px solid #555;
                    border-radius: 6px;
                    padding: 6px 12px;
                    text-shadow: none;
                    transition: background 0.2s;
                }}
                .driver-popover-footer button:hover {{
                    background-color: #555;
                }}
                
                /* Skip Button Custom Style */
                .driver-popover-footer .driver-skip-btn {{
                    background-color: transparent;
                    color: #ff6b6b; /* Soft Red */
                    border: none;
                    margin-right: auto; 
                    font-weight: bold;
                    cursor: pointer;
                    padding-left: 0;
                }}
                .driver-popover-footer .driver-skip-btn:hover {{
                    text-decoration: underline;
                    background-color: transparent;
                }}

                /* --- ARROW FIX --- */
                /* The arrow is generated by the borders. We need to color the side THAT TOUCHES the box */
                
                /* If the arrow is to the LEFT of the box (the box is to the right of the element) */
                .driver-popover-arrow-side-left {{
                    border-right-color: #1e1e1e !important; 
                }}
                
                /* If the arrow is to the RIGHT of the box */
                .driver-popover-arrow-side-right {{
                    border-left-color: #1e1e1e !important; 
                }}
                
                /* If the arrow is ABOVE the box */
                .driver-popover-arrow-side-top {{
                    border-bottom-color: #1e1e1e !important; 
                }}
                
                /* If the arrow is BELOW the box */
                .driver-popover-arrow-side-bottom {{
                    border-top-color: #1e1e1e !important; 
                }}
            `;

            if (!parentDoc.getElementById('driver-custom-style')) {{
                const style = parentDoc.createElement('style');
                style.id = 'driver-custom-style';
                style.innerHTML = customStyle;
                parentDoc.head.appendChild(style);
            }}

            // --- 2. JS Injection ---
            if (!parentDoc.getElementById('driver-js-script')) {{
                const script = parentDoc.createElement('script');
                script.id = 'driver-js-script';
                script.src = 'https://cdn.jsdelivr.net/npm/driver.js@1.0.1/dist/driver.js.iife.js';
                script.onload = () => runTour(parentWin, parentDoc);
                parentDoc.head.appendChild(script);
            }} else {{
                setTimeout(() => runTour(parentWin, parentDoc), 500);
            }}
        }}

        function runTour(parentWin, parentDoc) {{
            const driver = parentWin.driver.js.driver;
            
            // Helper to find elements by text content
            function findEl(tag, text, context = parentDoc) {{
                const elements = context.querySelectorAll(tag);
                for (const el of elements) {{
                    if (el.textContent.includes(text)) return el;
                }}
                return null;
            }}

            const sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');

            // Find the elements (that exist in the DOM even if closed)
            const archEl = findEl('summary', 'Architecture & Tech Stack', sidebar);
            const sqlEl = findEl('summary', 'SQLite DB', sidebar);

            const driverObj = driver({{
                showProgress: true,
                animate: true,
                allowClose: true,
                nextBtnText: 'Next →',
                prevBtnText: '← Back',
                doneBtnText: 'Finish',
                // Inject SKIP button
                onPopoverRendered: (popover) => {{
                    const footer = popover.wrapper.querySelector('.driver-popover-footer');
                    if (footer && !footer.querySelector('.driver-skip-btn')) {{
                        const skipBtn = document.createElement('button');
                        skipBtn.className = 'driver-skip-btn';
                        skipBtn.innerText = 'Skip Tutorial';
                        skipBtn.onclick = () => {{
                            driverObj.destroy();
                        }};
                        footer.insertBefore(skipBtn, footer.firstChild);
                    }}
                }},
                steps: [
                    {{ 
                        popover: {{ 
                            title: '👋 Welcome to Multi-Agents Ochestrator', 
                            description: 'A quick tour to demonstrate a Hierarchical Multi-Agents Orchestrator delegating tasks to Sub-Agents for Scheduling & Email Assistance.', 
                        }} 
                    }},
                    {{ 
                        element: parentDoc.querySelector('[data-testid="stChatInput"]'), 
                        popover: {{ 
                            title: '💬 Chat Console', 
                            description: 'Type here. You can ask the agent to schedule Meetings with your (fake) Team members and send Emails using natural language.', 
                            side: "top", align: 'start' 
                        }} 
                    }},
                    {{ 
                        element: findEl('summary', 'Architecture', sidebar), 
                        popover: {{ 
                            title: '🛠️ Tech Architecture', 
                            description: 'Open to see details about the Technical Stack used in this PoC.', 
                            side: "right", align: 'start' 
                        }} 
                    }},
                    // --- STEP WITH AUTO-OPEN ---
                    {{ 
                        element: sqlEl,
                        // This function is executed BEFORE showing the step
                        onHighlightStarted: () => {{
                            // If we found the Architecture element...
                            if (archEl && archEl.parentElement) {{
                                // ...force open the <details> tag (the parent expander)
                                archEl.parentElement.open = true;
                            }}
                        }},
                        popover: {{ 
                            title: '🗄️ Deterministic Data (SQL)', 
                            description: 'Check the REAL data tables used for this PoC. The SQL Agent queries these rosters and availability grids to ground its answers, preventing hallucinations', 
                            side: "right", align: 'start' 
                        }} 
                    }},
                    {{ 
                        element: findEl('summary', 'How to Test', sidebar), 
                        popover: {{ 
                            title: '🧪 Test Scenarios', 
                            description: 'Suggested prompts to test the agent capabilities. The coolest ones are those requiring multiple Sub-Agents to work together!', 
                            side: "right", align: 'start' 
                        }} 
                    }},
                    {{ 
                        element: findEl('h3', 'Live Metrics', sidebar), 
                        popover: {{ 
                            title: '📊 Live Metrics', 
                            description: 'Real-time monitoring of latency, tokens, and USD costs - broken down \\'Per-Agent\\' for granular FinOps analysis.', 
                            side: "right", align: 'start' 
                        }} 
                    }},
                    {{ 
                        element: findEl('h3', 'Reasoning Steps', sidebar), 
                        popover: {{ 
                            title: '🧠 Agent Reasoning', 
                            description: 'Visualize the "Chain of Thought". You will see the Supervisor assigning tasks and Sub-Agents executing tools in real-time, using the ReAct pattern.', 
                            side: "right", align: 'start' 
                        }} 
                    }},
                    {{ 
                        element: findEl('button', 'Restart Tutorial', sidebar), 
                        popover: {{ 
                            title: '🔄 Replay', 
                            description: 'Click here anytime to watch this tutorial again.', 
                            side: "top", align: 'center' 
                        }} 
                    }}
                ]
            }});

            driverObj.drive();
        }}

        injectAndRunTour();
    </script>
    """

# --- PYTHON LOGIC ---

if not st.session_state['tour_completed']:
    unique_run_id = str(time.time())
    components.html(get_tour_script(unique_run_id), height=0)
    st.session_state['tour_completed'] = True

# --- SIDEBAR BUTTON ---
with st.sidebar:
    st.markdown("---")
    def reset_tour():
        st.session_state['tour_completed'] = False
    
    st.button("🔄 Restart Tutorial", on_click=reset_tour)