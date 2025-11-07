# Agentic Personal Assistant with Calendar and Email Management
# Uses LangGraph ReAct framework with Google Gemini LLM.
# Capable of scheduling events and sending emails via natural language.

# ------------------------------
# Imports
# ------------------------------
import os
import threading
from pathlib import Path
import uuid
import logging
import queue
import re
import datetime
import json
import redis
import valkey
import time
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv

from helpers import (
    compute_cost
)

# LangGraph / LangChain Core
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Load environment variables from .env file if not in a rendering environment
if os.getenv("RENDER") != "true":
    load_dotenv()

# ------------------------------
# Custom Redis Logging Handler
# ------------------------------
class RedisLogHandler(logging.Handler):
    """
    A logging handler that publishes records to a capped Redis list.
    """
    def __init__(self, client, key, max_entries=500):
        super().__init__()
        self.client = client
        self.key = key
        self.max_entries = max_entries

    def emit(self, record):
        """
        Takes a log record, formats it, and pushes it to Redis.
        """
        try:
            # Format the log record into a string
            log_entry = self.format(record)
            # Push the entry to the left of the list
            self.client.lpush(self.key, log_entry)
            # Trim the list to keep only the latest max_entries
            self.client.ltrim(self.key, 0, self.max_entries - 1)
        except Exception:
            # If Redis fails, we don't want the logger to crash the app
            pass

# ------------------------------
# Configuration
# ------------------------------
MODEL = os.environ.get("GENAI_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

COST_PER_1K_INPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_INPUT", "0.002"))
COST_PER_1K_OUTPUT = float(os.getenv("COST_PER_1K_TOKENS_USD_OUTPUT", "0.002"))

LOG_DIR = Path(os.environ.get("CHAT_SUPERVISOR_LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("agent_test.db")
DB_URI = f"sqlite:///{DB_PATH}"

# ------------------------------
# Logging Setup
# ------------------------------
# Define the name for our Redis list for logs
REDIS_LOGS_KEY = "supervisor_logs"

# Get the top-level logger
logger = logging.getLogger("tool_logger")
logger.setLevel(logging.INFO)

# Prevent log messages from propagating to the root logger
logger.propagate = False

# Remove any existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# --- Create and add a handler for writing to a local file ---
file_handler = logging.FileHandler("logs/supervisor_debug.log", mode="a")
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


@st.cache_resource
def get_workflow_fallback_queue():
    """Gets a singleton in-memory queue instance for workflow fallback."""
    return queue.Queue()

@st.cache_resource
def get_usage_fallback_queue():
    """Gets a singleton in-memory queue instance for usage fallback."""
    return queue.Queue()

@st.cache_resource
@st.cache_resource
def get_kv_client():
    """
    Gets a singleton Valkey/Redis client connection.
    Uses REDIS_URL from environment variables for production (Render).
    Falls back to a local connection for development.
    """
    # Render provides the connection string for its Valkey service in the REDIS_URL env var
    redis_url = os.environ.get("REDIS_URL", None)

    try:
        if redis_url:
            # Production environment (Render) - use the provided URL
            logger.info("Connecting to Key-Value store via REDIS_URL.")
            kv_client = valkey.from_url(redis_url, decode_responses=True)
        else:
            # Local development - connect to your local Redis/Valkey instance
            logger.info("REDIS_URL not found. Connecting to localhost.")
            kv_client = valkey.Redis(host="localhost", port=6379, db=0, decode_responses=True)

        # Ping the server to check if the connection is alive
        kv_client.ping()
        logger.info("Successfully connected to Key-Value store.")
        return kv_client
        
    except valkey.exceptions.ConnectionError as e:
        logger.warning(f"Could not connect to Key-Value store: {e}. Using in-memory queue as fallback.")
        return None


# Get the singleton instances
kv_client = get_kv_client() # Valkey client (Redis-compatible)
WORKFLOW_QUEUE = get_workflow_fallback_queue() # This will be used only if Redis fails
USAGE_QUEUE = get_usage_fallback_queue()    # This will be used only if Redis fails

# Define the names for our Redis lists
REDIS_WORKFLOW_KEY = "workflow_queue"
REDIS_USAGE_KEY = "usage_queue"


# LOGGING -> Create and add the Redis logging handler IF kv_client is available ---
if kv_client:
    try:
        redis_handler = RedisLogHandler(client=kv_client, key=REDIS_LOGS_KEY)
        redis_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        redis_handler.setFormatter(redis_formatter)
        logger.addHandler(redis_handler)
        logger.info("Key-Value logging handler successfully configured.")
    except Exception as e:
        logger.warning(f"Failed to configure Key-Value logging handler: {e}")


# Initialize Streamlit session state variables
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

usd = 0.0
usd_last = 0.0

# ------------------------------
# Helpers
# ------------------------------
def buffer_workflow(key: str, entries: list[str]):
    if not entries:
        return
    
    # Debug: record process/thread so you can see where this runs
    logger.info("buffer_workflow called in PID=%s TID=%s key=%s entries=%d",
                os.getpid(), threading.get_ident(), key, len(entries))
    
    # Prepare the data packet to be stored
    data_packet = json.dumps({"key": key, "entries": entries})

    # Try to use Redis first
    if kv_client:
        try:
            kv_client.lpush(REDIS_WORKFLOW_KEY, data_packet)
            logger.info("Buffered workflow to Key-Value store for key=%s", key)
            return
        except valkey.exceptions.ConnectionError as e:
            logger.warning(f"Key-Value store connection error in buffer_workflow: {e}. Falling back to queue.")
            # Fall through to the queue logic below
    
    # Fallback logic
    try:
        WORKFLOW_QUEUE.put_nowait((key, entries))
    except Exception as e:
        logger.exception("Failed to buffer workflow to fallback queue: %s", e)


def buffer_usage(key: str, last_input: int, last_output: int, total_t: int):
    if not key or not last_input or not last_output or not total_t:
        return
    
    # Debug: record process/thread so you can see where this runs
    logger.info("buffer_usage called in PID=%s TID=%s key=%s total_token=%d",
                os.getpid(), threading.get_ident(), key, total_t)

    usage_data = {
        "key": key,
        "input_tokens": last_input,
        "output_tokens": last_output,
        "total_tokens": total_t
    }
    data_packet = json.dumps(usage_data)

    # Try to use Redis first
    if kv_client:
        try:
            kv_client.lpush(REDIS_USAGE_KEY, data_packet)
            logger.info("Buffered usage to Key-Value store for key=%s", key)
            return
        except valkey.exceptions.ConnectionError as e:
            logger.warning(f"Key-Value store connection error in buffer_usage: {e}. Falling back to queue.")
            # Fall through to the queue logic below
    
    # Fallback logic
    try:
        USAGE_QUEUE.put_nowait(usage_data)
    except Exception as e:
        logger.exception("Failed to buffer usage to fallback queue: %s", e)


def flush_workflow_queue():
    """Called from main Streamlit flow to merge buffered entries into st.session_state."""

    # --- Try flushing from Key-Value store first ---
    if kv_client:
        try:
            # RPOP atomically removes and returns the last element. Loop until the list is empty.
            while data_packet := kv_client.rpop(REDIS_WORKFLOW_KEY):
                item = json.loads(data_packet)
                key, entries = item["key"], item["entries"]
                logger.info("Flushing workflow from Key-Value store: key=%s entries=%d", key, len(entries))
                if key not in st.session_state or not isinstance(st.session_state.get(key), list):
                    st.session_state[key] = []
                st.session_state[key].extend([e for e in entries if e and (not isinstance(e, str) or e.strip())])
            return # If we successfully processed Key-Value store, we are done.
        except valkey.exceptions.ConnectionError as e:
            logger.warning(f"Key-Value store connection error during flush_workflow_queue: {e}. Checking fallback queue.")
            # Fall through to check the in-memory queue

    # --- Fallback: Flush from in-memory queue ---
    try:
        while not WORKFLOW_QUEUE.empty():
            key, entries = WORKFLOW_QUEUE.get_nowait()
            logger.info("Flushing workflow from fallback queue: key=%s entries=%d", key, len(entries))
            if key not in st.session_state or not isinstance(st.session_state.get(key), list):
                st.session_state[key] = []
            st.session_state[key].extend([e for e in entries if e and (not isinstance(e, str) or e.strip())])
    except Exception as e:
        logger.exception("Error flushing workflow fallback queue: %s", e)


def flush_usage_queue():
    """Called from main Streamlit flow to merge buffered entries (usage) into st.session_state."""

    # --- Try flushing from Key-Value store first ---
    if kv_client:
        try:
            while data_packet := kv_client.rpop(REDIS_USAGE_KEY):
                usage = json.loads(data_packet)
                key = usage.get("key")
                logger.info("Flushing usage from Key-Value store: key=%s usage=%s", key, usage)
                
                # Ensure a dict exists for this agent key
                if key not in st.session_state or not isinstance(st.session_state.get(key), dict):
                    st.session_state[key] = {
                        "input_tokens": 0, 
                        "output_tokens": 0, 
                        "total_tokens": 0
                        }

                # Accumulate per-agent tokens (sum multiple queued usages)
                existing = st.session_state[key]
                
                new_input = usage.get("input_tokens", 0)
                new_output = usage.get("output_tokens", 0)
                new_total = usage.get("total_tokens", 0)
                
                st.session_state[key] = {
                    "input_tokens": existing.get("input_tokens", 0) + new_input,
                    "output_tokens": existing.get("output_tokens", 0) + new_output,
                    "total_tokens": existing.get("total_tokens", 0) + new_total
                }

                # Also update the accumulated counters used for "last" and "total" displays
                st.session_state.input_tokens_last += new_input
                st.session_state.output_tokens_last += new_output
                st.session_state.total_tokens_last += new_total
                st.session_state.total_input_tokens += new_input
                st.session_state.total_output_tokens += new_output
                st.session_state.total_tokens += new_total
            return # If we successfully processed Key-Value store, we are done.

        except valkey.exceptions.ConnectionError as e:
            logger.warning(f"Key-Value store connection error during flush_usage_queue: {e}. Checking fallback queue.")
            # Fall through to check the in-memory queue

    # --- Fallback: Flush from in-memory queue ---
    try:
        while not USAGE_QUEUE.empty():
            usage = USAGE_QUEUE.get_nowait()
            logger.info("Flushing usage from fallback queue: usage=%s", usage)
            key = usage.get("key")

            # the rest of the logic is identical
            if key not in st.session_state or not isinstance(st.session_state.get(key), dict):
                st.session_state[key] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            existing = st.session_state[key]
            
            new_input = usage.get("input_tokens", 0)
            new_output = usage.get("output_tokens", 0)
            new_total = usage.get("total_tokens", 0)
            
            st.session_state[key] = {
                "input_tokens": existing.get("input_tokens", 0) + new_input,
                "output_tokens": existing.get("output_tokens", 0) + new_output,
                "total_tokens": existing.get("total_tokens", 0) + new_total
            }
            
            st.session_state.input_tokens_last += new_input
            st.session_state.output_tokens_last += new_output
            st.session_state.total_tokens_last += new_total
            st.session_state.total_input_tokens += new_input
            st.session_state.total_output_tokens += new_output
            st.session_state.total_tokens += new_total

    except Exception as e:
        logger.exception("Error flushing usage fallback queue: %s", e)


def get_workflow(agent_answer: dict, agent: int):
    """
    Get the workflow between agents in the last interaction
    Instead of writing directly into st.session_state (which may be inside a nested callback),
    accumulate entries into WORKFLOW_QUEUE. The main flow will flush the buffer into session_stat
    """
    # Debug: record process/thread so you can see where this runs
    logger.info("get_workflow called in PID=%s TID=%s agent=%s", os.getpid(), threading.get_ident(), agent)

    msgs = agent_answer.get("messages", [])
    
    if agent == 0:
        msgs = msgs[st.session_state.to_cut:]
        st.session_state.to_cut += len(msgs)

    first_content = getattr(msgs[0], "content", '') if (len(msgs) > 0 and hasattr(msgs[0], "content")) else (msgs[0].get("content") if (len(msgs) > 0 and isinstance(msgs[0], dict)) else '')
    super_first_content = getattr(msgs[1], "content", '') if (len(msgs) > 1 and hasattr(msgs[1], "content")) else (msgs[1].get("content") if (len(msgs) > 1 and isinstance(msgs[1], dict)) else '')
    first_id = getattr(msgs[0], "id", None) if (len(msgs) > 0 and hasattr(msgs[0], "id")) else (msgs[0].get("id") if (len(msgs) > 0 and isinstance(msgs[0], dict)) else None)
    super_first_id = getattr(msgs[1], "id", None) if (len(msgs) > 1 and hasattr(msgs[1], "id")) else (msgs[1].get("id") if (len(msgs) > 1 and isinstance(msgs[1], dict)) else None)
    print(f"First content: {first_content}")
    print(f"First ID: {first_id}")

    var = []
    if agent == 0: # SUPERVISOR
        var = "Supervisor"
    elif agent == 1: # SQL AGENT
        var = "SQL"
    elif agent == 2: # MAIL AGENT
        var = "Mail"
    elif agent == 3: # CALENDAR AGENT
        var = "Calendar"
    
    print(f"{var}")

    key = f"{var}_agent_history"

    print('HELLO')
    
    # Build local entries and put them into WORKFLOW_BUFFER instead of st.session_state
    local_entries = []
    total_token = 0

    if agent != 0:
        try:
            local_entries.append("_**Request**_:  \n" + first_content)
            print("OK!")
        except Exception as e:
            logger.exception("Errore building local workflow entries : %s", e)

    else:
        local_entries.append("_**Request**_:  \n" + super_first_content)
        print("OK!!!!!")
    

    for m in msgs:
        print(f"ID: {m.id}")
        
        try:
            mid = getattr(m, "id", None) if hasattr(m, "id") else (m.get("id") if isinstance(m, dict) else None)
        except Exception:
            mid = None
        
        if ((mid != first_id) & (mid != super_first_id if (agent == 0) else True)):
            try:
                content = getattr(m, "content", '') if hasattr(m, "content") else (m.get("content") if isinstance(m, dict) else '')
                # kwarg = getattr(m, "additional_kwarg", None) or m.additional_kwargs or (m.get("additional_kwarg") if isinstance(m, dict) else None)
                tool_calls = getattr(m, "tool_calls", None) if hasattr(m, "tool_calls") else (m.get("tool_calls") if isinstance(m, dict) else None)
                usage_metadata = getattr(m, "usage_metadata", None) if hasattr(m, "usage_metadata") else (m.get("usage_metadata") if isinstance(m, dict) else None)
                tool_name = getattr(m, "name", None) if hasattr(m, "name") else (m.get("name") if isinstance(m, dict) else None)
                status = getattr(m, "status", None) if hasattr(m, "status") else (m.get("status") if isinstance(m, dict) else None)
            except Exception as e:
                logger.info("Error getting info from agent answer: %s", e)
                content = ""

            if isinstance(m, AIMessage):
                local_entries.append(f'-> _**{var} Agent msg**_:  \n' + content)
                if tool_calls:
                    local_entries.append("-> _**Call Tool**_:  \n" + tool_calls[0]["name"])
                    local_entries.append("-> _**With ARGS**_:  \n" + str(tool_calls[0]["args"]))
                if usage_metadata:
                    local_entries.append("-> _**TOKEN USAGE** (agent)_:  \nInput= " + str(usage_metadata.get("input_tokens")) + ", Output= " + str(usage_metadata.get("output_tokens")) + ", TOT= " + str(usage_metadata.get("total_tokens")))
                    logger.info(f'{var} Tokens: Input= ' + str(usage_metadata.get("input_tokens")) + ", Output= " + str(usage_metadata.get("output_tokens")) + ", TOT= " + str(usage_metadata.get("total_tokens")))
                    total_token += usage_metadata.get("total_tokens")


            elif isinstance(m, ToolMessage):
                local_entries.append("-> _**Tool Name**_:  \n" + tool_name)
                local_entries.append("-> _**Tool msg**_:  \n" + content)
                if status:
                    local_entries.append("-> _**Status**_:  \n" + status)
            
            local_entries.append("####---------------------------------------####")

    logger.info(f'{var} TOTAL Tokens: {total_token}')

    if local_entries:
        # Debug: log where we buffered
        logger.info("Writing %d local_entries for %s (PID=%s TID=%s)",
                    len(local_entries), key, os.getpid(), threading.get_ident())
        buffer_workflow(key, local_entries)

    logger.info("Buffered Queue Workflow %s: %s", key, local_entries)


def get_usage(usage_metadata: dict, agent: int):
    # Token usage extraction
    try:
        # extract token usage from callback (last interaction only) - sum all if multiple calls/models
        last_input = sum(usage.get("input_tokens", 0) for usage in usage_metadata.values())
        print('Input: ', last_input)
        last_output = sum(usage.get("output_tokens", 0) for usage in usage_metadata.values())
        print('Output: ', last_output)
        # total tokens in the whole thread (not just last interaction) 
        total_t = sum(usage.get("total_tokens", 0) for usage in usage_metadata.values())
        print('TOTAL: ', total_t)
        
    except Exception as e:
        logger.exception("Could not extract token usage from callback: %s", e)
        last_input = 0
        last_output = 0

    var = []
    if agent == 0: # SUPERVISOR
        var = "Supervisor"
    elif agent == 1: # SQL AGENT
        var = "SQL"
    elif agent == 2: # MAIL AGENT
        var = "Mail"
    elif agent == 3: # CALENDAR AGENT
        var = "Calendar"
    
    print(f"{var} USAGE")

    key_tokens = f"{var}_last_tokens"

    buffer_usage(key_tokens, last_input, last_output, total_t)

    logger.info("Buffered Queue Usage %s: %d", key_tokens, total_t)


def resolve_natural_date(request_text: str, reference_date: datetime.date | None = None):
    """
    Try to resolve "Monday at 9", "next Monday 09:00", "tomorrow at 17:00", or explicit ISO date "2025-11-03".
    Returns tuple (date_iso: str, time_hhmm: str) or (None, None) if not found.
    Heuristic-only, useful for demos: assumes 'next' occurrence for weekday-only inputs.
    """
    if not request_text:
        return None, None

    ref = reference_date or datetime.date.today()
    text = request_text.lower()

    # explicit ISO date yyyy-mm-dd
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m:
        date_iso = m.group(1)
    else:
        # weekday name
        w = re.search(r'\b(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', text)
        if w:
            weekday_name = w.group(2)
            target_wd = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday'].index(weekday_name)
            today_wd = ref.weekday()  # Monday=0
            days_ahead = (target_wd - today_wd) % 7
            # if same day -> schedule next week's occurrence (assume "next" behavior for demo)
            if days_ahead == 0:
                days_ahead = 7
            target_date = ref + datetime.timedelta(days=days_ahead)
            date_iso = target_date.isoformat()
        else:
            # relative words
            if 'tomorrow' in text:
                date_iso = (ref + datetime.timedelta(days=1)).isoformat()
            elif 'today' in text:
                date_iso = ref.isoformat()
            else:
                date_iso = None

    # time extraction: supports "at 9", "at 9:00", "at 17:30", with optional am/pm
    time_iso = None
    tm = re.search(r'at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text)
    if tm:
        hour = int(tm.group(1))
        minute = int(tm.group(2)) if tm.group(2) else 0
        ampm = tm.group(3)
        if ampm:
            if ampm == 'pm' and hour < 12:
                hour += 12
            if ampm == 'am' and hour == 12:
                hour = 0
        time_iso = f"{hour:02d}:{minute:02d}"
    else:
        # also accept "at 09:00" with colon already covered, fallback to searching hh:mm anywhere
        tm2 = re.search(r'(\d{1,2}:\d{2})', text)
        if tm2:
            time_iso = tm2.group(1)

    return (date_iso, time_iso)


def wait_and_flush(timeout: float = 1.0, stable_period: float = 0.05):
    start = time.time()
    # initial flush
    flush_workflow_queue()
    flush_usage_queue()
    last_total = WORKFLOW_QUEUE.qsize() + USAGE_QUEUE.qsize()
    last_change_time = time.time()
    while time.time() - start < timeout:
        total = WORKFLOW_QUEUE.qsize() + USAGE_QUEUE.qsize()
        if total != last_total:
            logger.info("wait_and_flush: queue size changed %s -> %s; flushing", last_total, total)
            last_total = total
            last_change_time = time.time()
            flush_workflow_queue()
            flush_usage_queue()
        else:
            if time.time() - last_change_time >= stable_period:
                break
        time.sleep(0.01)


@st.cache_resource
def get_db():
    """Get the SQLDatabase instance connected to the Chinook sample database."""

    db = SQLDatabase.from_uri(DB_URI)

    print(f"Dialect: {db.dialect}")
    print(f"Available tables: {db.get_usable_table_names()}")
    print(f'Sample output: {db.run("SELECT * FROM staff LIMIT 5;")}')

    return db

# ------------------------------
# LLM
# ------------------------------
@st.cache_resource
def get_llm(temp: float = 0.1):
    """Create an LLM instance with a specific temperature (use for per-agent control)."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temp,
            #convert_system_message_to_human=True,
            safety_settings=None,
            )
    except Exception as e:
        logger.exception("Could not initialize LLM: %s", e)
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
        logger.info("create_calendar_event Created event: %s on %s at %s with attendees %s", title, date, start_time, attendees)
        pass  # Assume event created successfully
    else:
        logger.error("create_calendar_event Failed to create event: missing information.")
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
        logger.error("send_email Failed to send email: missing information.")
        raise ValueError("Missing required email information.")  
    
    logger.info("send_email Sent email to: %s with subject: %s", to, subject)

    return f"Email sent to {', '.join(to)} - Subject: {subject} - Body length: {len(body)} characters"


# ------------------------------
# SQL Database Tools - we don't need @tool here cause SQLDatabaseToolkit.get_tools() does it
# ------------------------------
@st.cache_resource
def get_sql_tools():
    """Get the list of tools for the SQL agent."""
    toolkit = SQLDatabaseToolkit(db=get_db(), llm=get_llm())

    tools = toolkit.get_tools()

    for tool in tools:
        print(f"{tool.name}: {tool.description}\n")

    return tools


# ------------------------------
# Build SQL Sub-Agent
# ------------------------------
@st.cache_resource
def get_sql_agent():
    
    SQL_AGENT_PROMPT = (
        """
        You are an agent whose only job is to answer questions by reading a SQL database in READ-ONLY mode.
        Given an input question, you will create a syntactically correct SQL query to run.
        You MUST follow these rules strictly and in order. Do not deviate.

        1) Tools allowed (and the only ones you may call): 
        - sql_db_list_tables
        - sql_db_schema
        - sql_db_query_checker
        - sql_db_query

        2) NEVER run any DML or DDL. If a user asks to modify data, state clearly that you only have read-only access and offer the SELECT queries you would run instead.

        3) Parse the user's natural language input FIRST (no tool calls yet). Extract and normalize:
        - person: first name and optional surname.
        - team name (if present).
        - day and time if this is an availability question.
        - intent (list emails, check availability, schedule planning, etc).

        Normalizations you MUST perform before building SQL:
        - Day normalization: map user variations into a canonical day token for display (Title Case), e.g. "monday", "MONDAY", "Mon", "lunedi" -> "Monday". Use Title Case for human-readable answers. 
            But for the SQL comparison always use case-insensitive matching (see rule 4.c).
        - Time normalization: normalize times to "HH:MM" 24-hour format. Examples:
            "9" -> "09:00", "9:00" -> "09:00", "0900" -> "09:00", "14" -> "14:00".
            If user gives a range or multiple times, parse accordingly.
        - Name normalization: trim whitespace and prefer full_name_norm when available.

        4) Canonical query-workflow (always follow):
        a) Call sql_db_list_tables to confirm available tables.
        b) Call sql_db_schema on relevant table(s) (staff, staff_name_mapping, availability) to learn exact column names and to sample how 'day' values are stored (e.g., "MONDAY", "Monday", "monday").
        c) Build a SELECT query using CASE-INSENSITIVE comparisons for free-text values. Preferred patterns:
            - Team: `WHERE LOWER(team) = LOWER(:team)`  OR `team_lower = :team_lower` (if such column exists).
            - First name: `WHERE LOWER(name) = LOWER(:firstname)`.
            - Full name: `WHERE full_name_norm = :full_name_norm` OR `WHERE LOWER(name) = LOWER(:firstname) AND LOWER(surname) = LOWER(:surname)`.
            - Day: use `LOWER(day) = LOWER(:day)` or `UPPER(day) = UPPER(:day)` in WHERE clauses.
            - Time: compare normalized `time` strings, e.g. `time = :time` where `:time` = "09:00".
            Example availability query (preferred, case-insensitive on day):
            ```sql
            SELECT time FROM availability
            WHERE LOWER(day) = LOWER(:day) AND "Marco_Rossi" = 1
            ORDER BY time;
            ```
            Or explicitly:
            ```sql
            SELECT day, time, "Marco_Rossi" as available
            FROM availability
            WHERE UPPER(day) = UPPER(:day) AND time = :time;
            ```
            Use quoting for identifiers when necessary (e.g., "Marco_Rossi") and ensure syntactic validity for {dialect}.
        d) Once you have composed your SQL query string, decide whether it requires validation:

            • If the query is simple and follows a standard SELECT pattern 
                (e.g., a single-table SELECT with WHERE conditions and no joins or subqueries), 
                you may SKIP the sql_db_query_checker to save cost and latency.
                In these simple cases, call sql_db_query directly.

            • If the query is complex (contains joins, subqueries, GROUP BY, HAVING, or text-based pattern matching),
                or if you are unsure about its syntax, then call sql_db_query_checker first.
                Use its corrected output (if provided), and only then run sql_db_query.

            • Regardless of checker usage, ALWAYS verify column names first with sql_db_schema before referencing them.

        5) Disambiguation policy:
        - If user gives both first name and surname -> treat as exact target (do not ask to clarify); use full_name_norm if available.
        - If user gives only first name -> find all staff with that name:
            `SELECT name, surname, full_name_norm, email FROM staff WHERE LOWER(name) = LOWER(:firstname);`
            - 0 matches -> ask the user to clarify.
            - 1 match -> use that full_name_norm.
            - >1 match -> present the list and ask which one, unless user asked to check all.

        6) Team name policy:
        - Treat team names case-insensitively. Use `LOWER(team) = LOWER(:team)` or `team_lower` if available.

        7) Availability checks (robust):
        - If checking availability, first identify the exact person(s) (rule 5). Then produce ONE SQL query (not per-time loops) that returns the requested information:
            - For “is X available on Day at Time?”: prefer:
                ```sql
                SELECT day, time, "<Full_Name_Norm>" AS available
                FROM availability
                WHERE LOWER(day) = LOWER(:day) AND time = :time;
                ```
            - For “when is X available on Day?”:
                ```sql
                SELECT time FROM availability
                WHERE LOWER(day) = LOWER(:day) AND "<Full_Name_Norm>" = 1
                ORDER BY time;
                ```
        - Normalize times to HH:MM before building the query.
        - If the schema sample rows indicate days are stored in UPPERCASE (e.g., "MONDAY"), prefer still to use LOWER(...) or UPPER(...) to match robustly; do not rely on exact-case string equality.

        8) Day/time fallback strategies:
        - If `LOWER(day) = LOWER(:day)` returns no rows, try a quick schema-based fallback:
            - Inspect sample rows from sql_db_schema: if day values seem stored as uppercase, attempt `UPPER(day) = UPPER(:day)`.
            - If still no rows, try common synonyms/abbrs (Mon, Tue) or localized names (Lunedi, Lunedì) mapping to canonical English day names (only if your dataset mixes locales).
        - For times, if exact `time = :time` returns no rows, attempt to:
            - normalize seconds (strip seconds),
            - consider "nearest" 15-min slot only if user asked for suggestion (do not assume availability otherwise).

        9) Defensive rules:
        - Limit results to at most {top_k} by default for non-availability listing queries. For availability list (when user asked "when is X available"), do **not** arbitrarily limit times unless user requests otherwise.
        - Avoid `SELECT *` unless user explicitly requests all columns.
        - Validate column names using sql_db_schema before referencing them; if a column is missing, use staff_name_mapping or full_name_norm to find a canonical column.
        - Log and show your normalized day/time in any debug information you print (for traceability).

        10) Tool-call etiquette:
            - Always show your proposed SQL query clearly in your reasoning or call parameters.
            - If the query is a simple, safe SELECT (single table, no joins, no aggregations), 
            you may call sql_db_query directly to optimize performance.
            - For more complex or uncertain SQL, use sql_db_query_checker first:
                1) Pass the full SQL string to sql_db_query_checker.
                2) If the checker suggests corrections, apply them before execution.
                3) Then call sql_db_query.
            - Never call the checker unnecessarily on trivial queries (like checking a single column’s value).
            - After getting results, synthesize a concise, plain-language answer describing the findings.

        11) Examples (do exactly like these):
        - User: "Is Daniele Celsa available on Monday at 10?"
            Steps:
            1) parse person = "Daniele Celsa", day_raw="Monday", time_raw="10" -> normalize time="10:00", normalize day display = "Monday"
            2) sql_db_list_tables -> sql_db_schema('availability')
            3) propose SQL to checker:
                `SELECT day, time, "Daniele_Celsa" AS available FROM availability WHERE LOWER(day) = LOWER('Monday') AND time = '10:00';`
            4) run checker, then run sql_db_query
            5) if result row shows available=1 -> answer "Yes, Daniele Celsa is available on Monday at 10:00."
        - User: "when is Anna available on monday?"
            1) normalize day display="Monday"
            2) build single query:
                `SELECT time FROM availability WHERE LOWER(day) = LOWER('Monday') AND "Anna_Garau" = 1 ORDER BY time;`
            3) run checker -> run query -> synthesize list (e.g., "Anna is available on Monday at 09:00 and 11:00").

        12) If you cannot follow these steps (missing columns, ambiguous names, unknown day format), STOP and ask the user a clarifying question rather than guessing.

        Be strict and literal: follow these steps exactly for every user query.
        """.format(
            dialect=get_db().dialect,
            top_k=5,
        )
    )
    
    temp = 0.0

    sql_agent = create_react_agent(
        model=get_llm(temp),
        tools=get_sql_tools(), # qui inseriamo i tool che vengono da SQLDB
        prompt=SystemMessage(content=SQL_AGENT_PROMPT)
    )

    return sql_agent


# ------------------------------
# SQL Sub-agent as Tool
# ------------------------------
@tool
def check_staff_info(request: str) -> str:
    """
    Check info about the staff.

    Use this to check staff emails, availability, team membership, etc.

    The request should specify what info is needed about which staff member(s) or team.
    If the request is ambiguous (e.g., only first name provided), the agent will ask for clarification.

    The database contains tables like 'staff' (with columns like name, surname, full_name_norm, email, team) and 'availability' (with columns like day of the week, time, and one column per staff member indicating availability - True/False - at that time/day).

    Input: Natural language request about staff info (e.g., 'Is Marco available
    Monday at 09:00?',  'What are the emails of the developer team members?')
    """

    callback = UsageMetadataCallbackHandler()
    config = {"callbacks": [callback]}

    result = get_sql_agent().invoke({
        "messages": [{"role": "user", "content": request}]
        },
        config=config
    )
    get_workflow(result, 1)
    get_usage(callback.usage_metadata, 1)

    logger.info("SQL_AGENT RESULT: %s", result)
    logger.info("SQL_AGENT CALBACKS: %s", callback.usage_metadata)

    return result["messages"][-1].text


# ------------------------------
# Build Calendar/Mail Sub-Agents
# ------------------------------
@st.cache_resource
def get_calendar_agent():
    
    CALENDAR_AGENT_PROMPT = (
        """
        You are CalendarAgent, a deterministic calendar scheduling assistant. Use ONLY the runtime tools:
            - create_calendar_event(title:str, date:str, start_time:str, attendees:list[str])
            - check_staff_info(request:str) -> returns staff emails, availability, or team membership info
            - (Optional) mail_agent(...) only if available

        OVERVIEW (must follow, in order)
        1) Parse the user's request into structured fields:
            - title (optional)
            - day or date (weekday like "Monday" or exact date YYYY-MM-DD)
            - time (start time, e.g. "17" or "17:00")
            - duration (minutes, default 30)
            - attendees (names, team names, or emails)
            - notify flag (boolean)
        2) Resolve attendees FIRST (do not ask for emails if you can fetch them):
            - If user says "<TEAM> team", call check_staff_info:
                e.g. "Who are the members of the DESIGN team? Provide names and emails."
            - If user provides person names, call check_staff_info:
                e.g. "Find staff: Marco Rossi"
            - If check_staff_info returns multiple matches for a person, ask a single focused disambiguation question listing the options.
        3) Normalize day/time BEFORE availability checks:
            - If user gives only a weekday (e.g., "Monday at 9"), ASSUME the next upcoming occurrence of that weekday relative to the system date (i.e., "next Monday") for demo convenience. Do NOT ask the user for the absolute date.
            - Normalize time into 24-hour HH:MM (examples: "9" -> "09:00", "5pm" -> "17:00").
            - When building payloads include ISO date YYYY-MM-DD and HH:MM.
        4) Check availability for ALL resolved attendees:
            - For each resolved attendee (with a staff identity), call check_staff_info with a precise string:
                e.g. "Is Daniele Celsa available on Monday at 09:00?"
            - Only create the event if ALL attendees are available at requested time.
            - If one or more are unavailable, return a concise list of unavailable attendees and suggest up to 3 alternative times (derive suggestions from returned availability).
        5) Event creation:
            - Build final create_calendar_event payload with:
                title (if missing, auto-generate: "Meeting - <team or names>"),
                date (ISO YYYY-MM-DD),
                start_time (HH:MM or full datetime as required),
                attendees (list of emails).
            - CALL create_calendar_event only when the payload is complete and availability checks passed.
        6) Clarification policy:
            - After attendee-resolution attempt: if attendees cannot be resolved (team not found or no emails), ask one focused question to obtain missing detail.
            - If user insisted on a specific absolute date, respect it. If they said only "Monday", assume next occurrence unless user objects.
        7) Final confirmation:
            - After creating the event, reply with a short confirmation including ISO date, time, attendee emails, and any event id.
            - If event not created, ask a single focused follow-up (e.g., "Daniele is not available at 17:00 — prefer 16:00 or 17:30?").

        OPERATIONAL / TRACEABILITY RULES:
            - Always include normalized date/time in tool payloads and log the exact strings used when calling check_staff_info.
            - Phrase check_staff_info requests precisely (examples above) and show them in your internal reasoning.
            - If a step fails (ambiguous results, missing columns, no tool result), STOP and ask one clarifying question.
            - Be procedural: resolve attendees -> normalize date/time -> check availability -> create event -> confirm.
            - Be concise and deterministic; avoid unnecessary clarifying questions for demo flows.
        """
    )

    temp = 0.0

    calendar_agent = create_react_agent(
        model=get_llm(temp),
        tools=[create_calendar_event, check_staff_info],
        prompt=SystemMessage(content=CALENDAR_AGENT_PROMPT)
        )
    
    return calendar_agent


@st.cache_resource
def get_mail_agent():
    
    MAIL_AGENT_PROMPT = (
        """
        You are MailAgent, an email assistant. Use ONLY the provided tools:
        - send_email(to:list[str], subject:str, body:str)
        - check_staff_info(request:str)

        RULES (must follow):
        1) Resolve recipients first. If user says "<team> team", call check_staff_info:
        e.g. "Who are the members of the DESIGN team? Provide names and emails."
        Do NOT ask the user for emails if check_staff_info can resolve them.
        2) If recipient names are ambiguous, ask a single disambiguation question listing options.
        3) If subject or body is missing, auto-generate a concise subject/body based on context. Keep tone appropriate to the request, and sign yourrself as 'Dani'.
        4) Only call send_email when recipient emails, subject and body are present.
        5) After send_email, respond with a short confirmation including recipients, subject and body.

        Be concise, deterministic and include any resolved recipient emails in your final message.
        """
    )

    temp = 0.2

    mail_agent = create_react_agent(
        model=get_llm(temp),
        tools=[send_email, check_staff_info],
        prompt=SystemMessage(content=MAIL_AGENT_PROMPT)
        )
    
    return mail_agent


# ------------------------------
# Calendar/Mail Sub-agents as Tools
# ------------------------------
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create calendar events.

    Provide as much details as possible in the request. 
    For example, for scheduling events include title, date, start time, and attendees, if possible.
    Some information may be missing, in which case the agent will ask for clarification or create generic titles if needed.
    But some others are essential to proceed, like date/start time and attendees.

    Input: Natural language request about scheduling calendar events (e.g., 'Schedule a meeting with Marco next Tuesday at 2pm').
    """

    callback = UsageMetadataCallbackHandler()
    config = {"callbacks": [callback]}

    # Try to resolve weekday+time to a concrete date/time for demo convenience
    date_iso, time_iso = resolve_natural_date(request)
    annotated_request = request
    if date_iso and time_iso:
        annotated_request = f"{request} (assumed_date:{date_iso}, start_time:{time_iso})"
    elif date_iso and not time_iso:
        annotated_request = f"{request} (assumed_date:{date_iso})"
    elif time_iso and not date_iso:
        annotated_request = f"{request} (assumed_start_time:{time_iso})"

    result = get_calendar_agent().invoke({
        "messages": [{"role": "user", "content": annotated_request}]
        },
        config=config
    )

    get_workflow(result, 3)
    get_usage(callback.usage_metadata, 3)

    logger.info("CALENDAR_AGENT RESULT: %s", result)
    logger.info("CALENDAR_AGENT CALBACKS: %s", callback.usage_metadata)
    
    return result["messages"][-1].text


@tool
def manage_mail(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send emails.
    Emails can be about any topic, including notifications about scheduled events.

    Provide as much details as possible in the request. 
    For example, include recipients, subject, and body content, if possible.
    Some information may be missing, in which case the agent will ask for clarification or create generic subject or body if context is clear enough.
    But some others are essential to proceed, like recipients and content.

    Input: Natural language request about sending email (e.g., 'Send an email to the design team about the project update').
    """

    callback = UsageMetadataCallbackHandler()
    config = {"callbacks": [callback]}

    result = get_mail_agent().invoke({
        "messages": [{"role": "user", "content": request}]
        },
        config=config
    )

    get_workflow(result, 2)
    get_usage(callback.usage_metadata, 2)

    logger.info("MAIL_AGENT RESULT: %s", result)
    logger.info("MAIL_AGENT CALBACKS: %s", callback.usage_metadata)
    
    return result["messages"][-1].text


# ------------------------------
# Supervisor Prompt
# ------------------------------
@st.cache_resource
def get_supervisor_prompt():
    """Get the prompt template for the supervisor agent."""
    
    system = SystemMessagePromptTemplate.from_template(
        """
        You are a helpful personal assistant.
        You can schedule calendar events and send emails, or simply provide information about the teams/staff (mail addresses, availability at a certain time-slot, team members, etc) by querying the company database.
        Break down user requests into appropriate tool calls and coordinate the results.
        When a request involves multiple actions, use multiple tools in sequence or in parallel as needed.
        Always think step-by-step about what needs to be done, and which tool to call for each step.

        You can use the following tools:
        - schedule_event: to create calendar events based on natural language requests.
        - manage_mail: to send emails based on natural language requests.
        - check_staff_info: to check staff emails, availability, team membership, etc by querying the company database.

        Use the check_staff_info ONLY if the user question is specifically related to staff information (like checking availability, getting email addresses, team members, etc).
        Examples of when to use check_staff_info:
        - "Is Marco available next Monday at 10am?"
        - "What are the emails of the developer team members?"
        Do not use it for scheduling or email sending, those tools have their own sub-agents that will call check_staff_info as needed.
        Example of when NOT to use check_staff_info:
        - "Schedule a meeting with Marco next Tuesday at 2pm"
        - "Send an email to the design team about the project update"
        For scheduling events or sending emails, always use schedule_event and manage_mail respectively.

        Example of using multiple tools in sequence:
        User: "Schedule a meeting with Marco next Tuesday at 2pm and send him a confirmation email."
        Assistant:
        1) Use schedule_event to create the meeting with Marco next Tuesday at 2pm. The tool will check his availability automatically.
        2) Once the event is successfully created, use manage_mail to send Marco a confirmation email about the scheduled meeting. The tool will gather his email address automatically.
        3) Finally, inform the user that the meeting has been scheduled and the confirmation email has been sent.

        If a tool needs more information once called, be sure you don't already have it before asking the user for clarification, and once you have it, call again the tool with a info-updated request.
        
        Prefer using schedule_event/manage_mail for scheduling and email tasks; do not call check_staff_info directly for scheduling — use the sub-agents.
        
        Always provide a final summary of actions taken in your last response. 
        
        Additional rules:
        - If the user ask to send email, when you provide the final summary also provide the body of the email
        - If the user ask to schedule an event, when you provide the final summary also provide the date and time of the event
        - If the user ask to send email about a scheduled event, make sure the event was actually scheduled before sending the email
        - If the user ask to send email but he does not provide the body or subject, you don't need to ask for it, manage_mail will generate it for you based on the context
        - If the user ask to schedule an event but he does not provide the title, you don't need to ask for it, schedule_event will generate it for you based on the context
        - If the user ask to send email but he does not provide the recipients (email addresses or names), ask for it before calling manage_mail
        """
    )

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
    return checkpointer


# ------------------------------
# Build Supervisor
# ------------------------------
def build_supervisor():
    """Build the supervisor agent coordinating calendar and email sub-agents."""

    temp = 0.2

    supervisor = create_react_agent(
        model=get_llm(temp),
        tools=[schedule_event, manage_mail, check_staff_info],
        prompt=get_supervisor_prompt(),
        checkpointer=get_checkpointer(),
    )

    return supervisor


supervisor = build_supervisor()


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Calendar & Mail Assistant", page_icon="🤖", layout="wide")
st.title("Calendar & Email Assistant")
body="""
## Schedule events and send emails with AI!

This demo showcases the ability to schedule events and send emails using AI.


### Goal of the Demo:
The idea is to show how to build a multi-agent system using LangGraph where a Supervisor agent can coordinate multiple Sub-Agents to perform complex tasks like scheduling calendar events and sending emails.
- Handling user requests that may involve multiple steps and tools
- Using specialized Sub-Agents for calendar scheduling and email sending (with stubbed tools for demo purposes)
- Use of LangChain/LangGraph for LLM orchestration
- Using Redis and SQLite DB for temporary and persistent storage
- Tracking reasoning steps and usage metrics - per Agent
- Estimating token usage and cost
- Using Streamlit and Render to host the demo
"""
with st.expander('About this demo:', expanded=False):
    st.markdown(body)

# Chat submission (note: using agent.invoke recommended to extract content)
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    thread_id = st.session_state.conversation_thread_id
    callback = UsageMetadataCallbackHandler()
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [callback]}
    inputs = {"messages": st.session_state.chat_history}

    # using invoke to obtain final structured response (astream requires custom handling)
    start = perf_counter()
    supervisor_answer = supervisor.invoke(inputs, config=config)
    st.session_state.latency = perf_counter() - start

    get_workflow(supervisor_answer, 0)
    get_usage(callback.usage_metadata, 0)

    wait_and_flush(timeout=1.0, stable_period=0.05)
    
    # Compute cost
    try:
        usd = compute_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        usd = 0.0
    
    try:
        usd_last = compute_cost(st.session_state.input_tokens_last, st.session_state.output_tokens_last, COST_PER_1K_INPUT, COST_PER_1K_OUTPUT)
    except Exception:
        usd_last = 0.0

    logger.info("SUPERVISOR ANSWER: %s", supervisor_answer)
    logger.info("SUPERVISOR CALBACKS: %s", callback.usage_metadata)

    # Extract text from supervisor_answer (it will be a dict-like structure)
    try:
        # try to extract last AI message text
        msgs = supervisor_answer.get("messages", [])
        final_text = ""
        for m in reversed(msgs):
            # m may be an object-like; try attributes
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if content:
                final_text = content
                break
    except Exception:
        logger.exception("Could not extract final text from agent answer")
        final_text = str(supervisor_answer)


    st.session_state.chat_history.append(AIMessage(content=final_text))

with st.sidebar:
    st.header("Project :green[info]:", divider="rainbow")
    st.markdown(" ")
    st.markdown(" ")
    with st.expander("Tech stack & How to use:"):
        with st.expander("This demo features:"):
            with st.expander("- Supervisor + Sub-Agents workflow"):
                st.markdown("This workflow allows the Supervisor to delegate tasks to Sub-Agents, which can act as specialized tools.")
                st.image("images/supervisor_subagents.png", caption="Supervisor + Sub-Agents workflow")
                st.markdown("send_email and create_calendar_event are stubbed tools, they just give a confirmation once they have correctly been called - means the agents provided all the required info")
                st.markdown("This can easily be changed with real API of Google Calendar, Google Mail or similar")
                st.markdown("The check_staff_info tool is backed by a SQL agent that can read from a SQLite database containing team & availability data")
                st.markdown("The SQL agent is based on a Langchain community toolkit (SQLDatabaseToolkit) for reading the DB schema, listing tables, checking and executing queries")
                st.markdown("The SQL agent is built with a custom prompt that makes it robust in parsing user queries and building correct SQL queries")
                st.markdown("The Calendar and Mail Sub-Agents also use check_staff_info to resolve staff info as needed")
            st.markdown("- LangGraph for LLM orchestration")
            st.markdown("- LangChain React Agents")
            st.markdown("- Thinking process with Intermediate Steps")
            st.markdown("- Tokens usage & cost estimation")
            st.markdown("- Persistent conversation memory (checkpointer)")
            with st.expander("- Usage of Handoff pattern"):
                st.markdown("Handoff is implemented in the Send Email flow: the Mail Agent asks the user a confirmation before it actuallly sends the email, providing recipients, subject and body ")
            with st.expander("- Redis DB for temporary storage"):
                st.markdown("A Redis DB is used to temporary store reasoning steps among Agents and to show them in the sidebar once the chatbot produced its final answer")
                st.markdown("It is also used to temporary store Token Usage metric per Agent")
            st.markdown("- SQLite DB for team & availability data")
            st.markdown("- Streamlit UI")
            st.markdown("- Hosted on Render")
        with st.expander("SQLite DB - :orange[check data]"):
            st.markdown("A SQLite DB is used to store Team Info and Availability - useful to test")
            st.image("images/db_team.png", caption="Team Membership & Emails")
            st.image("images/db_availability.png", caption="Weekly Staff Availability")
        with st.expander(":blue[How to test it?]"):
            st.markdown("Try with the following questions:")
            st.markdown("- (EASY): *Is Daniele Celsa available on Monday at 10?*")
            st.markdown("- (MEDIUM): *Tell me email addresses of Design team*")
            st.markdown("- (MEDIUM): *At what times is Anna Garau available on Monday?*")
            st.markdown("- (MEDIUM): *Send an email to x@y.com informing him the party is canceled, that he is free to have other plans and finish with a short poem of 3 sentences*")
            st.markdown("- (MEDIUM HARD): *Send an email to developer team saying I won't be at office today*")
            st.markdown("- (HARD): *Check if Marco Rossi is available on Friday at 10, then send him an email saying that he is doing a great job*")
            #st.markdown("- (VERY HARD - and tokens consuming): *Schedule a meeting with Marco Rossi and Anna Garau on Tuesday at 9am, make sure they are both available, otherwise suggest alternative times*")
            st.markdown("- (VERY HARD - and tokens consuming): *Create a meeting with Design team on Monday at 17, then notify them with an email*")
            st.markdown("- (VERY HARD - and tokens consuming): *Create a meeting with Developer team on Friday at 10, then notify them with an email*  \n:orange[->] not all developers are available at that time, see how the agent handles it")


    st.markdown("---")
    
    st.markdown(" ")
    with st.expander("Token Usage & Latency:"):
        #st.markdown("#### Token Usage (total and last interaction), Estimated Costs & Latency: ")
        st.metric(label="Latency - Response time (s)", value=f"{st.session_state.latency:.2f}", border=True, label_visibility="visible", help="Time taken by all the agents to produce the LAST response")
        st.metric("Total Tokens Used", f"{st.session_state.total_tokens:.2f}", border=True, label_visibility="visible", help="Total tokens used in the whole thread - included the tokens used by tools")
        st.metric("Estimated total cost (USD)", f"${usd:.5f}", border=True, label_visibility="visible", help="Calculated using %.4f per 1K input tokens and %.4f per 1K output tokens" % (COST_PER_1K_INPUT, COST_PER_1K_OUTPUT))
        with st.expander("Last Interaction - Tokens Breakdown by Agents"):
            st.metric("Supervisor", f"{st.session_state.Supervisor_last_tokens['total_tokens']:.2f}")
            st.metric("Mail Agent", f"{st.session_state.Mail_last_tokens['total_tokens']:.2f}")
            st.metric("Calendar Agent", f"{st.session_state.Calendar_last_tokens['total_tokens']:.2f}")
            st.metric("SQL Agent", f"{st.session_state.SQL_last_tokens['total_tokens']:.2f}")
            st.metric("Total tokens - Last", f"{st.session_state.total_tokens_last:.2f}", border=True, label_visibility="visible", help="Total tokens used in the whole thread plus the tokens used by tools")
            st.metric("Estimated Cost - Last", f"${usd_last:.5f}", border=True, label_visibility="visible", help="Total tokens used in the whole thread plus the tokens used by tools")

    st.markdown("---")
    #st.markdown(" ")

    st.markdown("Agents' Reasoning Steps:")

    if st.session_state.Supervisor_agent_history:
        with st.expander("**Supervisor Agent Thoughts**", expanded=False):
            for msg in st.session_state.Supervisor_agent_history:
                st.write(msg)

    if st.session_state.Calendar_agent_history:
        with st.expander("**Calendar Agent Thoughts**", expanded=False):
            for msg in st.session_state.Calendar_agent_history:
                st.write(msg)

    if st.session_state.Mail_agent_history:
        with st.expander("**Mail Agent Thoughts**", expanded=False):
            for msg in st.session_state.Mail_agent_history:
                st.write(msg)

    if st.session_state.SQL_agent_history:
        with st.expander("**SQL Agent Thoughts**", expanded=False):
            for msg in st.session_state.SQL_agent_history:
                st.write(msg)

    st.markdown("---")
    st.markdown("Developed by [Daniele Celsa](https://www.domenicodanielecelsa.com)")
    st.markdown("Source Code: [GitHub](github.com/domenicodanielecelsa/pdf-researcher)")

# Render chat    
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

