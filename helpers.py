# helpers for app.py
from langchain_core.callbacks import BaseCallbackHandler, Callbacks
from langchain_core.outputs import LLMResult
from typing import Any, List, Dict, Optional
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessage
from typing import Any, Dict, List, Optional
import uuid
import re
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx

def compute_cost(input_tokens: int, output_tokens: int, cost_per_1k_input: float, cost_per_1k_output: float) -> float:
    """Computes the cost of a given number of input and output tokens."""
    input_cost = (input_tokens / 1000.0) * cost_per_1k_input
    output_cost = (output_tokens / 1000.0) * cost_per_1k_output
    return input_cost + output_cost

def get_user_info(logger) -> dict:
    """
    Tries to get the user's IP address and User-Agent from Streamlit's internal API.
    This is an undocumented feature and is known to change between Streamlit versions.
    Returns a dictionary with 'ip' and 'user_agent'.
    """
    user_info = {"ip": "Unknown", "user_agent": "Unknown"}
    try:
        # Get the context for the current script run
        ctx = get_script_run_ctx()
        if ctx is None:
            logger.warning("Could not get Streamlit script run context.")
            return user_info

        # Get the unique session ID from the context
        session_id = ctx.session_id

        # Get the Streamlit runtime instance
        runtime = get_instance()
        
        # From the runtime, get the session manager, and then the specific session info
        session_info = runtime._session_mgr.get_session_info(session_id)

        if session_info is None:
            logger.warning("Could not find session info for the current session ID.")
            return user_info
        
        # The request headers are located in the 'client' attribute of the session info
        headers = session_info.client.request.headers

        # Logic for extracting IP and User-Agent remains the same
        if 'X-Forwarded-For' in headers:
            user_info['ip'] = headers['X-Forwarded-For'].split(',')[0].strip()
        elif hasattr(session_info.client.request, 'remote_ip'):
            user_info['ip'] = session_info.client.request.remote_ip

        if 'User-Agent' in headers:
            user_info['user_agent'] = headers['User-Agent']

    except Exception as e:
        logger.warning(f"Could not get user info from Streamlit headers: {e}")

    return user_info

class MultiAgentUsageHandler(BaseCallbackHandler):
    """
    Callback handler which tracks token usage per agent.
    This version is robust and correctly handles cases where event parameters
    are None, looking for metadata in multiple places.
    """
    def __init__(self):
        super().__init__()
        # Map to store the agent name for each run_id of a chain/agent
        self.run_id_to_agent_name: Dict[uuid.UUID, str] = {}
        # Final dictionary to accumulate tokens
        self.agent_usage: Dict[str, Dict[str, int]] = {}

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called at the start of a chain/agent. Here we safely capture metadata."""
        try:
            # Method 1: metadata is often passed directly in kwargs
            metadata = kwargs.get("metadata")

            # Method 2 (Fallback): if not in kwargs, check the serialized object (if not None)
            if not metadata and serialized:
                metadata = serialized.get("metadata")

            # Now proceed only if we actually found metadata
            if metadata and "agent_name" in metadata:
                agent_name = metadata["agent_name"]
                # Save the mapping between this run and the agent name
                self.run_id_to_agent_name[run_id] = agent_name
        except Exception as e:
            print(f"ERROR in MultiAgentUsageHandler.on_chain_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Called at the end of an LLM call. Here we attribute tokens."""
        agent_name = "Unknown"
        try:
            # Use the parent_run_id to find the agent name from our map
            if parent_run_id:
                agent_name = self.run_id_to_agent_name.get(parent_run_id, "Unknown")

            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                        token_metadata = gen.message.usage_metadata
                        if token_metadata:
                            #print(f"CALLBACK CAUGHT from '{agent_name}': {token_metadata}")
                            
                            if agent_name not in self.agent_usage:
                                self.agent_usage[agent_name] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                            
                            # Accumulate tokens for the correct agent
                            self.agent_usage[agent_name]["input_tokens"] += token_metadata.get("input_tokens", 0)
                            self.agent_usage[agent_name]["output_tokens"] += token_metadata.get("output_tokens", 0)
                            self.agent_usage[agent_name]["total_tokens"] += token_metadata.get("total_tokens", 0)
        except Exception as e:
            print(f"ERROR in MultiAgentUsageHandler.on_llm_end: {e}")

    def get_final_usage(self) -> Dict[str, Dict[str, int]]:
        """Returns the final dictionary with usage for each agent."""
        return self.agent_usage

# Data structure for a node of the execution tree
class WorkflowNode:
    def __init__(self, run_id: uuid.UUID, name: str, type: str, parent_id: Optional[uuid.UUID] = None):
        self.run_id = run_id
        self.name = name
        self.type = type # "agent" o "tool"
        self.parent_id = parent_id
        self.logs: List[str] = []
        self.children: List['WorkflowNode'] = []
    
    def to_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "logs": self.logs,
            "children": [c.to_dict() for c in self.children]
        }

class MultiAgentWorkflowHandler(BaseCallbackHandler):
    """
    Hierarchical handler that builds a complete execution tree.
    Maintains the parent-child relationship between Agents and Tools.
    """
    def __init__(self):
        super().__init__()
        # Map run_id -> Node for quick access
        self.run_nodes: Dict[uuid.UUID, WorkflowNode] = {}
        # List of root nodes (top-level agents, e.g., Supervisor)
        self.root_nodes: List[WorkflowNode] = []
        print("--- HierarchicalWorkflowHandler INITIALIZED ---")

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            # Determine the name and type
            name = "Unknown Chain"
            node_type = "chain" # generic
            
            # Look for metadata for the agent's name
            metadata = kwargs.get("metadata") or (serialized.get("metadata") if serialized else None)
            
            if metadata and "agent_name" in metadata:
                name = metadata["agent_name"]
                node_type = "agent"
            elif serialized and "name" in serialized:
                name = serialized["name"]
            
            # Create the node
            new_node = WorkflowNode(run_id, name, node_type, parent_run_id)
            self.run_nodes[run_id] = new_node

            # Attach to the tree
            if parent_run_id and parent_run_id in self.run_nodes:
                parent_node = self.run_nodes[parent_run_id]
                parent_node.children.append(new_node)
            else:
                # If it has no known parent, it's a root (or a parent we missed, but we treat it as root)
                self.root_nodes.append(new_node)
                
            # Optional: Capture initial Request if it's an agent
            if node_type == "agent" and inputs and 'messages' in inputs:
                # Logic to extract the last human message as Request
                 if isinstance(inputs['messages'], list) and inputs['messages']:
                    last_msg = inputs['messages'][-1]
                    if hasattr(last_msg, 'content') and last_msg.type == 'human':
                         new_node.logs.append(f"**Request**: {last_msg.content}")

        except Exception as e:
            print(f"ERROR in on_chain_start: {e}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Tools trigger on_tool_start instead of on_chain_start
        try:
            name = serialized.get("name", "Unknown Tool")
            new_node = WorkflowNode(run_id, name, "tool", parent_run_id)
            self.run_nodes[run_id] = new_node
            
            # Add the tool's arguments to its own logs
            new_node.logs.append(f"**Call Tool**: `{name}`")
            new_node.logs.append(f"**Args**: `{input_str}`")

            if parent_run_id and parent_run_id in self.run_nodes:
                self.run_nodes[parent_run_id].children.append(new_node)
            else:
                self.root_nodes.append(new_node)
        except Exception as e:
            print(f"ERROR in on_tool_start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            # LLM logs belong to the node that called the LLM (the Agent)
            if not parent_run_id or parent_run_id not in self.run_nodes:
                return

            current_node = self.run_nodes[parent_run_id]

            for gen_list in response.generations:
                for gen in gen_list:
                    if isinstance(gen.message, AIMessage):
                        msg = gen.message
                        
                        # Capture Thought / Message
                        if msg.content:
                            label = "**Thought**" if msg.tool_calls else f"**{current_node.name} msg**"
                            current_node.logs.append(f"{label}: {msg.content}")

                        # Note: The tool_calls here are redundant if we use on_tool_start,
                        # but we can leave them as textual logs to see them in the thought flow.
                        # For now, I omit them to avoid visual duplicates with Tool child nodes.

        except Exception as e:
            print(f"ERROR in on_llm_end: {e}")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            # Tool's output goes in the tool node itself
            if run_id in self.run_nodes:
                node = self.run_nodes[run_id]
                
                # Clean output (similar to before)
                clean_output = str(output)
                
                # CLEANING LOGIC:
                # We need a regex that captures text inside quotes but ignores escaped quotes (like \')
                if "content=" in clean_output:
                    match = None
                    
                    # 1. Try double quotes first: content="...\"..."
                    # Pattern explanation: ((?:[^"\\]|\\.)*)
                    # Match any character that is NOT a quote/backslash OR match a backslash followed by any char
                    match = re.search(r'content="((?:[^"\\]|\\.)*)"', clean_output)
                    
                    # 2. If not found, try single quotes: content='...\'...'
                    if not match: 
                        match = re.search(r"content='((?:[^'\\]|\\.)*)'", clean_output)
                        
                    if match: 
                        # Extract the captured group
                        raw_content = match.group(1)
                        
                        # Manually unescape Python's repr() formatting
                        clean_output = (raw_content
                                        .replace('\\"', '"')   # Unescape \" -> "
                                        .replace("\\'", "'")   # Unescape \' -> '
                                        .replace("\\n", "\n")  # Unescape \n -> Newline
                                        .replace("\\\\", "\\") # Unescape \\ -> \
                                        )

                node.logs.append(f"**Observation**: {clean_output}")
        except Exception as e:
            print(f"ERROR in on_tool_end: {e}")

    def get_workflow_tree(self) -> List[Dict]:
        """Returns the complete tree as a list of dictionaries."""
        return [node.to_dict() for node in self.root_nodes]

    # We keep this method for compatibility with existing code, 
    # but now we will use get_workflow_tree for the advanced UI.
    def get_final_workflows(self) -> Dict[str, List[str]]:
        return {}