# helpers for app.py
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