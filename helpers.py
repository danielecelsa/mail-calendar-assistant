# helpers for app.py

def compute_cost(input_tokens: int, output_tokens: int, price_input_per_1k: float, price_output_per_1k: float) -> float:
    return (input_tokens / 1000.0) * price_input_per_1k + (output_tokens / 1000.0) * price_output_per_1k