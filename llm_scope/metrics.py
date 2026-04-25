"""
llm-scope metrics — Phase 4: Metric calculation helpers.
"""


def calc_jitter(timestamps: list[float]) -> float:
    """
    Calculate the standard deviation of chunk intervals (ms).
    Returns 0.0 if fewer than 3 timestamps (not enough data).
    """
    if len(timestamps) < 3:
        return 0.0
    intervals = [
        (timestamps[i] - timestamps[i - 1]) * 1000
        for i in range(1, len(timestamps))
    ]
    mean = sum(intervals) / len(intervals)
    variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    return variance ** 0.5


def format_cost(cost_usd: float) -> str:
    """
    Format a USD cost value for display.
    Returns '$0.00' for negligible values, otherwise '$x.xxxxxx'.
    """
    if cost_usd < 0.000001:
        return "$0.00"
    return f"${cost_usd:.6f}"


def calc_tps(tokens: int, duration_ms: float) -> float:
    """
    Calculate tokens per second.
    Uses max(1.0, duration_ms) to avoid division by zero for ultra-fast responses.
    """
    safe_duration_s = max(1.0, duration_ms) / 1000.0
    return tokens / safe_duration_s
