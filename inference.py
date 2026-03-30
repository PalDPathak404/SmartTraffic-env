"""
inference.py  –  OpenEnv-required agent inference entry point
=============================================================
The OpenEnv checker looks for this file at the repository root.
It must expose a callable `get_action(observation: dict) -> int`
that returns one of the valid discrete actions for the environment:
    0  – All Red  (safety pause between signal switches)
    1  – Green North-South
    2  – Green East-West
"""

from src.models import State
from src.agent import DeterministicAgent

# ── Module-level singleton so the agent retains its internal timing state
# across sequential calls in the same session.
_agent = DeterministicAgent()


def get_action(observation: dict) -> int:
    """
    Given a raw observation dictionary (as returned by POST /reset or POST /step),
    return an integer action index.

    Parameters
    ----------
    observation : dict
        Keys match the environment's observation space defined in openenv.yaml:
            north_queue, south_queue, east_queue, west_queue,
            current_signal, waiting_time_total,
            emergency_vehicle_present, time_step,
            ns_growth, ew_growth, emergency_direction,
            ns_wait_time, ew_wait_time

    Returns
    -------
    int  –  0 (All Red) | 1 (Green NS) | 2 (Green EW)
    """
    state = State(
        north_queue=int(observation.get("north_queue", 0)),
        south_queue=int(observation.get("south_queue", 0)),
        east_queue=int(observation.get("east_queue", 0)),
        west_queue=int(observation.get("west_queue", 0)),
        current_signal=str(observation.get("current_signal", "red")),
        waiting_time_total=float(observation.get("waiting_time_total", 0.0)),
        emergency_vehicle_present=bool(observation.get("emergency_vehicle_present", False)),
        time_step=int(observation.get("time_step", 0)),
        ns_growth=float(observation.get("ns_growth", 0.0)),
        ew_growth=float(observation.get("ew_growth", 0.0)),
        emergency_direction=str(observation.get("emergency_direction", "none")),
        ns_wait_time=float(observation.get("ns_wait_time", 0.0)),
        ew_wait_time=float(observation.get("ew_wait_time", 0.0)),
    )
    return _agent.get_action(state)


def reset_agent():
    """Reset the agent's internal timing state (call when the environment resets)."""
    global _agent
    _agent = DeterministicAgent()


# ── Allow quick local testing: python inference.py
if __name__ == "__main__":
    sample_obs = {
        "north_queue": 10,
        "south_queue": 5,
        "east_queue": 2,
        "west_queue": 3,
        "current_signal": "red",
        "waiting_time_total": 0.0,
        "emergency_vehicle_present": False,
        "time_step": 0,
        "ns_growth": 0.0,
        "ew_growth": 0.0,
        "emergency_direction": "none",
        "ns_wait_time": 0.0,
        "ew_wait_time": 0.0,
    }
    action = get_action(sample_obs)
    action_names = {0: "All Red", 1: "Green NS", 2: "Green EW"}
    print(f"Sample observation → action: {action} ({action_names[action]})")
