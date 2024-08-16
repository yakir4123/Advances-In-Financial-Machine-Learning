import numpy as np
from numba import njit


@njit
def get_t_events(timestamps: np.ndarray, yt: np.ndarray, h: float) -> np.ndarray:
    """
    SNIPPET 2.4 THE SYMMETRIC CUSUM FILTER
    """
    t_events = np.empty(len(timestamps), dtype=timestamps.dtype)
    s_pos = 0.0
    s_neg = 0.0
    diff = np.diff(yt)
    event_count = 0
    for t in range(len(diff)):
        dt = diff[t]
        s_pos = max(0.0, s_pos + dt)
        s_neg = min(0.0, s_neg + dt)
        if s_neg < -h:
            s_neg = 0
            t_events[event_count] = timestamps[t + 1]
            event_count += 1
        elif s_pos > h:
            s_pos = 0
            t_events[event_count] = timestamps[t + 1]
            event_count += 1
    return t_events[:event_count]


@njit
def get_t_events_dynamic_h(
        timestamps: np.ndarray,
        yt: np.ndarray,
        h_timestamps: np.ndarray,
        h_values: np.ndarray,
        h_default: float,
) -> np.ndarray:
    """
    Same as get_t_events but also change h as a function of time,
    """
    t_events = np.empty(len(timestamps), dtype=timestamps.dtype)
    s_pos = 0.0
    s_neg = 0.0
    diff = np.diff(yt)
    event_count = 0
    threshold = h_default
    h_index = 0
    for t in range(len(diff)):
        timestamp = timestamps[t + 1]
        if timestamp > h_timestamps[h_index]:
            threshold = h_values[h_index]
            if h_index < len(h_timestamps):
                h_index += 1

        dt = diff[t]
        s_pos = max(0.0, s_pos + dt)
        s_neg = min(0.0, s_neg + dt)
        if s_neg < -threshold:
            s_neg = 0
            t_events[event_count] = timestamp
            event_count += 1
        elif s_pos > threshold:
            s_pos = 0
            t_events[event_count] = timestamp
            event_count += 1
    return t_events[:event_count]
