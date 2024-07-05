import numpy as np
from numba import njit


@njit
def imbalance_dollar_bar(prices: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """
    create the imbalance dollar bars as chapter 2 algorithm, the arguments names are similar to the algorithm arguments
    timestamps not mentions in the algorithm is the timestamps corresponding to the transactions
    Args:
        b_t: sign of the diff per transaction
        v_t: dollar or volume  per transaction
        timestamps: timestamps corresponding to the transactions
    Returns:
        imbalance dollar bars
    """
    bucket_size = int(1e7)
    res = np.zeros((7, bucket_size), dtype=np.float32)
    res_index = 0
    prev_T_star = 0
    T = 1
    prev_bt = 1
    prev_p_t = 0
    theta_T = 0
    E_T = 1.0
    estimation_v_plus_v_minus = 0
    while T + prev_T_star < len(prices):
        p_t = prices[prev_T_star + T]
        q_t = qty[prev_T_star + T]
        b_t = prev_bt if prev_p_t == p_t else np.sign(p_t - prev_p_t)
        prev_p_t = p_t

        theta_T += b_t * p_t * q_t
        estimation_v_plus_v_minus = (0.9 * estimation_v_plus_v_minus) + 0.1 * (
            b_t * p_t * q_t
        )
        if abs(theta_T) >= E_T * abs(estimation_v_plus_v_minus):
            if res_index >= res.shape[1]:
                # Resize the array if it's full
                new_res = np.zeros((7, res.shape[1] + bucket_size), dtype=np.float32)
                new_res[:, : res.shape[1]] = res
                res = new_res

            res[0, res_index] = prev_T_star + T  # open time
            res[1, res_index] = prices[prev_T_star]  # open
            res[2, res_index] = prices[prev_T_star + T]  # close
            res[3, res_index] = prices[prev_T_star : prev_T_star + T].max()  # high
            res[4, res_index] = prices[prev_T_star : prev_T_star + T].min()  # low
            res[5, res_index] = qty[prev_T_star : prev_T_star + T].sum()  # volume
            res[6, res_index] = (
                qty[prev_T_star : prev_T_star + T]
                * prices[prev_T_star : prev_T_star + T]
            ).sum()  # dollars
            res_index += 1
            prev_T_star += T

            # estimate E_T
            E_T = 0.5 * E_T + 0.5 * T
            T = 1
            theta_T = 0
        else:
            T += 1
    return res[:, :res_index]


@njit
def getTEvents(timestamps: np.ndarray, yt: np.ndarray, h: float) -> np.ndarray:
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
