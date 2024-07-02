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
    res = np.zeros(len(prices) // 10, dtype=np.int32)
    res_index = 0
    prev_T_star = 0
    T = 1
    sum_v_t = 0
    prev_bt = 1
    prev_p_t = 0
    theta_T = 0
    v_t_given_bt_1 = 0
    while T + prev_T_star < len(prices):
        p_t = prices[prev_T_star + T]
        q_t = qty[prev_T_star + T]
        b_t = prev_bt if prev_p_t == p_t else np.sign(p_t - prev_p_t)
        prev_p_t = p_t

        theta_T += b_t * p_t * q_t
        if res_index == 0:
            E_T = 1.0
        else:
            E_T = res[:res_index].mean()
        # 2v+ - E_v_t can be ewm of bt * v_t
        v_t_given_bt_1 += p_t * q_t if b_t == 1 else 0.0
        # len(bt==1) / len(bt) * sum(vt | bt == 1) / len(bt==1) => sum(vt | bt == 1) / len(bt)
        v_plus = v_t_given_bt_1 / (prev_T_star + T)

        sum_v_t += p_t * q_t
        E_v_t = sum_v_t / (prev_T_star + T)
        if abs(theta_T) >= E_T * abs((2 * v_plus - E_v_t)):
            res[res_index] = prev_T_star + T
            res_index += 1
            prev_T_star += T
            T = 1
            theta_T = 0
        else:
            T += 1
    return res[:res_index]
