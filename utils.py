import gc

import numpy as np
import pandas as pd
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


# the data is downloaded im using is crypto ticker data from
# https://www.binance.com/en/landing/data
# using BTC for this exercise download data of 5 months
def load_df(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    del df["id"]
    del df["quote_qty"]
    del df["is_buyer_maker"]
    gc.collect()
    return reduce_memory_usage(df)


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_float_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast="float")
            elif pd.api.types.is_integer_dtype(col_type):
                if c_min >= 0:
                    if c_max < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
        else:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {(start_mem - end_mem) / start_mem * 100:.1f}%")
    return df
