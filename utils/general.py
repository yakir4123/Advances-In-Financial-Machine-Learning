import gc
import os
from typing import Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd
from numba import njit


def convert_transaction_to_parquet(csv_file_path: str) -> None:
    if not csv_file_path.endswith(".csv"):
        csv_file_path += ".csv"
    parquet_file_path = csv_file_path.replace(".csv", ".parquet")
    if os.path.exists(parquet_file_path):
        return

    df = pd.read_csv(csv_file_path)
    del df["id"]
    del df["quote_qty"]
    del df["is_buyer_maker"]
    gc.collect()
    df = reduce_memory_usage(df)
    df.to_parquet(parquet_file_path)


def load_df(file_path: str) -> pd.DataFrame:
    file_type = file_path.split(".")[-1].lower()
    if file_type == "csv":
        ddf = dd.read_csv(file_path)
    elif file_type == "parquet":
        ddf = dd.read_parquet(file_path)
    else:
        raise ValueError("Unknown file_type")
    return ddf


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


def returns(prices):
    return prices.pct_change().dropna()


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
