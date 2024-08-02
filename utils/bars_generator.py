import gc
import os
from functools import wraps
from typing import Literal, Any

import numpy as np
import pandas as pd
from numba import njit

from dataclasses import dataclass, field
from utils.general import reduce_memory_usage


@dataclass
class Bars:
    time: list[float] = field(default_factory=list)
    open: list[float] = field(default_factory=list)
    high: list[float] = field(default_factory=list)
    low: list[float] = field(default_factory=list)
    close: list[float] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    vwap: list[float] = field(default_factory=list)

    def add_bar_params(self, transactions: pd.DataFrame) -> None:
        self.time.append(transactions["time"].iloc[-1])
        self.open.append(transactions["price"].iloc[0])
        self.high.append(transactions["price"].max())
        self.low.append(transactions["price"].min())
        self.close.append(transactions["price"].iloc[-1])
        self.volume.append(transactions["qty"].sum())
        self.vwap.append(
            (transactions["qty"] * transactions["price"]).sum()
            / transactions["qty"].sum()
        )

    def to_df(self):
        df = pd.DataFrame(
            {
                "time": self.time,
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "vwap": self.vwap,
            }
        )
        # in case there are bars with the same time (may happened with bad arguments)
        df["price_volume"] = df["close"] * df["volume"]
        df = (
            df.groupby("time")
            .agg(
                {
                    "open": "first",
                    "close": "last",
                    "high": "max",
                    "low": "min",
                    "volume": "sum",
                    "price_volume": "sum",
                }
            )
            .reset_index()
        )
        df["vwap"] = df["price_volume"] / df["volume"]
        df.drop(columns=["price_volume"], inplace=True)
        return df


def create_time_bars(transaction_df, T) -> tuple[pd.DataFrame, None]:
    # Convert epoch timestamp to datetime
    transaction_df["dt_time"] = pd.to_datetime(transaction_df["time"], unit="ms")
    transaction_df.set_index("dt_time", inplace=True)

    # Resample the dataframe to create bars for every T minutes
    if T[-1] == "m":
        T = T[:-1] + "min"
    resampled_df = transaction_df.resample(T).agg(
        {"price": ["first", "max", "min", "last"], "qty": "sum"}
    )

    resampled_df.columns = ["open", "high", "low", "close", "volume"]
    resampled_df.dropna(inplace=True)
    resampled_df.reset_index(inplace=True)
    resampled_df["time"] = resampled_df["dt_time"]
    del resampled_df["dt_time"]
    gc.collect()

    res = reduce_memory_usage(resampled_df).set_index("time")
    return res, None


def create_tick_bars(
    transaction_df: pd.DataFrame, T: float
) -> tuple[pd.DataFrame, None]:
    transaction_df = transaction_df.sort_values(by="time").reset_index(drop=True)
    n_bars = len(transaction_df) // T
    transactions = transaction_df.iloc[: n_bars * T]
    chunks = transactions.values.reshape(n_bars, T, -1)
    times = chunks[:, :, transaction_df.columns.get_loc("time")]
    prices = chunks[:, :, transaction_df.columns.get_loc("price")]
    qtys = chunks[:, :, transaction_df.columns.get_loc("qty")]

    times_end = times[:, -1]
    opens = prices[:, 0]
    highs = prices.max(axis=1)
    lows = prices.min(axis=1)
    closes = prices[:, -1]
    volumes = qtys.sum(axis=1)

    tick_bars_df = pd.DataFrame(
        {
            "time": times_end,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    res = reduce_memory_usage(tick_bars_df).set_index("time")
    return res


def create_volume_bars(transaction_df: pd.DataFrame, T: float) -> pd.DataFrame:
    transaction_df = transaction_df.sort_values(by="time").reset_index(drop=True)
    transaction_df["cumulative_volume"] = transaction_df["qty"].cumsum()
    # Identify the indices where the cumulative volume reaches the threshold T
    volume_thresholds = np.arange(T, transaction_df["cumulative_volume"].max(), T)
    volume_bar_indices = np.searchsorted(
        transaction_df["cumulative_volume"].values, volume_thresholds
    )

    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    previous_index = 0
    for index in volume_bar_indices:
        if previous_index != index:
            chunk = transaction_df.iloc[previous_index:index]
        times.append(chunk["time"].iloc[-1])
        opens.append(chunk["price"].iloc[0])
        highs.append(chunk["price"].max())
        lows.append(chunk["price"].min())
        closes.append(chunk["price"].iloc[-1])
        volumes.append(chunk["qty"].sum())
        previous_index = index

    volume_bars_df = pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    res = reduce_memory_usage(volume_bars_df).set_index("time")
    return res


def create_dollar_bars(transaction_df: pd.DataFrame, T: float) -> pd.DataFrame:
    transaction_df = transaction_df.sort_values(by="time").reset_index(drop=True)
    transaction_df["cumulative_dollar"] = (
        transaction_df["price"] * transaction_df["qty"]
    ).cumsum()
    # Identify the indices where the cumulative volume reaches the threshold T
    volume_thresholds = np.arange(T, transaction_df["cumulative_dollar"].max(), T)
    volume_bar_indices = np.searchsorted(
        transaction_df["cumulative_dollar"].values, volume_thresholds
    )

    times = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    previous_index = 0
    for index in volume_bar_indices:
        if previous_index != index:
            chunk = transaction_df.iloc[previous_index:index]
        times.append(chunk["time"].iloc[-1])
        opens.append(chunk["price"].iloc[0])
        highs.append(chunk["price"].max())
        lows.append(chunk["price"].min())
        closes.append(chunk["price"].iloc[-1])
        volumes.append(chunk["qty"].sum())
        previous_index = index

    dollar_bars_df = pd.DataFrame(
        {
            "time": times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    dollar_bars_df = (
        dollar_bars_df.groupby("time")
        .agg(
            {
                "open": "first",
                "close": "last",
                "high": "max",
                "low": "min",
                "volume": "sum",
            }
        )
        .reset_index()
    )

    res = reduce_memory_usage(dollar_bars_df).set_index("time")
    return res


@njit
def _imbalance_dollar_bar_fast(prices: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """
    create the imbalance dollar bars as chapter 2 algorithm, the arguments names are similar to the algorithm arguments
    timestamps not mentions in the algorithm is the timestamps corresponding to the transactions
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


def create_imbalance_dollar_bar(transaction_df: pd.DataFrame) -> pd.DataFrame:
    data = _imbalance_dollar_bar_fast(
        transaction_df["price"].values, transaction_df["qty"].values
    )

    res = pd.DataFrame(
        data.T, columns=["time", "open", "close", "high", "low", "volume", "dollars"]
    )
    res = reduce_memory_usage(res).set_index("time")
    return res


@njit
def generate_range_bars(
    prices: np.ndarray, qtys: np.ndarray, timestamps: np.ndarray, T: float
):
    bars = []
    high = prices[0]
    low = prices[0]
    open_price = prices[0]
    volume = 0

    for i in range(len(prices)):
        price = prices[i]
        qty = qtys[i]
        timestamp = timestamps[i]
        high = max(high, price)
        low = min(low, price)
        volume += qty

        if high - low >= T:
            close_price = price
            bars.append([timestamp, open_price, high, low, close_price, volume])
            open_price = price
            high = price
            low = price
            volume = 0

    return np.array(bars)


def create_range_bars(df: pd.DataFrame, T: float):
    prices = df["price"].values
    qtys = df["qty"].values
    timestamps = df["time"].values
    bars_array = generate_range_bars(prices, qtys, timestamps, T)
    bars_df = pd.DataFrame(
        bars_array, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    bars_df.set_index("timestamp", inplace=True)
    return bars_df


def bars_generator(func):
    @wraps(func)
    def wrapper(
        transaction_df: pd.DataFrame,
        last_bar_transactions: pd.DataFrame | None,
        is_last_transactions: bool = False,
        *args: Any,
        **kwargs: Any
    ):
        if last_bar_transactions is not None:
            transaction_df = pd.concat(
                [last_bar_transactions, transaction_df], ignore_index=True
            )
        if not transaction_df["time"].is_monotonic_increasing:
            # sort if It's not already sorted
            transaction_df = transaction_df.sort_values(by="time")
        transaction_df.reset_index(drop=True)

        result = func(transaction_df, *args, **kwargs)

        result = reduce_memory_usage(result).set_index("time")
        if is_last_transactions:
            return result, None

        # faster way than transaction_df[transaction_df['time'] >= result.index[-1]]
        idx = np.searchsorted(
            transaction_df["time"],
            result.index.values[-1],
            side="left",
        )
        last_bar_transactions = transaction_df.iloc[idx:]
        return result.iloc[:-1], last_bar_transactions

    return wrapper


@bars_generator
def create_time_bars__rolling(
    transaction_df: pd.DataFrame,
    T: int,
    unit: Literal["min", "H", "D", "W"],
) -> pd.DataFrame:
    transaction_df["dt_time"] = pd.to_datetime(transaction_df["time"], unit="ms")
    transaction_df.set_index("dt_time", inplace=True)

    # Resample the dataframe to create bars for every T minutes
    resampled_df = transaction_df.resample(str(T) + unit).agg(
        {"price": ["first", "max", "min", "last"], "qty": "sum"}
    )

    resampled_df.columns = ["open", "high", "low", "close", "volume"]
    resampled_df.dropna(inplace=True)
    resampled_df["vwap"] = (transaction_df["price"] * transaction_df["qty"]).resample(
        str(T) + unit
    ).sum() / transaction_df["qty"].resample(str(T) + unit).sum()

    resampled_df.reset_index(inplace=True)
    resampled_df["time"] = resampled_df["dt_time"].values.astype("uint64") // 10**6
    del resampled_df["dt_time"]
    gc.collect()
    return resampled_df


@njit
def _create_tick_bars_fast(
    time_values: np.ndarray,
    price_values: np.ndarray,
    qty_values: np.ndarray,
    T: int,
):
    n_bars = len(time_values) // T
    remainder = len(time_values) % T
    open_time = np.empty(n_bars + (1 if remainder else 0), dtype=time_values.dtype)
    opens = np.empty(n_bars + (1 if remainder else 0), dtype=price_values.dtype)
    highs = np.empty(n_bars + (1 if remainder else 0), dtype=price_values.dtype)
    lows = np.empty(n_bars + (1 if remainder else 0), dtype=price_values.dtype)
    closes = np.empty(n_bars + (1 if remainder else 0), dtype=price_values.dtype)
    volumes = np.empty(n_bars + (1 if remainder else 0), dtype=qty_values.dtype)
    vwaps = np.empty(n_bars + (1 if remainder else 0), dtype=np.float64)

    # Process each full bar
    for i in range(n_bars):
        start_idx = i * T
        end_idx = start_idx + T
        open_time[i] = time_values[start_idx]
        opens[i] = price_values[start_idx]
        highs[i] = np.max(price_values[start_idx:end_idx])
        lows[i] = np.min(price_values[start_idx:end_idx])
        closes[i] = price_values[end_idx - 1]
        volumes[i] = np.sum(qty_values[start_idx:end_idx])
        vwaps[i] = (
            np.sum(price_values[start_idx:end_idx] * qty_values[start_idx:end_idx])
            / volumes[i]
        )

    # Process the remainder bar if there are remaining ticks
    if remainder > 0:
        start_idx = n_bars * T
        open_time[n_bars] = time_values[start_idx]
        opens[n_bars] = price_values[start_idx]
        highs[n_bars] = np.max(price_values[start_idx:])
        lows[n_bars] = np.min(price_values[start_idx:])
        closes[n_bars] = price_values[-1]
        volumes[n_bars] = np.sum(qty_values[start_idx:])
        vwaps[n_bars] = (
            np.sum(price_values[start_idx:] * qty_values[start_idx:]) / volumes[n_bars]
        )

    return open_time, opens, highs, lows, closes, volumes, vwaps


@bars_generator
def create_tick_bars__rolling(
    transaction_df: pd.DataFrame,
    T: int,
) -> pd.DataFrame:

    # Calculate bars using Numba-accelerated function
    open_time, opens, highs, lows, closes, volumes, vwaps = _create_tick_bars_fast(
        transaction_df["time"].values,
        transaction_df["price"].values,
        transaction_df["qty"].values,
        T,
    )

    # Create DataFrame for the bars
    bars = pd.DataFrame(
        {
            "time": open_time,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "vwap": vwaps,
        }
    )

    return bars

@bars_generator
def create_volume_bars__rolling(
    transaction_df: pd.DataFrame,
    T: float,
    generate_dollar_bars: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if generate_dollar_bars:
        cumulative_volume = (transaction_df["price"] * transaction_df["qty"]).cumsum()
    else:
        cumulative_volume = transaction_df["qty"].cumsum()

    # Identify the indices where the cumulative volume reaches the threshold T
    volume_threshold = np.arange(T, cumulative_volume.max(), T)
    volume_bar_indices = np.searchsorted(cumulative_volume.values, volume_threshold)
    bars = Bars()

    previous_index = 0
    chunk = pd.DataFrame()
    for index in volume_bar_indices[:-1]:
        if previous_index != index:
            chunk = transaction_df.iloc[previous_index:index]
        bars.add_bar_params(chunk)
        previous_index = index

    last_bar_transactions = transaction_df.iloc[volume_bar_indices[-1] :]
    last_bar = (last_bar_transactions["price"] * last_bar_transactions["qty"]).sum()
    assert last_bar < T

    volume_bars_df = bars.to_df()
    res = reduce_memory_usage(volume_bars_df).set_index("time")
    return res, last_bar_transactions


def create_imbalance_dollar_bar__rolling(
    transaction_df: pd.DataFrame,
    last_bar_transactions: pd.DataFrame | None,
    is_last_transactions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if last_bar_transactions is not None:
        transaction_df = pd.concat(
            [last_bar_transactions, transaction_df], ignore_index=True
        )
    transaction_df = transaction_df.sort_values(by="time").reset_index(drop=True)

    data = _imbalance_dollar_bar_fast(
        transaction_df["price"].values, transaction_df["qty"].values
    )

    res = pd.DataFrame(
        data.T, columns=["time", "open", "close", "high", "low", "volume", "dollars"]
    )
    res = reduce_memory_usage(res).set_index("time")
    if is_last_transactions:
        return res, None

    last_bar_transactions = transaction_df.iloc[res.index[-1] :]
    return res.iloc[:-1], last_bar_transactions


"""
i have several functions with this functionality as this example below
at the beginning and the end, i wonder how can i simply decorate the functions with it instead of repeat it every time, the arguments
 `last_bar_transactions` and `is_last_transactions` needed only for these beginning and ending of the code
def example(
    transaction_df: pd.DataFrame,
    last_bar_transactions: pd.DataFrame | None,
    is_last_transactions: bool = False,
    <uniquely args for each function>
):
    if last_bar_transactions is not None:
        transaction_df = pd.concat(
            [last_bar_transactions, transaction_df], ignore_index=True
        )
    transaction_df = transaction_df.sort_values(by="time").reset_index(drop=True)
    date = <unique functionality>
    res = pd.DataFrame(
        data.T, columns=["time", "open", "close", "high", "low", "volume", "dollars"]
    )
    res = reduce_memory_usage(res).set_index("time")
    if is_last_transactions:
        return res, None

    last_bar_transactions = transaction_df.iloc[res.index[-1] :]
    return res.iloc[:-1], last_bar_transactions
"""
