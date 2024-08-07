import gc
from functools import wraps
from typing import Literal, Any

import numpy as np
import pandas as pd
from numba import njit

from utils.general import reduce_memory_usage


def bars_generator(func):
    @wraps(func)
    def wrapper(
        transaction_df: pd.DataFrame,
        last_bar_transactions: pd.DataFrame | None,
        is_last_transactions: bool = False,
        *args: Any,
        **kwargs: Any,
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
def create_time_bars(
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
def create_tick_bars(
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
def create_volume_bars(
    transaction_df: pd.DataFrame,
    T: float,
    generate_dollar_bars: bool = False,
) -> pd.DataFrame:
    if generate_dollar_bars:
        cumulative_volume = (transaction_df["price"] * transaction_df["qty"]).cumsum()
    else:
        cumulative_volume = transaction_df["qty"].cumsum()

    # Identify the indices where the cumulative volume reaches the threshold T
    volume_threshold = np.arange(T, cumulative_volume.max(), T)
    volume_bar_indices = np.searchsorted(cumulative_volume.values, volume_threshold)

    previous_index = 0
    chunk = pd.DataFrame()
    n_bars = len(volume_bar_indices)
    open_time = np.empty(n_bars, dtype=np.float64)
    opens = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    closes = np.empty(n_bars, dtype=np.float64)
    volumes = np.empty(n_bars, dtype=np.float64)
    vwaps = np.empty(n_bars, dtype=np.float64)
    for i, index in enumerate(volume_bar_indices):
        if previous_index != index:
            chunk = transaction_df.iloc[previous_index:index]

        open_time[i] = chunk["time"].iloc[-1]
        opens[i] = chunk["price"].iloc[0]
        highs[i] = chunk["price"].max()
        lows[i] = chunk["price"].min()
        closes[i] = chunk["price"].iloc[-1]
        volumes[i] = chunk["qty"].sum()
        vwaps[i] = (chunk["qty"] * chunk["price"]).sum() / chunk["qty"].sum()
        previous_index = index

    dollar_bars_df = pd.DataFrame(
        {
            "time": open_time,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

    return dollar_bars_df


@njit
def _create_bar(
    timestamps: np.ndarray,
    prices: np.ndarray,
    qty: np.ndarray,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    return np.array(
        [
            timestamps[start_index],  # open time
            prices[start_index],  # open
            prices[end_index],  # close
            prices[start_index : end_index+1].max(),  # high
            prices[start_index : end_index+1].min(),  # low
            qty[start_index : end_index+1].sum(),  # volume
            (
                qty[start_index : end_index+1] * prices[start_index : end_index+1]
            ).sum()
            / qty[start_index : end_index+1].sum(),  # vwap
        ]
    )


@njit
def _imbalance_bar_fast(
    timestamps: np.ndarray,
    prices: np.ndarray,
    qty: np.ndarray,
    init_expected_ticks: int,
    ticks_ewm_alpha: float,
    time_ewm_alpha: float,
    mode: int,
) -> np.ndarray:
    """
    create the imbalance bars as chapter 2 algorithm, the arguments names are similar to the algorithm arguments
    timestamps: np.ndarray,
    prices: np.ndarray,
    qty: np.ndarray,
    ewm_window: int,
    ewm_alpha: float,
    mode: int, 0 - ticks, 1 - volume, 2 - dollar
    Returns:
        imbalance <mode> bars
    """
    bucket_size = int(1e7)
    res = np.zeros((7, bucket_size), dtype=np.float32)
    ticks = np.zeros(5 * init_expected_ticks, dtype=np.float32)
    ticks_head = 0
    res_index = 0
    prev_T_star = 0
    prev_bt = 1
    prev_p_t = 0
    theta_T = 0.0
    E_T = init_expected_ticks
    expected_imbalance = float(init_expected_ticks)

    for i in range(len(timestamps)):
        p_t = prices[i]
        q_t = qty[i]
        b_t = prev_bt if prev_p_t == p_t else np.sign(p_t - prev_p_t)

        if mode == 0:
            ticks[ticks_head] = b_t
        elif mode == 1:
            ticks[ticks_head] = b_t * q_t
        elif mode == 2:
            ticks[ticks_head] = b_t * q_t * p_t
        ticks_head += 1
        if ticks_head == len(ticks):
            ticks[:init_expected_ticks] = ticks[-init_expected_ticks:]
            ticks_head = init_expected_ticks

        prev_p_t = p_t
        theta_T += ticks[ticks_head]
        T = i - prev_T_star
        if len(res) == 0 and T >= init_expected_ticks:
            expected_imbalance = ticks[ticks_head - init_expected_ticks : ticks_head].sum()

        if abs(theta_T) >= E_T * abs(expected_imbalance):
            if res_index >= res.shape[1]:
                # Resize the result array size if it's full
                new_res = np.zeros((7, res.shape[1] + bucket_size), dtype=np.float32)
                new_res[:, : res.shape[1]] = res
                res = new_res

            res[:, res_index] = _create_bar(timestamps, prices, qty, prev_T_star, i)
            res_index += 1
            prev_T_star = i+1

            E_T = (1 - time_ewm_alpha) * E_T + time_ewm_alpha * T
            imbalance = ticks[ticks_head - init_expected_ticks : ticks_head].sum()
            expected_imbalance = (1 - time_ewm_alpha) * expected_imbalance + time_ewm_alpha * imbalance
            theta_T = 0

    res[:, res_index] = _create_bar(
        timestamps, prices, qty, prev_T_star, len(timestamps)-1
    )
    res_index += 1
    return res[:, :res_index]


@bars_generator
def create_imbalance_tick_bars(
    transaction_df: pd.DataFrame,
    init_expected_ticks: int = 100,
    ticks_ewm_alpha: float = 0.95,
    time_ewm_alpha: float = 0.8,
) -> pd.DataFrame:

    data = _imbalance_bar_fast(
        transaction_df["time"].values,
        transaction_df["price"].values,
        transaction_df["qty"].values,
        init_expected_ticks,
        ticks_ewm_alpha,
        time_ewm_alpha,
        mode=0,
    )

    res = pd.DataFrame(
        data.T, columns=["time", "open", "close", "high", "low", "volume", "vwap"]
    )
    return res


@bars_generator
def create_imbalance_volume_bars(
    transaction_df: pd.DataFrame,
    init_expected_ticks: int = 100,
    ticks_ewm_alpha: float = 0.95,
    time_ewm_alpha: float = 0.8,
) -> pd.DataFrame:

    data = _imbalance_bar_fast(
        transaction_df["time"].values,
        transaction_df["price"].values,
        transaction_df["qty"].values,
        init_expected_ticks,
        ticks_ewm_alpha,
        time_ewm_alpha,
        mode=1,
    )

    res = pd.DataFrame(
        data.T, columns=["time", "open", "close", "high", "low", "volume", "vwap"]
    )
    return res


@bars_generator
def create_imbalance_dollar_bars(
    transaction_df: pd.DataFrame,
    init_expected_ticks: int = 100,
    ticks_ewm_alpha: float = 0.95,
    time_ewm_alpha: float = 0.8,
) -> pd.DataFrame:

    data = _imbalance_bar_fast(
        transaction_df["time"].values,
        transaction_df["price"].values,
        transaction_df["qty"].values,
        init_expected_ticks,
        ticks_ewm_alpha,
        time_ewm_alpha,
        mode=2,
    )

    res = pd.DataFrame(
        data.T, columns=["time", "open", "close", "high", "low", "volume", "vwap"]
    )
    return res


@njit
def _run_bar_fast(
    timestamps: np.ndarray,
    prices: np.ndarray,
    qty: np.ndarray,
    init_expected_ticks: int,
    volume_ewm_alpha: float,
    time_ewm_alpha: float,
    pbt_ewm_alpha: float,
    mode: int,
) -> np.ndarray:
    """
    create the run bars as chapter 2 algorithm, the arguments names are similar to the algorithm arguments
    timestamps: np.ndarray,
    prices: np.ndarray,
    qty: np.ndarray,
    ewm_window: int,
    ewm_alpha: float,
    mode: int, 0 - ticks, 1 - volume, 2 - dollar
    Returns:
        imbalance <mode> bars
    """
    bucket_size = int(1e7)
    res = np.zeros((7, bucket_size), dtype=np.float32)
    res_index = 0
    prev_T_star = 0
    prev_bt = 1
    prev_p_t = 0
    E_T = init_expected_ticks
    buy_volume = 0.0
    count_buy_sequence = 0.0
    total_volume = 0.0
    P_bt_buy = 0.5
    E_buy_volumes = 1
    E_sell_volumes = 1
    for i in range(len(timestamps)):
        p_t = prices[i]
        q_t = qty[i]
        b_t = prev_bt if prev_p_t == p_t else np.sign(p_t - prev_p_t)

        if mode == 0:
            tick = b_t
        elif mode == 1:
            tick = b_t * q_t
        else:
            tick = b_t * q_t * p_t

        if tick > 0:
            buy_volume += tick
            count_buy_sequence += 1
        total_volume += abs(tick)

        prev_p_t = p_t
        theta_T = max(buy_volume, total_volume - buy_volume)
        T = i - prev_T_star

        if abs(theta_T) >= E_T * max(
            P_bt_buy * E_buy_volumes, (1 - P_bt_buy) * E_sell_volumes
        ):
            if res_index >= res.shape[1]:
                # Resize the result array size if it's full
                new_res = np.zeros((7, res.shape[1] + bucket_size), dtype=np.float32)
                new_res[:, : res.shape[1]] = res
                res = new_res

            res[:, res_index] = _create_bar(timestamps, prices, qty, prev_T_star, i)
            res_index += 1

            P_bt_buy = (1 - pbt_ewm_alpha) * P_bt_buy + pbt_ewm_alpha * (
                count_buy_sequence / (i - prev_T_star)
            )
            E_T = (1 - time_ewm_alpha) * E_T + time_ewm_alpha * T
            if mode > 0:
                E_buy_volumes = (
                    1 - volume_ewm_alpha
                ) * E_buy_volumes + volume_ewm_alpha * buy_volume
                E_sell_volumes = (
                    1 - volume_ewm_alpha
                ) * E_sell_volumes + volume_ewm_alpha * (total_volume - buy_volume)
            buy_volume = 0.0
            count_buy_sequence = 0.0
            total_volume = 0.0
            prev_T_star = i

    res[:, res_index] = _create_bar(
        timestamps, prices, qty, prev_T_star, len(timestamps)-1
    )
    return res[:, :res_index+1]



@bars_generator
def create_ticks_run_bars(
    transaction_df: pd.DataFrame,
    init_expected_ticks: int = 10000,
    ticks_ewm_alpha: float = 0.95,
    time_ewm_alpha: float = 0.8,
    pbt_ewm_alpha: float = 0.8,
) -> pd.DataFrame:

    data = _run_bar_fast(
        transaction_df["time"].values,
        transaction_df["price"].values,
        transaction_df["qty"].values,
        init_expected_ticks,
        ticks_ewm_alpha,
        time_ewm_alpha,
        pbt_ewm_alpha,
        mode=0,
    )

    res = pd.DataFrame(
        data.T, columns=["time", "open", "close", "high", "low", "volume", "vwap"]
    )
    return res


@njit
def _create_range_bars_fast(
    prices: np.ndarray, qtys: np.ndarray, timestamps: np.ndarray, T: float
):
    bars = []
    high = prices[0]
    low = prices[0]
    open_price = prices[0]
    volume = 0
    vwap_sum = 0

    for i in range(len(prices)):
        price = prices[i]
        qty = qtys[i]
        timestamp = timestamps[i]
        high = max(high, price)
        low = min(low, price)
        volume += qty
        vwap_sum += qty * price

        if high - low >= T:
            close_price = price
            bars.append(
                [
                    timestamp,
                    open_price,
                    high,
                    low,
                    close_price,
                    volume,
                    vwap_sum / volume,
                ]
            )
            open_price = price
            high = price
            low = price
            volume = 0
            vwap_sum = 0

    return np.array(bars)


@bars_generator
def create_range_bars(transaction_df: pd.DataFrame, T: float):
    prices = transaction_df["price"].values
    qtys = transaction_df["qty"].values
    timestamps = transaction_df["time"].values
    bars_array = _create_range_bars_fast(prices, qtys, timestamps, T)
    bars_df = pd.DataFrame(
        bars_array, columns=["time", "open", "high", "low", "close", "volume", "vwap"]
    )
    return bars_df
