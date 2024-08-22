from typing import Any

import numpy as np
import pandas as pd

from afml.data_analyst.bars_generator import bars_generator, create_time_bars


@bars_generator
def mock_process_transactions(
    transaction_df: pd.DataFrame, *args: Any, **kwargs: Any
) -> tuple[pd.DataFrame, dict]:
    # Mock implementation, returns DataFrame with a "time" index
    transaction_df["time"] = pd.to_datetime(transaction_df["time"], unit="ms")
    return transaction_df, {}


def test_basic_functionality() -> None:
    # Test without last_bar_transactions
    data = {
        "time": pd.date_range(start="2023-01-01", periods=5, freq="T").astype(np.int64)
        // 10**6,
        "value": range(5),
    }
    df = pd.DataFrame(data)

    result, last_bar_transactions, state = mock_process_transactions(df)

    assert len(result) == 4
    assert state == "mock_state"
    assert last_bar_transactions is not None
    assert result.index[-1] < last_bar_transactions["time"].iloc[0]


def test_with_last_bar_transactions() -> None:
    # Test with last_bar_transactions
    data1 = {
        "time": pd.date_range(start="2023-01-01", periods=3, freq="T").astype(np.int64)
        // 10**6,
        "value": range(3),
    }
    data2 = {
        "time": pd.date_range(start="2023-01-01 00:03", periods=3, freq="T").astype(
            np.int64
        )
        // 10**6,
        "value": range(3, 6),
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    result, last_bar_transactions, state = mock_process_transactions(
        df2, last_bar_transactions=df1
    )

    assert len(result) == 5
    assert state == "mock_state"
    assert last_bar_transactions is not None
    assert result.index[-1] < last_bar_transactions["time"].iloc[0]


def test_non_monotonic_time_data() -> None:
    # Test with non-monotonic time data
    data = {
        "time": [
            pd.Timestamp("2023-01-01 00:02:00").value // 10**6,
            pd.Timestamp("2023-01-01 00:01:00").value // 10**6,
            pd.Timestamp("2023-01-01 00:03:00").value // 10**6,
        ],
        "value": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    result, last_bar_transactions, state = mock_process_transactions(df)

    assert len(result) == 2
    assert state == "mock_state"
    assert result.index.is_monotonic_increasing
    assert last_bar_transactions is not None
    assert result.index[-1] < last_bar_transactions["time"].iloc[0]


def test_is_last_transactions_flag() -> None:
    # Test with is_last_transactions=True
    data = {
        "time": pd.date_range(start="2023-01-01", periods=5, freq="T").astype(np.int64)
        // 10**6,
        "value": range(5),
    }
    df = pd.DataFrame(data)

    result, last_bar_transactions, state = mock_process_transactions(
        df, is_last_transactions=True
    )

    assert len(result) == 5
    assert len(last_bar_transactions) == 1


def test_time_bars_basic_functionality() -> None:
    data = {
        "time": pd.date_range(start="2023-01-01", periods=10, freq="T").astype(np.int64)
        // 10**6,
        "price": np.random.random(10) * 100,
        "qty": np.random.randint(1, 10, size=10),
    }
    df = pd.DataFrame(data)

    result, last_bar_transactions, state = create_time_bars(
        df, is_last_transactions=True, T=5, unit="min"
    )

    assert len(result) == 2  # We expect 2 bars of 5 minutes each
    assert result.columns.tolist() == ["open", "high", "low", "close", "volume", "vwap"]
    assert result["volume"].sum() == df["qty"].sum()


def test_time_bars_handling_different_units() -> None:
    data = {
        "time": pd.date_range(start="2023-01-01", periods=120, freq="T").astype(
            np.int64
        )
        // 10**6,
        "price": np.random.random(120) * 100,
        "qty": np.random.randint(1, 10, size=120),
    }
    df = pd.DataFrame(data)

    # Test hourly bars
    result, last_bar_transactions, state = create_time_bars(df, T=1, unit="H")
    assert len(result) == 2  # We expect 2 hourly bars

    # Test daily bars
    result, last_bar_transactions, state = create_time_bars(df, T=1, unit="D")
    assert len(result) == 1  # We expect 1 daily bar

    # Test weekly bars
    result, last_bar_transactions, state = create_time_bars(df, T=1, unit="W")
    assert len(result) == 1  # We expect 1 weekly bar


def test_time_bars_empty_dataframe() -> None:
    df = pd.DataFrame(columns=["time", "price", "qty"])

    result, last_bar_transactions, state = create_time_bars(df, T=5, unit="min")

    assert result.empty
    assert last_bar_transactions.empty


def test_time_bars_single_row_dataframe() -> None:
    data = {
        "time": [pd.Timestamp("2023-01-01 00:00:00").value // 10**6],
        "price": [100.0],
        "qty": [5],
    }
    df = pd.DataFrame(data)

    result, last_bar_transactions, state = create_time_bars(df, T=5, unit="min")

    assert len(result) == 1
    assert result.iloc[0]["open"] == 100.0
    assert result.iloc[0]["volume"] == 5
    assert last_bar_transactions.empty
