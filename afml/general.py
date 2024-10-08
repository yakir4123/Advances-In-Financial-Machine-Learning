import gc
import os
import re
from typing import Any, Callable

import dask.dataframe as dd
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    return reduce_memory_usage(ddf.compute())


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def find_files_with_regex(path: str, recursive: bool = True) -> list[str]:
    """
    Finds all files in the given directory that match the provided regex pattern.

    :param path: The path to a file, accept regex.
    :param recursive: Whether to search subdirectories recursively. Defaults to True.
    :return: A list of file paths that match the regex pattern.
    """
    # Split the directory and pattern
    directory, pattern = path.rsplit("/", 1)
    matching_files = []
    regex = re.compile(pattern)
    for root, _, files in os.walk(directory):
        for file in files:
            if regex.match(file):
                matching_files.append(os.path.join(root, file))
        if not recursive:
            break

    return matching_files


def load_transactions_and_generate(
    file_path: str | list[str],
    bars_generator: Callable,
    progress_bar: bool = True,
    **generator_kwargs: Any,
) -> pd.DataFrame:
    """
    Process CSV transactions files and generate for each file and do it periodically to save memory.

    Parameters:
    file_path: file_path containing the transactions, allow regex-string or list of files.
    bars_generator: method that return the dataframe of the bars

    Returns:
    pd.DataFrame: DataFrame with consolidated bars.
    """
    all_bars = []
    last_bar_transactions = None

    if isinstance(file_path, str):
        file_names = find_files_with_regex(file_path)
    else:
        file_names = file_path
    file_names.sort()
    if progress_bar:
        _tqdm = tqdm
    else:
        _tqdm = lambda x: x
    for file_name in _tqdm(file_names):
        df = load_df(file_name)
        bars_df, last_bar_transactions, _ = bars_generator(
            df, last_bar_transactions, **generator_kwargs
        )
        all_bars.append(bars_df)
        del df
        gc.collect()

    if last_bar_transactions is not None:
        bars_df, _ = bars_generator(
            last_bar_transactions, None, is_last_transactions=True, **generator_kwargs
        )
        all_bars.append(bars_df)

    if len(all_bars) == 1:
        return all_bars[0]
    return pd.concat(all_bars).sort_index()


def returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def get_daily_vol(bars: pd.DataFrame, span0: int = 100) -> pd.Series:
    # daily vol, reindexed to close
    ms_a_day = 24 * 60 * 60 * 1000

    df0 = bars.index.searchsorted(bars.index - ms_a_day)
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        bars.index[df0 - 1], index=bars.index[bars.shape[0] - df0.shape[0] :]
    )
    df0 = (
        bars.close.loc[df0.index] / bars.close.loc[df0.values].values - 1
    )  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0.rename("daily_vol").dropna()
