import gc
from typing import Any

import pandas as pd
from tqdm import tqdm

from afml import general
from afml.data_analyst import bars_generator


def load_transactions_and_generate(
    file_path: str | list[str], generator_items: list[dict[str, Any]]
) -> list[pd.DataFrame]:
    _slice_size = int(1e9)
    if isinstance(file_path, str):
        file_names = general.find_files_with_regex(file_path)
    else:
        file_names = file_path
    file_names.sort()
    all_bars: list[list] = [[] for _ in range(len(generator_items))]
    last_bar_transactions = [None for _ in range(len(generator_items))]
    states: list[dict] = [{} for _ in range(len(generator_items))]
    with tqdm(total=len(file_names) * len(generator_items)) as pbar:
        for file_name in file_names:
            transactions = general.load_df(file_name)
            for _slice in range(0, len(transactions), _slice_size):
                slice_transactions = transactions[_slice : _slice + _slice_size]
                for i, bars_kwargs in enumerate(generator_items):
                    if i == 0:
                        pbar.set_description(
                            f"{file_name} - {bars_kwargs['label']}: {bars_kwargs['generator-kwargs']}"
                        )
                        pbar.update(n=1)
                    bars_generator_func = bars_kwargs["generator"]
                    bars_generator_kwargs = bars_kwargs["generator-kwargs"]
                    bars_df, last_bar_transactions[i], states[i] = bars_generator_func(
                        slice_transactions,
                        last_bar_transactions[i],
                        **(bars_generator_kwargs | states[i]),
                    )
                    all_bars[i].append(bars_df)
            del transactions
            gc.collect()

    for i, bars_kwargs in enumerate(generator_items):
        if last_bar_transactions[i] is not None:
            bars_generator_func = bars_kwargs["generator"]
            bars_generator_kwargs = bars_kwargs["generator-kwargs"]
            bars_df, _ = bars_generator_func(
                last_bar_transactions[i],
                None,
                is_last_transactions=True,
                **bars_generator_kwargs,
            )
            all_bars[i].append(bars_df)

    res = []
    for i in range(len(all_bars)):
        if len(all_bars[i]) == 1:
            res.append(all_bars[i][0])
        else:
            res.append(pd.concat(all_bars[i]).sort_index())
    return res


def time_bars_configurations() -> list[dict]:
    kwargs = [
        {"T": 1, "unit": "D"},
        {"T": 4, "unit": "h"},
        {"T": 1, "unit": "h"},
        {"T": 15, "unit": "min"},
    ]
    return [
        {
            "generator": bars_generator.create_time_bars,
            "generator-kwargs": kwarg,
            "label": "time",
        }
        for kwarg in kwargs
    ]


def tick_bars_configurations() -> list[dict]:
    kwargs = [{"T": 10**i} for i in range(4, 8)]
    return [
        {
            "generator": bars_generator.create_tick_bars,
            "generator-kwargs": kwarg,
            "label": "tick",
        }
        for kwarg in kwargs
    ]


def volume_bars_configurations() -> list[dict]:
    kwargs = [{"T": 10**i} for i in range(3, 6)]
    return [
        {
            "generator": bars_generator.create_volume_bars,
            "generator-kwargs": kwarg,
            "label": "volume",
        }
        for kwarg in kwargs
    ]


def dollar_bars_configurations() -> list[dict]:
    kwargs = [{"T": 10**i, "generate_dollar_bars": True} for i in range(5, 9)]
    return [
        {
            "generator": bars_generator.create_volume_bars,
            "generator-kwargs": kwarg,
            "label": "dollar",
        }
        for kwarg in kwargs
    ]


def imbalance_ticks_bars_configurations() -> list[dict]:
    return [
        {
            "generator": bars_generator.create_tick_imbalance_bars,
            "generator-kwargs": {"ticks_ewm_alpha": 0.9},
            "label": "imbalance_ticks",
        },
    ]


def range_bars_configurations() -> list[dict]:
    return [
        {
            "generator": bars_generator.create_range_bars,
            "generator-kwargs": {"T": 500},
            "label": "range",
        },
    ]


if __name__ == "__main__":
    # configurations = imbalance_ticks_bars_configurations()
    configurations = dollar_bars_configurations()
    bars = load_transactions_and_generate(
        # r"/Volumes/Extreme Pro/transactions/BTCUSDT-trades-202[1-9]\-.*\.parquet",
        r"/Volumes/Extreme Pro/transactions/BTCUSDT-trades-.*\-.*\.parquet",
        # [
        #     r"/Volumes/Extreme Pro/transactions/BTCUSDT-trades-2021-01.parquet",
        #     # r"/Volumes/Extreme Pro/transactions/BTCUSDT-trades-2021-02.parquet",
        # ],
        configurations,
    )
    for config, bars_item in zip(configurations, bars):
        bars_item.to_parquet(
            f"../data/{config['label']}-bars-{'-'.join(map(str,config['generator-kwargs'].values()))}.parquet"
        )
