import numpy as np
import pandas as pd


def get_daily_vol(bars: pd.DataFrame, span0: int = 100) -> pd.DataFrame:
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


def get_vertical_next_day(bars: pd.DataFrame, t0: pd.Series, ms: int) -> pd.Series:
    t1 = bars.index.searchsorted(t0 + ms)
    t1 = t1[t1 < len(bars)]
    # timestamp of event -> timestamp of closed bar that as a vertical barrier
    return pd.Series(bars.index[t1].values, index=t0[: t1.shape[0]])


def apply_tp_sl(
    close: pd.Series,
    events: pd.DataFrame,
    target: pd.Series,
    tp_scale: float,
    sl_scale: float,
) -> pd.DataFrame:
    """
    Snippet 3.2
    The output from this function is a pandas dataframe containing the timestamps
     (if any) at which each barrier was touched
    """
    out = events[["t1"]].copy(deep=True)
    if tp_scale > 0:
        tp = tp_scale * target
    else:
        tp = pd.Series(index=events.index)  # NaNs
    if sl_scale > 0:
        sl = -sl_scale * target
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        df0 = close.loc[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events.at[loc, "side"]  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "tp"] = df0[df0 > tp[loc]].index.min()  # earliest profit taking.
    return out


def get_vertical_barrier(bars: pd.DataFrame, t0: pd.Series, ms: int) -> pd.Series:
    t1 = bars.index.searchsorted(t0 + ms)
    t1 = t1[t1 < len(bars)]
    # timestamp of event -> timestamp of closed bar that as a vertical barrier
    return pd.Series(bars.index[t1].values, index=t0[: t1.shape[0]])


def get_events_triple_barrier(
    close: pd.Series,
    t0: np.ndarray,
    tp_scale: float,
    sl_scale: float,
    target: pd.Series,
    min_return: float,
    t1: pd.Series | bool = False,
    side: pd.Series = None,
) -> pd.DataFrame:
    """
    Snippet 3.3
    Getting times of the first barrier touch

        Parameters:
            close (pd.Series): close prices of bars
            t0 (np.ndarray): np.ndarray of timestamps that seed every barrier (they can be generated
                                  by CUSUM filter for example)
            tp_scale (float): non-negative float that sets the width of the two barriers (if 0 then no barrier)
            sl_scale (float): non-negative float that sets the width of the two barriers (if 0 then no barrier)
            target (pd.Series): series of targets expressed in terms of absolute returns
            min_return (float): minimum target return required for running a triple barrier search
            numThreads (int): number of threads to use concurrently
            t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                            to disable vertical barriers)
            side (pd.Series) (optional): metalabels containing sides of bets

        Returns:
            events (pd.DataFrame): dataframe with columns:
                - t1: timestamp of the first barrier touch
                - target: target that was used to generate the horizontal barriers
                - side (optional): side of bets
    """
    target = target.reindex(t0, method="ffill")
    if min_return > 0:
        target = target[target > min_return]
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t0)
    if side is None:
        side_ = pd.Series(np.array([1.0] * len(target.index)), index=target.index)
    else:
        side_ = side.loc[target.index.intersection(side.index)]
    events = pd.concat({"t1": t1, "side": side_}, axis=1).dropna()
    df0 = apply_tp_sl(close, events, target, tp_scale, sl_scale)
    events["t1"] = df0.dropna(how="all").min(axis=1)
    events.rename(columns={"t1": "touch"}, inplace=True)
    if side is None:
        events = events.drop("side", axis=1)
    return events


def get_bins(
    events: pd.DataFrame, close: pd.Series, t1: pd.Series | None = None
) -> pd.DataFrame:
    """
    SNIPPET 3.5
    Labeling for side and size
    """
    events_ = events.dropna(subset=["touch"])
    px = events_.index.union(events_["touch"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["touch"].values].values / px.loc[events_.index] - 1
    if "side" in events_:
        out["ret"] *= events_["side"]  # meta-labeling
    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0  # meta-labeling
    out["bin"] = np.sign(out["ret"])
    if t1 is not None:
        out[events_["touch"].isin(t1)] = 0
    return out
