import numpy as np
import pandas as pd


def apply_tp_sl(close: pd.Series, events, target, tp_scale, sl_scale, molecule):
    """
    Snippet 3.2
    """
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    if tp_scale > 0:
        tp = tp_scale * target
    else:
        tp = pd.Series(index=events.index)  # NaNs
    if sl_scale > 0:
        sl = -sl_scale * target
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        df0 = close.loc[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, "side"]  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "tp"] = df0[df0 > tp[loc]].index.min()  # earliest profit taking.
    return out


# including metalabeleing possibility
def get_events_tripple_barrier(
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
            target (pd.Series): s series of targets expressed in terms of absolute returns
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
    df0 = apply_tp_sl(close, events, target, tp_scale, sl_scale, events.index)
    events["t1"] = df0.dropna(how="all").min(axis=1)
    events.rename(columns={"t1": "touch"}, inplace=True)
    if side is None:
        events = events.drop("side", axis=1)
    return events
