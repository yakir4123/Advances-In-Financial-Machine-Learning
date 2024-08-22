import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightweight_charts import JupyterChart
from matplotlib import pyplot as plt


def plot(bars: pd.DataFrame, lines: pd.DataFrame, markers: list) -> JupyterChart:
    chart = JupyterChart()

    _bars = bars.copy(deep=True)
    _bars["date"] = pd.to_datetime(_bars.index, unit="ms")
    chart.set(_bars)

    lines["date"] = pd.to_datetime(lines.index, unit="ms")
    for column in lines.columns:
        if column == "date":
            continue
        line = chart.create_line(column)
        line.set(lines)

    chart.marker_list(markers)
    return chart


def plot_autocorr(title: str, data: np.ndarray, ax: plt.Axes, lags: int = 1) -> None:
    sm.graphics.tsa.plot_acf(
        data,
        lags=lags,
        ax=ax,
        alpha=0.05,
        unbiased=True,
        fft=True,
        zero=False,
        auto_ylims=True,
        title=title,
    )

    plt.tight_layout()
