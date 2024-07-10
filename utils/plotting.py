import pandas as pd
from lightweight_charts import JupyterChart


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
