"""Interactions with rainfall and river data."""

import numpy as np
import pandas as pd

__all__ = ["get_station_data"]


def get_station_data(filename, station_reference):
    """Return readings for a specified recording station from .csv file.

    Parameters
    ----------

    filename: str
        filename to read
    station_reference
        station_reference to return.

    >>> data = get_station_data('resources/wet_day.csv)
    """
    frame = pd.read_csv(filename)
    frame = frame.loc[frame.stationReference == station_reference]

    return pd.to_numeric(frame.value.values)
