"""Module to check that your flood tool will be scorable."""
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import flood_tool
import numpy as np
import pandas as pd

tool = flood_tool.Tool()
POSTCODES = ['YO62 4LS', 'DE2 3DA']

PRICE_METHODS = tool.get_house_price_methods()
CLASS_METHODS = tool.get_flood_class_methods()


def test_get_easting_northing():
    """Check return type."""
    data = tool.get_easting_northing(POSTCODES)
    assert issubclass(type(data), pd.DataFrame)


def test_get_lat_long():
    """Check return type."""
    data = tool.get_lat_long(POSTCODES)
    assert issubclass(type(data), pd.DataFrame)


def test_get_flood_class():
    """Check return types."""
    for method in CLASS_METHODS.values():
        data = tool.get_flood_class(POSTCODES, method)
        assert issubclass(type(data), pd.Series)


def test_get_median_house_price_estimate():
    """Check return type."""
    for method in PRICE_METHODS.values():
        data = tool.get_median_house_price_estimate(POSTCODES, method)
        assert issubclass(type(data), pd.Series)


SCORES = np.array([[100, 80, 60, 60, 30, 0, -30, -600, -1800, -2400],
                   [80, 100, 80, 90, 60, 30, 0, -300, -1200, -1800],
                   [60, 80, 100, 120, 90, 60, 30, 0,  -600, -1200],
                   [40, 60, 80,  0, 120, 90, 60, 300, 0, -600],
                   [20, 40, 60, 120, 150, 120, 90, 600, 600, 0],
                   [0, 20, 40, 90, 120, 150, 120, 900, 1200, 600],
                   [-20, 0, 20, 60, 90, 120, 150, 1200, 1800, 1200],
                   [-40, -20, 0, 30, 60, 90, 120, 1500, 2400, 1800],
                   [-60, -40, -20, 0, 30, 60, 90, 1200, 3000, 2400],
                   [-80, -60, -40, -30, 0, 30, 60, 900, 2400, 3000]])


def score_prediction(predicted, truth):
    return sum([SCORES[_p-1, _t-1]
                for _p, _t in zip(predicted, truth)])
