"""Test Module."""


# from flood_tool import *
import numpy as np

from pytest import mark
import sys
import os
import inspect
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tool import *
tool = Tool()

def test_get_house_price_methods():
    """Check """

    data = tool.get_house_price_methods()

    assert type(data) is dict

def test_get_flood_class():
    """Check """

    data = tool.get_median_house_price_estimate(['YO62 4LS', 'LN5 7RW', 'SL6 3BS'])
    assert type(data) is pd.Series
    assert np.isclose(data, [772000.000000, 117333.333333, 598250.000000], 1.0e-3).all()


data = tool.get_median_house_price_estimate(['YO62 4LS', 'LN5 7RW', 'SL6 3BS'])

print(data)