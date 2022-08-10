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

postcode_file = os.sep.join((os.path.dirname(__file__),
                              '..',
                              'resources',
                              'postcodes_unlabelled.csv'))

tool = Tool()

def test_get_flood_class_methods():
    """Check """

    data = tool.get_flood_class_methods()

    assert type(data) is dict

def test_get_flood_class():
    """Check """

    data = tool.get_flood_class(['YO62 4LS', 'LN5 7RW', 'SL6 3BS'])
    data_postcodes = pd.read_csv(postcode_file).postcode
    data2 = tool.get_flood_class(data_postcodes)

    assert type(data) is pd.Series
    assert type(data2) is pd.Series
    assert np.array([(data2 == i).sum() for i in range(1,11)]).sum() == data2.size
    assert np.isclose(data, 1, 1.0e-3).all()



