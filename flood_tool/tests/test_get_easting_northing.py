"""Test Module."""

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from tool import *
import numpy as np

from pytest import mark

tool = Tool()


def test_get_easting_northing():
    """Test get_easting_northing function"""
    east = tool.get_easting_northing('YO62 4LS')['easting']
    north = tool.get_easting_northing('YO62 4LS')['northing']
    
    if east is NotImplemented:
        assert False

    assert np.isclose(east[0], 467631.0).all()
    assert np.isclose(north[0], 472825.0).all()

if __name__ == "__main__":
    test_get_easting_northing()
