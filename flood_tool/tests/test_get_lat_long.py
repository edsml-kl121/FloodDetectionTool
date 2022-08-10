"""Test Module."""

import sys
import sys
import os
import inspect
# sys.path.insert(0, '/Users/jc6821/Desktop/ads-deluge-wye/')

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# import flood_tool
from tool import *
import numpy as np

from pytest import mark

tool = Tool()


def test_get_lat_long():
    """Test get_lat_long function"""
    lat = tool.get_lat_long('YO62 4LS')['lat']
    lon = tool.get_lat_long('YO62 4LS')['lon']
    
    if lat is NotImplemented:
        assert False

    assert np.isclose(lat[0], 54.1466).all()
    assert np.isclose(lon[0], -0.96449).all()

if __name__ == "__main__":
    test_get_lat_long()
