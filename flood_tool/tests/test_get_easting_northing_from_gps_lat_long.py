"""Test Module."""

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from geo import *
# from flood_tool.geo import get_easting_northing_from_gps_lat_long
import numpy as np
from pytest import mark


def test_get_easting_northing_from_gps_lat_long():
    """Check get_easting_northing_from_gps_lat_long"""

    data1,data2 = get_easting_northing_from_gps_lat_long([55.5], [-1.54])

    if data1 is NotImplemented:
        assert False
    #It passes when they are close
    assert np.isclose(data1[0], 429157.0).all()
    assert np.isclose(data2[0], 623009.0).all()
    
    
if __name__ == "__main__":
    test_get_easting_northing_from_gps_lat_long()


