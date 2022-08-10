"""Test Module."""

import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from tool import *
from geo import *

import numpy as np

from pytest import mark



tool = Tool()


import numpy as np

from pytest import mark
def test_get_gps_lat_long_from_easting_northing():
    """Check """

    data1,data2 = get_gps_lat_long_from_easting_northing([429157], [623009])

    if data1 is NotImplemented:
        assert False

    assert np.isclose(data1[0], 55.5).all()
    assert np.isclose(data2[0], -1.5384).all()
    
    
if __name__ == "__main__":
    test_get_gps_lat_long_from_easting_northing()