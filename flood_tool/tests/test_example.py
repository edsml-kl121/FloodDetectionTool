"""Test Module."""

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
import numpy as np

from pytest import mark



tool = Tool()


def test_get_easting_northing():
    """Check """

    data = tool.get_easting_northing(['YO62 4LS'])

    if data is NotImplemented:
        assert False

    assert np.isclose(data.iloc[0].easting, 467631.0).all()
    assert np.isclose(data.iloc[0].northing, 472825.0).all()


@mark.xfail  # We expect this test to fail until we write some code for it.
def test_get_lat_long():
    """Check """

    data = tool.get_lat_long(['YO62 4LS'])

    if data is NotImplemented:
        assert False

    assert np.isclose(data.iloc[0].latitude, 54.147, 1.0e-3).all()
    assert np.isclose(data.iloc[0].northing, -0.966, 1.0e-3).all()


if __name__ == "__main__":
    test_get_easting_northing()
