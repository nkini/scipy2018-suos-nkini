"""Test the `geology` module."""

from hugs.calc import snell_angle

from numpy.testing import assert_almost_equal

import pytest

def test_snell():
    """Test the basic wind component calculation."""
    res = snell_angle(12, 2500, 4000)
    assert_almost_equal(res, 19.43022, 4)


def test_exception_snell_angle():
    """Test that warning is raised when wind direction > 360."""
    with pytest.raises(ValueError):
        snell_angle(3, 0, 12)
