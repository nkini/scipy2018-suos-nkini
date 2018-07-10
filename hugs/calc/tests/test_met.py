"""Test the `met` module."""

from hugs.calc import get_wind_dir, get_wind_speed, get_wind_components, snell_angle

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal


def test_speed():
    """Test calculating wind speed."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.])

    assert_array_almost_equal(true_speed, speed, 4)


def test_scalar_speed():
    """Test wind speed with scalars."""
    s = get_wind_speed(-3., -4.)
    assert_almost_equal(s, 5., 3)


def test_dir():
    """Test calculating wind direction."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    direc = get_wind_dir(u, v)

    true_dir = np.array([270., 225., 180., 270.])

    assert_array_almost_equal(true_dir, direc, 4)


def test_scalar_components():
    """Test calculating wind components with scalars"""
    components = get_wind_components(150, 0)
    assert_almost_equal(components, (0, -150))
    components = get_wind_components(-150, 90)
    assert_almost_equal(components, (150, 0))
    components = get_wind_components(10, 30)
    assert_almost_equal(components, (-5, -8.660254))
    components = get_wind_components(20, 45)
    assert_almost_equal(components, (-14.1421356, -14.1421356))
    

def test_vector_components():
    """Test calculating wind components with vectors"""
    components = get_wind_components(np.array([150, -150, 10, 20]), np.array([0, 90, 30, 45]))
    results_should_be = [[0, 150, -5, -14.1421356],
                         [-150, 0, -8.660254, -14.1421356]]
    assert_array_almost_equal(results_should_be, components)


def test_warning_direction():
    """Test that warning is raised when wind direction > 360."""
    with pytest.warns(UserWarning):
        get_wind_components(3, 460)
