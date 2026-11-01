import numpy as np
import pytest

from dros import DeltaGeometry

GEOM = DeltaGeometry(joint_min=-0.5, joint_max=1.0)


def test_joint_limits_accept_angles_up_to_and_including_the_limits():
    GEOM.check_joint_limits([-0.5, 0.0, 1.0])


@pytest.mark.parametrize("theta", [1.0, np.array([[0.0, -0.5], [1.0, 0.5]])])
def test_joint_limits_accept_scalars_and_arrays(theta):
    GEOM.check_joint_limits(theta)


@pytest.mark.parametrize("theta", [np.nan, np.array([[0.0, np.nan], [0.0, 0.0]])])
def test_joint_limits_refuse_nan_in_scalars_and_arrays(theta):
    with pytest.raises(ValueError):
        GEOM.check_joint_limits(theta)


@pytest.mark.parametrize("bad", [-0.6, 1.1, np.nan, np.inf, -np.inf])
def test_joint_limits_refuse_anything_outside_them(bad):
    with pytest.raises(ValueError):
        GEOM.check_joint_limits([0.0, bad, 0.0])
