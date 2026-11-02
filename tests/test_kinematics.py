import numpy as np
import pytest

from dros import DeltaGeometry, UnreachableError, forward, inverse, jacobian
from dros.servo import MockBus, rad_to_ticks, safe_write, ticks_to_rad

GEOM = DeltaGeometry()
RNG = np.random.default_rng(0)


def test_home_is_centred_below_base():
    p = forward(GEOM, np.zeros(3))
    assert abs(p[0]) < 1e-12 and abs(p[1]) < 1e-12
    assert p[2] < 0


def test_ik_fk_round_trip_joint_space():
    for _ in range(2000):
        theta = RNG.uniform(-0.3, 1.0, size=3)
        p = forward(GEOM, theta)
        assert np.allclose(inverse(GEOM, p), theta, atol=1e-9)


def test_fk_ik_round_trip_cartesian():
    z0 = forward(GEOM, np.full(3, 0.4))[2]
    for _ in range(2000):
        p = np.array([*RNG.uniform(-0.04, 0.04, size=2), z0 + RNG.uniform(-0.03, 0.03)])
        try:
            theta = inverse(GEOM, p)
        except UnreachableError:
            continue
        assert np.allclose(forward(GEOM, theta), p, atol=1e-9)


def test_symmetry_on_z_axis():
    theta = inverse(GEOM, [0.0, 0.0, -0.2])
    assert np.allclose(theta, theta[0])


def test_unreachable_raises():
    with pytest.raises(UnreachableError):
        inverse(GEOM, [0.0, 0.0, -1.0])


def test_safe_write_enforces_limits():
    bus = MockBus()
    safe_write(bus, GEOM, [0.1, 0.2, 0.3])
    assert np.allclose(bus.read_angles(), [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        safe_write(bus, GEOM, [0.0, 0.0, 3.0])


def test_tick_conversion_round_trip():
    theta = np.array([-0.5, 0.0, 1.2])
    back = ticks_to_rad(rad_to_ticks(theta, zero_tick=2000, direction=-1),
                        zero_tick=2000, direction=-1)
    assert np.allclose(back, theta, atol=2 * np.pi / 4096)


def test_jacobian_matches_a_finite_difference_of_forward():
    rng = np.random.default_rng(3)
    step = 1e-6
    for _ in range(50):
        theta = rng.uniform(-0.2, 1.0, 3)
        numeric = np.empty((3, 3))
        for i in range(3):
            delta = step * np.eye(3)[i]
            numeric[:, i] = (forward(GEOM, theta + delta) - forward(GEOM, theta - delta)) / (
                2.0 * step
            )
        assert np.allclose(jacobian(GEOM, theta), numeric, rtol=1e-6, atol=1e-8)


def test_jacobian_of_a_centred_end_effector_by_hand():
    # Shoulder offset 0.06, upper arm 0.02, lower arm 0.10: a 6-8-10 triangle at theta = 0.
    # Lower arm of arm i runs along d_i = (-0.08 cos phi_i, -0.08 sin phi_i, -0.06) and the
    # elbow moves straight down, so d_i . dp = 0.0012 dtheta_i. Solving for arm 0 alone:
    #   dx = -0.01, dy = 0, dz = -1/150
    geom = DeltaGeometry(base_radius=0.08, end_effector_radius=0.02, upper_arm=0.02, lower_arm=0.10)
    assert np.allclose(jacobian(geom, np.zeros(3))[:, 0], [-0.01, 0.0, -1.0 / 150.0])
