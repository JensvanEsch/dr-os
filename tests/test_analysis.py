from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from dros import DeltaGeometry, UnreachableError, forward, inverse
from dros.analysis import (
    SINGULARITY_LIMIT,
    Cause,
    CylinderCheck,
    Load,
    Shortfall,
    TargetCylinder,
    TorqueReport,
    WorstTorque,
    check_cylinder,
    reachable,
    shoulder_torque,
    tick_error,
    torque_report,
    unreachable_cause,
)
from dros.kinematics import singularity_margin
from dros.servo import RAD_PER_TICK

# Shoulder offset 0.06 m, so a centred end-effector forms a 6-8-10 triangle at
# theta = 0: elbow at radius 0.08, lower arm 0.10, wrist 0.06 below.
#   theta = 0     -> z = -0.06
#   theta = pi/2  -> z = -(0.02 + sqrt(0.10^2 - 0.06^2)) = -0.10
LIMITED = DeltaGeometry(
    base_radius=0.08,
    end_effector_radius=0.02,
    upper_arm=0.02,
    lower_arm=0.10,
    joint_min=0.0,
    joint_max=np.pi / 2,
)

# Same idea with limits out of the way. With arm 0 fully stretched the elbow lies
# on the line from shoulder to wrist, so sqrt(0.06^2 + z^2) = 0.04 + 0.06.
#   stretched -> z = -0.08
STRETCHED = DeltaGeometry(
    base_radius=0.08,
    end_effector_radius=0.02,
    upper_arm=0.04,
    lower_arm=0.06,
    joint_min=-np.pi,
    joint_max=np.pi,
)


# joint_min carves a bowl under each arm. Its tip is at radius 0.035 + 0.09 cos(0.6)
# = 0.109 m and z = 0.09 sin(0.6) - 0.2 = -0.149 m.
BOWL = DeltaGeometry(
    base_radius=0.060,
    end_effector_radius=0.025,
    upper_arm=0.090,
    lower_arm=0.200,
    joint_min=-0.6,
    joint_max=1.4,
)


def test_joint_limits_bound_a_centred_end_effector():
    assert reachable(LIMITED, [0.0, 0.0, -0.08])
    assert unreachable_cause(LIMITED, [0.0, 0.0, -0.08]) is None
    assert not reachable(LIMITED, [0.0, 0.0, -0.059])
    assert unreachable_cause(LIMITED, [0.0, 0.0, -0.059]) is Cause.JOINT_LIMITS
    assert not reachable(LIMITED, [0.0, 0.0, -0.101])
    assert unreachable_cause(LIMITED, [0.0, 0.0, -0.101]) is Cause.JOINT_LIMITS


def test_arm_length_bounds_a_centred_end_effector():
    assert reachable(STRETCHED, [0.0, 0.0, -0.079])
    assert not reachable(STRETCHED, [0.0, 0.0, -0.081])
    assert unreachable_cause(STRETCHED, [0.0, 0.0, -0.081]) is Cause.ARM_LENGTH


def test_reachability_of_many_positions_keeps_their_shape():
    positions = np.array(
        [[[0.0, 0.0, -0.08], [0.0, 0.0, -0.059]], [[0.0, 0.0, -0.101], [0.0, 0.0, -0.07]]]
    )
    assert reachable(LIMITED, positions).tolist() == [[True, False], [False, True]]


def test_a_position_forward_cannot_return_is_not_reachable():
    # theta = pi/2 puts the elbows at z = -0.02 and radius 0.06, and the lower arms
    # close upward to the wrists at z = 0.06. forward() returns only the lower solution.
    wide = replace(LIMITED, joint_min=-np.pi, joint_max=np.pi)
    assert not reachable(wide, [0.0, 0.0, 0.06])
    assert unreachable_cause(wide, [0.0, 0.0, 0.06]) is Cause.UPPER_SOLUTION


def test_a_position_just_above_the_elbow_plane_is_not_reachable():
    # STRETCHED centred at theta = pi/2: elbows at (0.06, z = -0.04), lower arm 0.06,
    # so the lower arms lie flat and the wrists are in the elbow plane at z = -0.04.
    assert unreachable_cause(STRETCHED, [0.0, 0.0, -0.04 + 1e-7]) is Cause.UPPER_SOLUTION


def test_lower_arms_flat_in_the_elbow_plane_are_not_called_too_short():
    # forward() can fail on rounding here, but the arms do close.
    assert unreachable_cause(STRETCHED, [0.0, 0.0, -0.04]) is not Cause.ARM_LENGTH


def test_cylinder_that_fits_between_the_joint_limits_passes():
    # Centred, z is free from -0.06 to -0.10: a 40 mm window. A 1 mm wide cylinder
    # loses under a millimetre at each end to the wrists moving off-centre.
    check = check_cylinder(LIMITED, TargetCylinder(diameter=0.001, height=0.030))
    assert check.passed
    assert check.shortfalls == ()
    assert abs(check.centre_z - -0.08) <= 0.002


def test_cylinder_taller_than_the_joint_limits_allow_fails():
    check = check_cylinder(LIMITED, TargetCylinder(diameter=0.001, height=0.041))
    assert not check.passed
    assert {shortfall.cause for shortfall in check.shortfalls} == {Cause.JOINT_LIMITS}


def test_the_target_size_can_be_changed():
    assert check_cylinder(LIMITED, TargetCylinder(diameter=0.001, height=0.030)).passed
    assert not check_cylinder(LIMITED, TargetCylinder(diameter=0.200, height=0.030)).passed


def test_a_small_hole_in_the_workspace_is_not_stepped_over():
    # This cylinder's top layer cuts the bowl just under its tip, leaving a hole about
    # 0.11 m out from the axis with reachable positions all around it.
    assert not reachable(BOWL, [0.1095, 0.0, -0.1482])
    assert not check_cylinder(BOWL, TargetCylinder(diameter=0.260, height=0.0266)).passed


def test_the_default_target_is_160_mm_wide_and_100_mm_tall():
    assert TargetCylinder() == TargetCylinder(diameter=0.160, height=0.100)


def test_a_failed_check_says_where_the_geometry_falls_short():
    # At most 40 mm of this 60 mm tall cylinder fits, so the misses sit past the
    # -0.06 and -0.10 ends of the window, give or take the 1 mm width.
    target = TargetCylinder(diameter=0.001, height=0.060)
    check = check_cylinder(LIMITED, target)
    assert not check.passed
    top = check.centre_z + target.height / 2
    bottom = check.centre_z - target.height / 2
    for shortfall in check.shortfalls:
        x, y, z = shortfall.position
        assert not reachable(LIMITED, shortfall.position)
        assert bottom - 1e-9 <= z <= top + 1e-9
        assert np.hypot(x, y) <= target.diameter / 2 + 1e-9
        assert z > -0.06 - 0.002 or z < -0.10 + 0.002
    report = str(check)
    assert "FAIL" in report
    assert "joint limits" in report


def test_report_keeps_separate_bands_of_misses_apart():
    check = CylinderCheck(
        target=TargetCylinder(),
        centre_z=-0.2,
        samples=100,
        shortfalls=(
            Shortfall((0.08, 0.0, -0.15), Cause.JOINT_LIMITS),
            Shortfall((0.07, 0.0, -0.152), Cause.JOINT_LIMITS),
            Shortfall((0.08, 0.0, -0.25), Cause.JOINT_LIMITS),
            Shortfall((0.08, 0.0, -0.25), Cause.ARM_LENGTH),
        ),
    )
    assert str(check).splitlines()[1:] == [
        "  arm length: 1 positions, z -0.250 to -0.250 m, radius 0.080 to 0.080 m",
        "  joint limits: 1 positions, z -0.250 to -0.250 m, radius 0.080 to 0.080 m",
        "  joint limits: 2 positions, z -0.152 to -0.150 m, radius 0.070 to 0.080 m",
    ]


def test_report_of_a_passing_check_names_the_placement():
    check = check_cylinder(LIMITED, TargetCylinder(diameter=0.001, height=0.030))
    assert "PASS" in str(check)
    assert f"{check.centre_z:.3f}" in str(check)


def test_a_cylinder_taller_than_the_robot_fails_cleanly():
    check = check_cylinder(LIMITED, TargetCylinder(diameter=0.001, height=1.0))
    assert not check.passed


def test_a_target_needs_positive_dimensions():
    with pytest.raises(ValueError):
        TargetCylinder(diameter=0.0)
    with pytest.raises(ValueError):
        TargetCylinder(height=-0.1)


def test_rest_position_is_reachable_and_within_joint_limits():
    geom = DeltaGeometry()
    assert reachable(geom, geom.rest_position)
    geom.check_joint_limits(inverse(geom, geom.rest_position))


def test_every_shortfall_is_really_out_of_reach_for_the_cause_given():
    check = check_cylinder(BOWL, TargetCylinder())
    assert not check.passed
    for shortfall in check.shortfalls:
        assert unreachable_cause(BOWL, shortfall.position) is shortfall.cause


def test_a_passing_cylinder_holds_up_when_resampled_more_densely():
    # 146 mm is about the widest this geometry fits. The grid here is finer than the
    # check's and includes angles it skips, so a sample standing in for the wrong
    # position shows up as a miss.
    target = TargetCylinder(diameter=0.146, height=0.100)
    check = check_cylinder(BOWL, target)
    assert check.passed
    radii = np.linspace(0.0, target.diameter / 2, 19)
    angles = np.deg2rad(np.arange(0.0, 360.0, 5.0))
    heights = check.centre_z + np.linspace(-1.0, 1.0, 11) * target.height / 2
    r, a, z = np.meshgrid(radii, angles, heights)
    assert reachable(BOWL, np.stack([r * np.cos(a), r * np.sin(a), z], axis=-1)).all()


# Shoulder offset 0.06, a 6-8-10 triangle at theta = 0 with the elbow straight out from
# the shoulder. Only the vertical part of the lower arm force turns the shoulder, so each
# one carries a third of the weight on a lever of upper_arm = 0.02.
FLAT = replace(LIMITED, joint_min=-1.0)
FLAT_POSITION = [0.0, 0.0, -0.06]

# Shoulder offset 0.04, upper arm 0.05 at sin(theta) = 0.6: elbow at (0.08, -0.03) in the
# plane of the arm, lower arm 0.10 running along (-0.8, -0.6) to the wrist at z = -0.09.
TILTED = DeltaGeometry(
    base_radius=0.06,
    end_effector_radius=0.02,
    upper_arm=0.05,
    lower_arm=0.10,
    joint_min=-1.0,
    joint_max=1.4,
)
TILTED_POSITION = [0.0, 0.0, -0.09]


def test_static_torque_of_a_centred_end_effector_on_a_horizontal_upper_arm():
    load = Load(payload=0.4, end_effector_mass=0.2, upper_arm_mass=0.0, acceleration=0.0)
    # (0.6 kg * 9.81 / 3) * 0.02 m
    assert np.allclose(shoulder_torque(FLAT, FLAT_POSITION, load).static, 0.03924)


def test_static_torque_of_a_centred_end_effector_on_a_tilted_upper_arm():
    load = Load(payload=0.4, end_effector_mass=0.2, upper_arm_mass=0.0, acceleration=0.0)
    # Lower arm tension T from 0.6 T = W / 3, with W = 0.6 kg * 9.81. The tension meets
    # the upper arm's direction of travel (0.6, 0.8) with cosine 0.96, so
    # torque = 0.05 * 0.96 * (5/9) W = 0.15696 N m
    assert np.allclose(shoulder_torque(TILTED, TILTED_POSITION, load).static, 0.15696)


def test_static_torque_includes_the_upper_arm_at_half_its_length():
    load = Load(payload=0.4, end_effector_mass=0.2, upper_arm_mass=0.03, acceleration=0.0)
    # 0.15696 from the end-effector, plus 0.03 kg * 9.81 * 0.025 m * cos(theta) = 0.8
    assert np.allclose(shoulder_torque(TILTED, TILTED_POSITION, load).static, 0.162846)


def test_static_torque_is_the_derivative_of_the_potential_energy():
    geom = DeltaGeometry()
    load = Load(payload=0.5, end_effector_mass=0.05, upper_arm_mass=0.03, acceleration=0.0)
    mass = load.payload + load.end_effector_mass

    def energy(theta: np.ndarray) -> float:
        upper_arms = load.upper_arm_mass * -0.5 * geom.upper_arm * np.sin(theta).sum()
        return 9.81 * (mass * forward(geom, theta)[2] + upper_arms)

    rng = np.random.default_rng(5)
    for theta in rng.uniform(0.0, 1.2, (30, 3)):
        position = forward(geom, theta)
        step = 1e-6 * np.eye(3)
        numeric = [(energy(theta + d) - energy(theta - d)) / 2e-6 for d in step]
        assert np.allclose(shoulder_torque(geom, position, load).static, np.abs(numeric))


def test_shoulder_torque_refuses_a_position_out_of_reach():
    load = Load(payload=0.0, end_effector_mass=0.0, upper_arm_mass=0.0, acceleration=0.0)
    with pytest.raises(UnreachableError, match="joint limits"):
        shoulder_torque(LIMITED, [0.0, 0.0, -0.059], load)


@pytest.mark.parametrize("field", ["payload", "end_effector_mass", "upper_arm_mass", "acceleration"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_load_refuses_a_value_that_is_not_a_size(field, value):
    good = {"payload": 0.5, "end_effector_mass": 0.1, "upper_arm_mass": 0.1, "acceleration": 1.0}
    with pytest.raises(ValueError):
        Load(**{**good, field: value})


def test_accelerating_torque_of_a_centred_end_effector():
    load = Load(payload=0.4, end_effector_mass=0.2, upper_arm_mass=0.0, acceleration=2.0)
    # Column 0 of the Jacobian is (-0.01, 0, -1/150) m/rad, 0.012019 long, so the worst
    # direction adds 0.6 kg * 2 m/s^2 * 0.012019 m/rad = 0.014422 to the static 0.03924.
    torque = shoulder_torque(FLAT, FLAT_POSITION, load)
    assert np.allclose(torque.static, 0.03924)
    assert np.allclose(torque.accelerating, 0.053662, atol=1e-6)


def test_accelerating_torque_of_an_upper_arm_alone():
    load = Load(payload=0.0, end_effector_mass=0.0, upper_arm_mass=0.03, acceleration=2.0)
    # Upper arm inertia 0.03 * 0.02^2 / 3 = 4e-6 kg m^2. Lower arm (-0.08, 0, -0.06) over 0.0012
    # gives dtheta/dp of (-200/3, 0, -50) rad/m, 83.33 long: 6.667e-4 N m on top of the
    # static 0.03 * 9.81 * 0.01 = 0.002943.
    torque = shoulder_torque(FLAT, FLAT_POSITION, load)
    assert np.allclose(torque.accelerating, 0.0036097, atol=1e-6)


def test_accelerating_end_effector_and_upper_arm_add_as_vectors():
    load = Load(payload=0.4, end_effector_mass=0.2, upper_arm_mass=0.03, acceleration=2.0)
    # 0.6 * (-0.01, -1/150) + 4e-6 * (-200/3, -50) = (-0.0062667, -0.0042), 0.0075440 long
    torque = shoulder_torque(FLAT, FLAT_POSITION, load)
    assert np.allclose(torque.accelerating, 0.03924 + 0.002943 + 2.0 * 0.0075440, atol=1e-6)


def test_without_acceleration_the_two_torques_are_equal():
    load = Load(payload=0.5, end_effector_mass=0.1, upper_arm_mass=0.03, acceleration=0.0)
    torque = shoulder_torque(DeltaGeometry(), [0.05, 0.02, -0.2], load)
    assert np.array_equal(torque.accelerating, torque.static)


def test_tick_error_of_a_centred_end_effector():
    # One tick is 0.001 rad, so each shoulder may be 0.0005 rad off. From the Jacobian of
    # FLAT, (+, -, -) moves the end-effector (-0.02, 0, 1/150) per radian, which is more
    # than the (0, 0, -0.02) of all three together: 0.021082 m/rad.
    assert tick_error(FLAT, FLAT_POSITION, joint_resolution=0.001) == pytest.approx(1.0541e-5, 1e-3)


def test_tick_error_matches_rounding_the_shoulders_through_forward():
    geom = DeltaGeometry()
    half = 0.0005
    corners = half * np.array(list(product([-1.0, 1.0], repeat=3)))
    rng = np.random.default_rng(6)
    for theta in rng.uniform(0.0, 1.2, (30, 3)):
        position = forward(geom, theta)
        worst = max(np.linalg.norm(forward(geom, theta + c) - position) for c in corners)
        assert tick_error(geom, position, joint_resolution=0.001) == pytest.approx(worst, 1e-2)


def test_tick_error_defaults_to_the_resolution_of_the_servo_encoder():
    expected = tick_error(FLAT, FLAT_POSITION, joint_resolution=RAD_PER_TICK)
    assert tick_error(FLAT, FLAT_POSITION) == expected


def test_accelerating_torque_off_centre_matches_finite_differences():
    load = Load(payload=0.4, end_effector_mass=0.2, upper_arm_mass=0.3, acceleration=2.0)
    position = np.array([0.06, -0.04, -0.2])
    theta = inverse(BOWL, position)
    step = 1e-6
    moves = np.array(
        [
            (forward(BOWL, theta + d) - forward(BOWL, theta - d)) / (2.0 * step)
            for d in step * np.eye(3)
        ]
    )
    rates = np.array(
        [
            (inverse(BOWL, position + d) - inverse(BOWL, position - d)) / (2.0 * step)
            for d in step * np.eye(3)
        ]
    )  # rates[j, i] is dtheta_i / dp_j
    inertia = load.upper_arm_mass * BOWL.upper_arm**2 / 3.0
    size = np.linalg.norm(0.6 * moves + inertia * rates.T, axis=1)
    torque = shoulder_torque(BOWL, position, load)
    assert np.allclose(torque.accelerating - torque.static, 2.0 * size, rtol=1e-6)


def test_torque_and_tick_error_follow_the_symmetry_of_the_arms():
    load = Load(payload=0.5, end_effector_mass=0.05, upper_arm_mass=0.03, acceleration=2.0)
    position = np.array([0.07, 0.03, -0.2])
    x, y, z = position
    c, s = np.cos(2.0 * np.pi / 3.0), np.sin(2.0 * np.pi / 3.0)
    rotated = np.array([c * x - s * y, s * x + c * y, z])  # arm i now looks like arm i + 1
    mirrored = np.array([x, -y, z])  # arms 1 and 2 swap
    for image, order in ((rotated, [1, 2, 0]), (mirrored, [0, 2, 1])):
        before = shoulder_torque(BOWL, position, load)
        after = shoulder_torque(BOWL, image, load)
        assert np.allclose(after.static[order], before.static)
        assert np.allclose(after.accelerating[order], before.accelerating)
        assert tick_error(BOWL, image) == pytest.approx(tick_error(BOWL, position))


UPPER_ARMS_ONLY = Load(payload=0.0, end_effector_mass=0.0, upper_arm_mass=0.03, acceleration=0.0)
TARGET_LOAD = Load(payload=0.5, end_effector_mass=0.05, upper_arm_mass=0.03, acceleration=2.0)

# End-effector wider than the base and upper arms well above horizontal: the whole
# usable workspace lies above the shoulders.
ABOVE_THE_SHOULDERS = DeltaGeometry(
    base_radius=0.02,
    end_effector_radius=0.06,
    upper_arm=0.10,
    lower_arm=0.06,
    joint_min=-1.5,
    joint_max=-0.3,
    rest_position=(0.0, 0.0, 0.04),
)

# Shoulder offset 0.06: at sin(theta) = 0.6 the elbow is at radius 0.10, z = -0.03, and
# the lower arms 0.10 lie flat with the end-effector centred: a singular position.
LOWER_ARMS_FLAT = replace(TILTED, base_radius=0.08)

SWEPT_GEOMETRIES = [BOWL, TILTED, LOWER_ARMS_FLAT, replace(BOWL, joint_min=-1.5, joint_max=1.57)]


@pytest.fixture(scope="module")
def report() -> TorqueReport:
    return torque_report(BOWL, TARGET_LOAD)


def test_worst_case_for_upper_arms_alone_is_an_upper_arm_at_horizontal():
    # Torque is 0.03 kg * 9.81 * 0.045 m * cos(theta), at most 0.013244 N m.
    worst = torque_report(BOWL, UPPER_ARMS_ONLY).static
    assert worst.torque == pytest.approx(0.013244, rel=1e-3)
    assert inverse(BOWL, worst.position)[worst.shoulder] == pytest.approx(0.0, abs=0.05)


def test_report_says_where_the_worst_case_is(report):
    for worst, kind in ((report.static, "static"), (report.accelerating, "accelerating")):
        at_worst = getattr(shoulder_torque(BOWL, worst.position, TARGET_LOAD), kind)
        assert worst.shoulder in (0, 1, 2)
        assert at_worst[worst.shoulder] == pytest.approx(worst.torque)
        assert at_worst.max() == pytest.approx(worst.torque)


def test_accelerating_worst_case_is_at_least_the_static_one(report):
    assert report.accelerating.torque > report.static.torque


@pytest.mark.parametrize("geom", SWEPT_GEOMETRIES)
def test_no_usable_position_is_much_worse_than_reported(geom):
    report = torque_report(geom, TARGET_LOAD)
    depth = geom.upper_arm + geom.lower_arm
    reach = geom.base_radius - geom.end_effector_radius + depth
    box = np.array([reach, reach, depth])
    rng = np.random.default_rng(7)
    found = 0
    for position in rng.uniform(-box, box, (4000, 3)):
        if unreachable_cause(geom, position) is not None:
            continue
        if singularity_margin(geom, inverse(geom, position)) < SINGULARITY_LIMIT:
            continue
        found += 1
        torque = shoulder_torque(geom, position, TARGET_LOAD)
        assert torque.static.max() < 1.1 * report.static.torque
        assert torque.accelerating.max() < 1.1 * report.accelerating.torque
        assert tick_error(geom, position) < 1.1 * report.tick_error
    assert found > 10


def test_positions_near_a_singularity_are_left_out_and_counted():
    report = torque_report(LOWER_ARMS_FLAT, TARGET_LOAD)
    assert report.near_singular > 0
    assert report.accelerating.torque < 100.0
    assert report.tick_error < 0.01
    assert f"{report.near_singular} positions near a singularity" in str(report)


def test_a_workspace_clear_of_singularities_leaves_nothing_out(report):
    assert report.near_singular == 0
    assert "singularity" not in str(report)


def test_sweep_covers_a_workspace_above_the_shoulders():
    report = torque_report(ABOVE_THE_SHOULDERS, TARGET_LOAD)
    assert report.static.position[2] > 0.0


def test_tick_error_of_the_report_is_at_its_position(report):
    assert tick_error(BOWL, report.tick_error_position) == pytest.approx(report.tick_error)


def test_report_refuses_a_geometry_without_a_workspace():
    narrow = replace(BOWL, joint_min=1.0, joint_max=1.0)
    with pytest.raises(ValueError, match="workspace"):
        torque_report(narrow, TARGET_LOAD)


def test_margin_with_torque_to_spare():
    margin = WorstTorque(0.4, (0.0, 0.0, -0.2), 1).margin(0.6)
    assert margin.spare == pytest.approx(0.2)
    assert margin.ratio == pytest.approx(1.5)
    assert margin.passed


def test_margin_short_of_torque():
    margin = WorstTorque(0.8, (0.0, 0.0, -0.2), 1).margin(0.6)
    assert margin.spare == pytest.approx(-0.2)
    assert margin.ratio == pytest.approx(0.75)
    assert not margin.passed
    assert "short" in str(margin).lower()


def test_margin_without_any_required_torque_passes():
    margin = WorstTorque(0.0, (0.0, 0.0, -0.2), 1).margin(0.6)
    assert margin.passed
    assert margin.ratio == float("inf")


@pytest.mark.parametrize("continuous", [0.0, -1.0, float("nan")])
def test_margin_refuses_a_servo_without_torque(continuous):
    with pytest.raises(ValueError):
        WorstTorque(0.4, (0.0, 0.0, -0.2), 1).margin(continuous)


def test_report_text_gives_torques_positions_and_tick_error(report):
    text = str(report)
    assert f"{report.static.torque:.3f} N m" in text
    assert f"{report.accelerating.torque:.3f} N m" in text
    assert f"shoulder {report.static.shoulder}" in text
    assert f"{report.tick_error:.6f} m" in text
