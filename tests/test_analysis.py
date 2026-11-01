from dataclasses import replace

import numpy as np
import pytest

from dros import DeltaGeometry, inverse
from dros.analysis import (
    Cause,
    CylinderCheck,
    Shortfall,
    TargetCylinder,
    check_cylinder,
    reachable,
    unreachable_cause,
)

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
