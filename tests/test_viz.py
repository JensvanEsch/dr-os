from io import BytesIO

import pytest

pytest.importorskip("matplotlib")

from matplotlib.patches import Circle, Rectangle

from dros import DeltaGeometry
from dros.analysis import Load, TargetCylinder, workspace_map
from dros.viz import plot_workspace

# Small and short-armed, so the map is quick to build.
GEOMETRY = DeltaGeometry(
    base_radius=0.030,
    end_effector_radius=0.0125,
    upper_arm=0.045,
    lower_arm=0.100,
    joint_min=-0.6,
    joint_max=1.4,
)
LOAD = Load(payload=0.5, end_effector_mass=0.05, upper_arm_mass=0.03, acceleration=2.0)
TARGET = TargetCylinder(diameter=0.060, height=0.050)


@pytest.fixture(scope="module")
def workspace():
    return workspace_map(GEOMETRY, LOAD, TARGET)


def test_plot_draws_the_target_cylinder_in_both_views(workspace):
    side, top = plot_workspace(workspace).axes[:2]
    (outline,) = [p for p in side.patches if isinstance(p, Rectangle)]
    assert outline.get_width() == pytest.approx(TARGET.diameter)
    assert outline.get_height() == pytest.approx(TARGET.height)
    assert outline.get_y() == pytest.approx(workspace.cylinder.centre_z - TARGET.height / 2)
    (circle,) = [p for p in top.patches if isinstance(p, Circle)]
    assert circle.get_radius() == pytest.approx(TARGET.diameter / 2)


def test_plot_has_a_torque_colour_bar(workspace):
    figure = plot_workspace(workspace)
    assert any("N m" in axes.get_ylabel() for axes in figure.axes[2:])


def test_plot_saves_without_a_display(workspace):
    buffer = BytesIO()
    plot_workspace(workspace).savefig(buffer, format="png")
    assert buffer.getvalue().startswith(b"\x89PNG")
