"""Plots of the workspace. Needs the viz extra: pip install "dr-os[viz]"."""
from __future__ import annotations

import numpy as np

try:
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle
except ImportError as error:
    raise ImportError(
        'dros.viz needs the viz extra: pip install "dr-os[viz]"', name="matplotlib"
    ) from error

from .analysis import CylinderCheck, WorkspaceMap, WorkspaceSlice

X, Y, Z = 0, 1, 2  # index of each coordinate in a position

_NEAR_SINGULAR = ListedColormap(["#c8c8c8"])
_OUTLINE = {"fill": False, "edgecolor": "black", "linestyle": "--", "linewidth": 1.2}
_PADDING = 0.01  # m of blank space around the reachable positions


def _draw(
    axes: Axes, slice_: WorkspaceSlice, across: int, up: int, scale: tuple[float, float]
) -> QuadMesh:
    """Draw slice_ with coordinate across to the right and up upward (X, Y or Z)."""
    right = slice_.positions[..., across]
    above = slice_.positions[..., up]
    near_singular = np.where(slice_.reachable & np.isnan(slice_.torque), 1.0, np.nan)
    axes.pcolormesh(right, above, near_singular, cmap=_NEAR_SINGULAR, shading="nearest")
    image = axes.pcolormesh(
        right, above, slice_.torque, cmap="viridis", vmin=scale[0], vmax=scale[1],
        shading="nearest",
    )
    if slice_.reachable.any():
        right, above = right[slice_.reachable], above[slice_.reachable]
        axes.set_xlim(right.min() - _PADDING, right.max() + _PADDING)
        axes.set_ylim(above.min() - _PADDING, above.max() + _PADDING)
    axes.set_aspect("equal")
    axes.grid(True, alpha=0.3)
    return image


def _verdict(cylinder: CylinderCheck) -> str:
    size = f"{cylinder.target.diameter:.3f} m wide, {cylinder.target.height:.3f} m tall"
    if cylinder.passed:
        return f"PASS: target cylinder {size} fits"
    return (
        f"FAIL: target cylinder {size} does not fit, "
        f"{len(cylinder.shortfalls)} of {cylinder.samples} sampled positions out of reach"
    )


def plot_workspace(view: WorkspaceMap) -> Figure:
    """A side and a top view of the workspace, coloured by shoulder torque.

    The target cylinder is drawn dashed at its best placement. Grey positions are
    reachable but too close to a singularity to give a torque. The colours cover only
    these two slices; the worst case over the whole workspace is in torque_report.
    Uses no display; save the figure with figure.savefig(path).
    """
    torques = np.concatenate([view.side.torque.ravel(), view.top.torque.ravel()])
    usable = torques[~np.isnan(torques)]
    scale = (float(usable.min()), float(usable.max())) if len(usable) else (0.0, 1.0)

    figure = Figure(figsize=(12.0, 5.5), layout="constrained")
    side, top = figure.subplots(1, 2)
    image = _draw(side, view.side, X, Z, scale)
    _draw(top, view.top, X, Y, scale)

    cylinder = view.cylinder
    radius = cylinder.target.diameter / 2.0
    bottom = cylinder.centre_z - cylinder.target.height / 2.0
    side.add_patch(
        Rectangle((-radius, bottom), cylinder.target.diameter, cylinder.target.height, **_OUTLINE)
    )
    side.axhline(cylinder.centre_z, color="black", linewidth=0.6, alpha=0.5)
    top.add_patch(Circle((0.0, 0.0), radius, **_OUTLINE))

    side.set(title="Side, y = 0 (arm 0 along +x)", xlabel="x (m)", ylabel="z (m)")
    top.set(
        title=f"Top, z = {cylinder.centre_z:.3f} m (solid line in side view)",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    load = view.load
    figure.suptitle(
        f"{_verdict(cylinder)}, best placement z = {cylinder.centre_z:.3f} m\n"
        f"Largest shoulder torque at {load.acceleration:.1f} m/s^2, "
        f"payload {load.payload:.3f} kg, on these slices only (whole workspace: torque_report)"
    )
    figure.colorbar(image, ax=[side, top], label="accelerating shoulder torque (N m)")
    return figure
