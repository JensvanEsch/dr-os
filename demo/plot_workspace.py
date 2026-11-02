"""Plot the workspace of the current DeltaGeometry, with the target cylinder and torque.

Needs the viz extra: pip install -e ".[viz]". Takes about 15 s.

    python demo/plot_workspace.py                      # writes workspace.png
    python demo/plot_workspace.py out.png --payload 0.3 --open
"""
from __future__ import annotations

import argparse
import subprocess

from dros import DeltaGeometry
from dros.analysis import Load, workspace_map
from dros.viz import plot_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default="workspace.png", help="image to write")
    # Assumed masses, not measured: replace once the parts are weighed.
    parser.add_argument("--payload", type=float, default=0.5, help="kg (assumed)")
    parser.add_argument("--end-effector-mass", type=float, default=0.05, help="kg (assumed)")
    parser.add_argument("--upper-arm-mass", type=float, default=0.03, help="kg (assumed)")
    parser.add_argument("--acceleration", type=float, default=2.0, help="m/s^2")
    parser.add_argument("--open", action="store_true", help="open the image when done")
    args = parser.parse_args()

    load = Load(
        payload=args.payload,
        end_effector_mass=args.end_effector_mass,
        upper_arm_mass=args.upper_arm_mass,
        acceleration=args.acceleration,
    )
    plot_workspace(workspace_map(DeltaGeometry(), load)).savefig(args.output, dpi=110)
    print(f"wrote {args.output}")
    if args.open:
        subprocess.run(["xdg-open", args.output], check=False)


if __name__ == "__main__":
    main()
