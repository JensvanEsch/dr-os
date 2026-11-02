# dr-os

**Delta Robot Open Source** — an open-source, as-cheap-as-possible delta robot:
3D-printed parts and three Feetech STS3215 serial-bus servos, plus a growing
software ecosystem for parallel manipulators.

```bash
pip install -e ".[dev,viz]"   # without viz, the plot tests are skipped
pytest
```

Plot the workspace of a geometry, with the target cylinder and shoulder torque
(needs the `viz` extra). The masses below are assumptions, not measurements:

```python
from dros import DeltaGeometry
from dros.analysis import Load, workspace_map
from dros.viz import plot_workspace

# Assumed masses: replace with the real ones once parts are weighed.
load = Load(payload=0.5, end_effector_mass=0.05, upper_arm_mass=0.03, acceleration=2.0)
plot_workspace(workspace_map(DeltaGeometry(), load)).savefig("workspace.png")
```

Status: early development. See `docs/devlog/` for progress.

Project vocabulary is defined in [`GLOSSARY.md`](GLOSSARY.md).

The software is licensed under [Apache-2.0](LICENSE).
