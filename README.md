# dr-os

**Delta Robot Open Source** — an open-source, as-cheap-as-possible delta robot:
3D-printed parts and three Feetech STS3215 serial-bus servos, plus a growing
software ecosystem for parallel manipulators.

```bash
pip install -e ".[dev,viz]"   # without viz, the plot tests are skipped
pytest
```

Plot the workspace of the current geometry, with the target cylinder and shoulder
torque (needs the `viz` extra; the default masses are assumptions, see `--help`):

```bash
python demo/plot_workspace.py --open
```

Status: early development. See `docs/devlog/` for progress.

Project vocabulary is defined in [`GLOSSARY.md`](GLOSSARY.md).

The software is licensed under [Apache-2.0](LICENSE).
