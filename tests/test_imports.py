import subprocess
import sys

# Importing dros.viz must fail in the same interpreter, or blocking matplotlib proved nothing.
_SCRIPT = """
import sys
sys.modules["matplotlib"] = None
import dros, dros.analysis, dros.kinematics, dros.geometry, dros.servo, dros.sim
try:
    import dros.viz
except ImportError as error:
    print(f"core ok, viz needs {error.name}")
"""


def test_the_core_package_imports_without_the_viz_extra():
    run = subprocess.run(
        [sys.executable, "-c", _SCRIPT], check=False, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "core ok, viz needs matplotlib"
