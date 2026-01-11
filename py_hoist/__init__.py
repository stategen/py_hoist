"""`py_hoist` 包入口。

此模块对外导出主要 API：`hoist_src`（属性提升）和 `slots_src`（slots 优化）。
实现位于同一包内的模块：`hoist_cst`、`slots_optimizer`。
"""
from .hoist_ast import hoist_src
from .slots_optimizer import slots_src  # re-export class-based optimizer
from .multi_items import MultiItems
from pathlib import Path
import sys


def get_nuitka_plugin_path() -> str:
    """Return the filesystem path to py_hoist's `nuitka_hoist_plugin.py` if present.

	Returns empty string if not found.
	"""
    p = Path(__file__).with_name('nuitka_hoist_plugin.py')
    return str(p) if p.exists() else ""


def _get_run_hoist_path() -> str:
    from . import hoist_code as _r
    return str(getattr(_r, 'hoist_path', ''))


hoist_path = _get_run_hoist_path()

__all__ = ["hoist_src", "slots_src", "hoist_path", "get_nuitka_plugin_path"]

# If a pre-imported `py_hoist.hoist_code` entry exists (e.g. created
# by earlier import activity), remove it to avoid `runpy` emitting a
# RuntimeWarning when executing the submodule with `-m`.
_runner_mod = __name__ + '.hoist_code'
if _runner_mod in sys.modules:
    del sys.modules[_runner_mod]

__version__ = "0.1.0"
