"""Nuitka pre-processing plugin: optimize slots inheritance in-memory.

This plugin runs in-memory transformations to improve inheritance
performance for classes using slots (e.g. ``@define(slots=True)`` or
``@dataclass(slots=True)``). The plugin does not write files; it only
returns an optimized source string to Nuitka.
"""
import sys
from pathlib import Path
from nuitka.plugins.PluginBase import NuitkaPluginBase

# Ensure local tools package is importable when Nuitka loads the plugin.
_tools_dir = Path(__file__).parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from py_hoist import slots_src


class SlotsOptimizerPlugin(NuitkaPluginBase):
    """Nuitka plugin: optimize slots inheritance in-memory.

    Only project source files under the ``src`` tree are processed; third-party
    or system modules are skipped.
    """

    plugin_name = "slots-optimizer"
    plugin_desc = ("Optimize inheritance performance for @define(slots=True) and "
                   "@dataclass(slots=True)")

    def onModuleSourceCode(self, module_name, source_code, source_filename):
        """Nuitka hook: optionally transform module source string.

        Args:
            module_name: the module name being compiled
            source_code: original source code as a string
            source_filename: original source filename (may be None)

        Returns:
            The (possibly) optimized source code string to feed back to Nuitka.
        """

        # Only process project source files (files under 'src')
        if source_filename and 'src' in str(source_filename):
            # Optimize code in-memory (do not modify files)
            optimized_code = slots_src(source_code, Path(source_filename) if source_filename else None)

            # Log if code was modified
            if optimized_code != source_code:
                self.info(f"\x1b[34mOptimized slots inheritance for module {module_name}\x1b[0m")

            return optimized_code

        return source_code
