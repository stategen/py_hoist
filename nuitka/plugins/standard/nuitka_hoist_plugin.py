"""
"""
import sys
from pathlib import Path
from nuitka.plugins.PluginBase import NuitkaPluginBase

# Ensure local tools package is importable when Nuitka loads the plugin.
_tools_dir = Path(__file__).parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from py_hoist import hoist_src


class HoistAttributesPlugin(NuitkaPluginBase):
    """Nuitka plugin that applies attribute-hoisting to project modules.

    It only processes files whose filename contains 'src' to avoid
    transforming third-party or system modules.
    """

    plugin_name = "hoist-attributes"
    plugin_desc = "Hoist repeated attribute accesses in project source"

    def onModuleSourceCode(self, module_name, source_code, source_filename):
        if source_filename and 'src' in str(source_filename):
            try:
                new_code = hoist_src(source_code, source_filename)
                self.info(f"\x1b[34mhoist_attributes applied for {module_name}\x1b[0m")
            except Exception as exc:  # pragma: no cover - runtime safety
                self.warning(f"hoist_attributes failed for {module_name}: {exc}")
                return source_code

            if new_code != source_code:
                self.info(f"\x1b[34mHoisted attributes in module {module_name}\x1b[0m")
            return new_code

        return source_code
