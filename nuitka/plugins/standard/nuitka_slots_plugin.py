"""Nuitka 预处理插件：在内存中优化 slots 继承（不修改文件路径）"""
import sys
from pathlib import Path
from nuitka.plugins.PluginBase import NuitkaPluginBase

# 添加工具目录到路径
_tools_dir = Path(__file__).parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from py_hoist import slots_src


class SlotsOptimizerPlugin(NuitkaPluginBase):
    """Nuitka 插件类：在内存中优化 slots 继承"""
    plugin_name = "slots-optimizer"
    plugin_desc = "优化 @define(slots=True) 和 @dataclass(slots=True) 的继承性能"

    def onModuleSourceCode(self, module_name, source_code, source_filename):
        """Nuitka 预处理钩子：在内存中优化源代码（不修改文件路径）
        
        Args:
            module_name: 模块名
            source_code: 源代码字符串
            source_filename: 源文件路径
        
        Returns:
            优化后的源代码字符串
        """
        # 只处理项目源代码（src 目录中的文件）
        if source_filename and 'src' in str(source_filename):
            # 在内存中优化代码（不修改文件路径）
            optimized_code = slots_src(source_code, Path(source_filename) if source_filename else None)

            # 如果代码被修改了，记录日志
            if optimized_code != source_code:
                self.info(f"优化了模块 {module_name} 的 slots 继承")

            return optimized_code

        return source_code
