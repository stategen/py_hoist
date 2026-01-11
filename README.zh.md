# py_hoist

[English version (README.md)](README.md)



## 1. py_hoist.hoist_src — 使用说明
`py_hoist.hoist_src` 基于 AST 实现“深层属性访问变量化”。
它会将函数中多次出现的深层属性访问提升为局部变量（例如将 `self.a` 提取为 `self_a = self.a`，或将 `obj.path.join` 提取为 `obj_path_join = obj.path.join`），
以减少热路径上的重复查找，提高运行时性能，并降低因短路/守卫（guard）引入的回归风险。

核心能力：
- 自动检测函数体或循环体内的深层属性访问，并在保守、安全的前提下生成局部变量提升（hoist）。
- 采用“先前后上”（forward-then-up）搜索策略，并结合 Stop-on-Write、if/test 首项保护、BoolOp/短路保护等保守规则，以尽量避免语义回归。


## 示例（sample.py）

```py
    def test2(items):
        total = 0
        for it in items:
            if it.deep:
                total+= util.do_something(it.deep)
        return total 

```
## 转换后（示例预期结果）sample.py

```py
    def test2(items):
        total = 0
        util_do_something=util.do_something
        for it in items:
            it_deep=it.deep
            if it_deep:
                total+= util_do_something(it_deep)
        return total
```


## 使用方式（命令行）
1) 可选：将包安装到虚拟环境（便于从任意目录通过 `-m` 调用）：

```powershell
pip install -e external\py_hoist
```

2) 命令行调用（处理整个 `src/`，默认为每个修改的文件生成备份 `.py.back`）：

```powershell
python -m py_hoist.hoist_code \path\to\your\project\src
```

3) 或只处理单个文件（示例）。若不想创建备份可加上 `-noback`：

```powershell
python -m py_hoist.hoist_code \path\to\your\project\src\db\symbol_config_mng.py -noback
```




##### 命令行参数说明

- `input_path`：要处理的文件或目录（目录会递归查找 `.py` 文件）。
- `-noback`：可选开关。默认会为每个修改的文件生成备份后缀 `.py.back`；传入 `-noback` 则不创建备份。





4) 代码方式（仅预览或在脚本中调用）
```py
from pathlib import Path
from py_hoist import hoist_path

p = Path(r"path\to\your\project\sample.py")
hoist_path(p, create_backup =True)

```
5) Nuitka 使用示例：

```
--user-plugin=external/py_hoist/nuitka/plugins/standard/nuitka_hoist_plugin.py
```

## 2. py_hoist.slots_src — 使用说明
`py_hoist.slots_src` 是一个面向 Nuitka 的插件。当 Nuitka 编译代码时，若发现子类使用 `@dataclass(slots=True)`，
且其父类定义了 `@dataclass` 字段，插件会将父类的 dataclass 字段复制到子类中，以便在编译时获得更好的 slots/内存优化。

Nuitka 使用示例：

```
--user-plugin=external/py_hoist/nuitka/plugins/standard/nuitka_slots_plugin.py
```





- 欢迎提交回归测试用例或就提升策略提出 issue。默认策略偏保守，以降低引入回归的风险。
