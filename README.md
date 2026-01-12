# py_hoist

[中文说明（README.zh.md）](README.zh.md)

py_hoist is a Python toolkit designed to improve runtime performance by speeding up hot-path attribute accesses. It currently provides two tools: `hoist_src` and `slots_src`.


## 1. py_hoist.hoist_src
`py_hoist.hoist_src` implements "deep-access variableization" based on the Python AST.
It extracts frequently accessed deep attribute expressions inside a function into local variables (for example, turning `self.a` into `self_a = self.a`, or `obj.path.join` into `obj_path_join = obj.path.join`).
This reduces repeated deep lookups on hot paths, improves runtime performance, and reduces the risk of regressions caused by short-circuiting/guards.

Key capabilities:
- Automatically detect deep attribute accesses inside function bodies or loops and, under conservative safety rules, create local hoist variables.
- Use a "forward-then-up" search strategy combined with conservative rules such as Stop-on-Write, protecting first operands of `if`/`test`, and BoolOp/short-circuit protections to avoid semantic regressions.


## Example (sample.py)

```py
    def test2(items):
        total = 0
        for it in items:
            if it.deep:
                total+= util.do_something(it.deep)
        return total 

```

Transformed (expected example result):

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


## Usage (CLI)
1) Optional: install the package into a virtual environment (so you can call it with `-m` from any location):

```powershell
pip install -e external\py_hoist
```

2) Run over the whole `src/` tree (by default a `.py.back` backup is written for each modified file):

```powershell
python -m py_hoist.hoist_code \path\to\your\project\src
```

3) Or run on a single file (example). Use `-noback` to skip writing backups:

```powershell
python -m py_hoist.hoist_code \path\to\your\project\src\db\symbol_config_mng.py -noback
```


##### CLI arguments

- `input_path`: file or directory to process (directories are recursively searched for `.py`).
- `-noback`: optional switch. By default a `.py.back` backup is created for each modified file; pass `-noback` to skip backups.


4) Programmatic usage (for previewing or calling from scripts):

```py
from pathlib import Path
from py_hoist import hoist_path

p = Path(r"path\to\your\project\sample.py")
hoist_path(p, create_backup=True)

```

5) Nuitka example (pass the plugin path to the build):

```
--user-plugin=external/py_hoist/nuitka/plugins/standard/nuitka_hoist_plugin.py
```


## 2. py_hoist.slots_src — Usage
`py_hoist.slots_src` is a plugin for Nuitka. When Nuitka compiles code and encounters a subclass annotated with `@dataclass(slots=True)`,
and the parent class defines `@dataclass` fields, this plugin copies the parent's dataclass fields into the subclass so Nuitka can perform better slots/memory optimizations during compilation.

Nuitka example:

```
--user-plugin=external/py_hoist/nuitka/plugins/standard/nuitka_slots_plugin.py
```


- Contributions: welcome regression tests or issues discussing hoisting strategies. The default strategy is conservative to reduce regression risk.

