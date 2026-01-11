#kernprof -l -v external\py_hoist\py_hoist\tests\run_hoist_kernprof.py
import sys
import runpy

# 把要传给模块的 argv 放在这里
sys.argv = ['py_hoist.hoist_code', '-noback', './src/']

# 以 -m py_hoist.hoist_code 的方式运行模块主体
runpy.run_module('py_hoist.hoist_code', run_name = '__main__')
