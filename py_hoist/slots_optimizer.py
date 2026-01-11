"""Nuitka 编译前优化：将父类 slots 属性复制到子类以获得性能优势

主要 API:
    optimize_code(source_code: str) -> str  # 在内存中优化代码，返回优化后的代码字符串
"""
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class SlotsOptimizer:
    """专门为 Nuitka 编译优化的 slots 属性复制工具"""

    def __init__(self, source_code: str, file_path: Optional[Path] = None):
        self.source_code = source_code
        self.file_path = file_path
        self.lines = source_code.split('\n')
        self.tree = ast.parse(source_code, filename = str(file_path) if file_path else '<unknown>')

        # 类信息：类名 -> {attrs: [], parents: [], has_slots: bool, lineno: int}
        self.class_info: Dict[str, Dict] = {}

        self._analyze()

    def _analyze(self):
        """分析代码结构"""
        # 收集类信息
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                self._analyze_class(node)

    def _analyze_class(self, node: ast.ClassDef):
        """分析单个类的结构"""
        class_name = node.name
        attrs: List[Tuple[str, Optional[str]]] = []  # (attr_name, type_annotation)
        parents: List[str] = []
        parent_imports: Dict[str, str] = {}  # 父类名 -> 导入模块路径
        has_slots = False
        has_dataclass = False

        # 检查装饰器：@define(slots=True) 或 @dataclass 或 @dataclass(slots=True)
        for decorator in node.decorator_list:
            if self._check_slots_decorator(decorator):
                has_slots = True
                has_dataclass = True
                break
            elif self._check_dataclass_decorator(decorator):
                has_dataclass = True

        # 获取父类及其导入信息
        for base in node.bases:
            if isinstance(base, ast.Name):
                parent_name = base.id
                parents.append(parent_name)
                # 查找父类的导入信息
                parent_imports[parent_name] = self._find_import_for_name(parent_name)
            elif isinstance(base, ast.Attribute):
                # 处理 module.Class 这种情况
                parent_name = base.attr
                parents.append(parent_name)
                module_path = self._extract_attribute_path(base)
                parent_imports[parent_name] = module_path

        # 获取类的属性（类型注解），记录是否有默认值及其默认值
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attr_name = item.target.id
                    attr_type = self._extract_type_annotation(item.annotation)
                    # 检查是否有默认值（value 不为 None 表示有默认值）
                    has_default = item.value is not None
                    # 提取默认值（如果有）
                    default_value_str = None
                    if has_default:
                        default_value_str = self._extract_default_value(item.value)
                    attrs.append((attr_name, attr_type, has_default, default_value_str))

        # 记录所有 @dataclass 类（无论是否有 slots=True，都要记录以便作为父类时提取属性）
        if has_dataclass:
            self.class_info[class_name] = {
                        'attrs': attrs,
                        'parents': parents,
                        'parent_imports': parent_imports,
                        'has_slots': has_slots,
                        'has_dataclass': has_dataclass,
                        'lineno': node.lineno,
                        'node': node
            }

    def _check_slots_decorator(self, decorator) -> bool:
        """检查是否是 slots=True 的装饰器"""
        if isinstance(decorator, ast.Call):
            func = decorator.func
            decorator_name = None

            if isinstance(func, ast.Name):
                decorator_name = func.id
            elif isinstance(func, ast.Attribute):
                decorator_name = func.attr

            if decorator_name in ('define', 'dataclass'):
                # 检查是否有 slots=True 参数
                for keyword in decorator.keywords:
                    if keyword.arg == 'slots':
                        if isinstance(keyword.value, ast.Constant):
                            return keyword.value.value is True

        return False

    def _check_dataclass_decorator(self, decorator) -> bool:
        """检查是否是 @dataclass 装饰器（不管是否有 slots）"""
        if isinstance(decorator, ast.Name):
            return decorator.id in ('dataclass', 'define')
        elif isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name):
                return func.id in ('dataclass', 'define')
            elif isinstance(func, ast.Attribute):
                return func.attr in ('dataclass', 'define')
        return False

    def _find_import_for_name(self, name: str) -> str:
        """查找类名的导入路径"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        if alias.asname == name or (alias.asname is None and alias.name == name):
                            return node.module
        return ''

    def _extract_attribute_path(self, node: ast.Attribute) -> str:
        """提取 module.Class 的模块路径"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        return '.'.join(parts) if parts else ''

    def _extract_type_annotation(self, annotation) -> Optional[str]:
        """提取类型注解的字符串表示"""
        if annotation is None:
            return None
        return ast.unparse(annotation)

    def _parse_attr_info(self, attr_info) -> Tuple[Optional[str], bool, Optional[str]]:
        """解析属性信息，返回 (attr_type, has_default, default_value_str)"""
        if isinstance(attr_info, tuple) and len(attr_info) >= 3:
            return attr_info[0], attr_info[1], attr_info[2]
        elif isinstance(attr_info, tuple) and len(attr_info) == 2:
            return attr_info[0], attr_info[1], None
        elif isinstance(attr_info, tuple) and len(attr_info) == 1:
            return attr_info[0], False, None
        else:
            return attr_info[0] if isinstance(attr_info, tuple) and attr_info else None, False, None

    def _extract_default_value(self, value_node) -> Optional[str]:
        """提取默认值的字符串表示"""
        try:
            return ast.unparse(value_node)
        except Exception:
            return None

    def _get_name_to_import_info(self) -> Dict[str, tuple]:
        """获取名称到导入信息的映射（从当前文件的导入中）
        返回: {name: (module, import_type)} 
        import_type: 'from' 或 'import'
        """
        name_to_import = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        name_to_import[name] = (node.module, 'from')
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    # import module 的情况
                    name_to_import[name] = (name, 'import')
        return name_to_import

    def _extract_imports_from_default_value(self, value_node, existing_imports: Set[str], name_to_import: Dict[str,
                                                                                                               tuple]) -> Set[tuple]:
        """从默认值中提取需要的导入（返回 (module, name, import_type) 的集合）
        import_type: 'from' 或 'import'
        """
        imports_needed = set()

        if isinstance(value_node, ast.Name):
            name = value_node.id
            # 排除内置常量、类型和内置函数
            builtin_names = ('None', 'True', 'False', 'Ellipsis', '...', 'list', 'dict', 'tuple', 'set', 'frozenset', 'str', 'int',
                             'float', 'bool', 'bytes', 'bytearray', 'complex', 'object', 'type', 'range', 'slice', 'memoryview',
                             'array')
            if name not in builtin_names and name not in existing_imports:
                # 查找名称对应的导入信息
                if name in name_to_import:
                    module, import_type = name_to_import[name]
                    imports_needed.add((module, name, import_type))
                else:
                    imports_needed.add((None, name, 'import'))
        elif isinstance(value_node, ast.Attribute):
            # 对于 module.Class，需要导入模块
            if isinstance(value_node.value, ast.Name):
                module_name = value_node.value.id
                attr_name = value_node.attr
                # 检查是否已经是完整的导入路径
                if module_name not in existing_imports:
                    imports_needed.add((module_name, attr_name, 'from'))
        elif isinstance(value_node, ast.Call):
            # 函数调用，检查函数名（如 field）
            if isinstance(value_node.func, ast.Name):
                func_name = value_node.func.id
                if func_name not in existing_imports and func_name not in ('dict', 'list', 'tuple', 'set', 'frozenset'):
                    # 查找函数名对应的导入信息
                    if func_name in name_to_import:
                        module, import_type = name_to_import[func_name]
                        imports_needed.add((module, func_name, import_type))
                    else:
                        imports_needed.add((None, func_name, 'import'))
            elif isinstance(value_node.func, ast.Attribute):
                if isinstance(value_node.func.value, ast.Name):
                    module_name = value_node.func.value.id
                    attr_name = value_node.func.attr
                    if module_name not in existing_imports:
                        imports_needed.add((module_name, attr_name, 'from'))

            # 递归处理函数调用的所有参数和关键字参数（如 field(default_factory=AppConfig)）
            for arg in value_node.args:
                imports_needed.update(self._extract_imports_from_default_value(arg, existing_imports, name_to_import))
            for keyword in value_node.keywords:
                if keyword.value:
                    imports_needed.update(self._extract_imports_from_default_value(keyword.value, existing_imports, name_to_import))

        return imports_needed

    def optimize(self) -> str:
        """优化代码，返回优化后的代码"""
        if not self.class_info:
            return self.source_code

        # 找到需要优化的类（有父类且父类有属性的子类）
        modifications: List[Tuple[int, str, str]] = []  # (line_number, new_code, indent)

        for class_name, info in self.class_info.items():
            if not info.get('has_slots'):
                continue  # 只有 slots=True 的子类才需要优化
            if not info['parents']:
                continue  # 没有父类，跳过

            # 检查子类已有的属性：当子类有相同字段名的字段时，不要把父类的相同字段名的字段加进来
            existing_attrs = {
                        attr_info[0]
                        for attr_info in info['attrs']
                        if isinstance(attr_info, (list, tuple)) and len(attr_info) >= 1 and isinstance(attr_info[0], str)
            }

            # 直接从父类收集属性（不包含子类自己的属性）
            # 父类只要有 @dataclass 都要提取属性（不管是否有 slots=True）
            inherited_attrs = {}
            for parent in info['parents']:
                parent_attrs, parent_name_to_import = self._collect_inherited_attrs(parent, visited = set())
                for attr, attr_info in parent_attrs.items():
                    # 关键检查：只添加子类没有定义的父类属性
                    # 如果子类已经有同名字段，则跳过父类的同名字段
                    if attr in existing_attrs:
                        continue  # 跳过子类已有的属性
                    if attr in inherited_attrs:
                        continue  # 跳过已经收集过的属性（多个父类可能有同名属性）
                    inherited_attrs[attr] = attr_info

            # missing_attrs 就是 inherited_attrs（因为已经排除了子类已有的）
            missing_attrs = inherited_attrs

            if not missing_attrs:
                continue  # 所有属性都已存在

            # 复制所有父类属性到子类（包括有默认值的）
            # 需要确保参数顺序正确：无默认值的在前，有默认值的在后

            # 分析子类已有属性的顺序，找到正确的插入位置
            child_attrs_order = self._analyze_child_attrs_order(info['node'])

            # 分离生成的属性：无默认值的和有默认值的
            new_attrs_without_default = {}
            new_attrs_with_default = {}
            for attr, attr_info in missing_attrs.items():
                attr_type, has_default, default_value_str = self._parse_attr_info(attr_info)
                if has_default:
                    new_attrs_with_default[attr] = (attr_type, has_default, default_value_str)
                else:
                    new_attrs_without_default[attr] = (attr_type, has_default, default_value_str)

            # 找到插入位置和缩进（使用AST节点）：
            # 1. 无默认值的属性：如果子类有最后一个无默认值属性，插入到其后；否则如果有第一个有默认值属性，插入到其之前
            # 2. 有默认值的属性：插入到子类最后一个有默认值属性之后
            last_without_default_node = child_attrs_order.get('last_without_default_node')
            last_without_default_line = child_attrs_order.get('last_without_default_line')
            first_with_default_node = child_attrs_order.get('first_with_default_node')
            first_with_default_line = child_attrs_order.get('first_with_default_line')
            last_with_default_node = child_attrs_order.get('last_with_default_node')
            last_with_default_line = child_attrs_order.get('last_with_default_line')
            class_node = child_attrs_order.get('class_node')

            if new_attrs_without_default:
                # 无默认值的属性：应该插入到子类的最后一个无默认值属性之后，或第一个有默认值属性之前
                if last_without_default_node is not None:
                    # 子类有无默认值的属性，插入到最后一个无默认值属性之后
                    insert_line = last_without_default_line + 1
                    indent = self._get_indent_from_ast_node(last_without_default_node)
                elif first_with_default_node is not None:
                    # 子类没有无默认值的属性，但有有默认值的属性，插入到第一个有默认值属性之前
                    insert_line = first_with_default_line
                    indent = self._get_indent_from_ast_node(first_with_default_node)
                else:
                    # 子类没有任何属性，插入到类体开始
                    insert_line = self._find_insertion_point(info['node'])
                    if insert_line is None:
                        continue
                    # 从类节点计算缩进：类体应该比类定义多一层缩进
                    indent = self._calculate_class_body_indent(class_node)

                new_attrs_code = self._generate_attrs_code(new_attrs_without_default)
                modifications.append((insert_line, new_attrs_code, indent))

            if new_attrs_with_default:
                # 有默认值的属性：插入到子类最后一个有默认值属性之后
                if last_with_default_node is not None:
                    insert_line = last_with_default_line + 1
                    indent = self._get_indent_from_ast_node(last_with_default_node)
                elif first_with_default_node is not None:
                    # 这种情况不应该发生，但为了安全起见
                    insert_line = first_with_default_line
                    indent = self._get_indent_from_ast_node(first_with_default_node)
                else:
                    # 子类没有有默认值的属性，插入到类体开始（或最后一个无默认值属性之后）
                    if last_without_default_node is not None:
                        insert_line = last_without_default_line + 1
                        indent = self._get_indent_from_ast_node(last_without_default_node)
                    else:
                        insert_line = self._find_insertion_point(info['node'])
                        if insert_line is None:
                            continue
                        indent = self._calculate_class_body_indent(class_node)

                new_attrs_code = self._generate_attrs_code(new_attrs_with_default)
                modifications.append((insert_line, new_attrs_code, indent))

        # 应用修改（从后往前插入，避免行号变化）
        if not modifications:
            return self.source_code

        # 按行号从大到小排序，从后往前插入避免行号变化
        modifications.sort(key = lambda x: x[0], reverse = True)
        new_lines = self.lines[:]

        for line_num, new_code, indent in modifications:

            # 为每一行添加缩进
            indented_lines = []
            for line in new_code.split('\n'):
                if line.strip():  # 非空行
                    indented_lines.append(indent + line)
                else:
                    indented_lines.append('')

            # 插入新代码
            new_lines.insert(line_num, '\n'.join(indented_lines))

        # 检查是否需要添加默认值需要的导入
        optimized_code = '\n'.join(new_lines)

        # 收集所有生成的属性中默认值需要的导入
        imports_needed = set()  # Set[tuple]: (module, name, import_type)
        existing_imports = self._get_existing_imports()
        name_to_import = self._get_name_to_import_info()

        # 从 missing_attrs 中提取所有有默认值的属性的导入需求
        for class_name, info in self.class_info.items():
            if not info.get('has_slots') or not info.get('parents'):
                continue

            # 获取该类要添加的属性（missing_attrs）
            existing_attrs = {
                        attr_info[0]
                        for attr_info in info['attrs']
                        if isinstance(attr_info, (list, tuple)) and len(attr_info) >= 1 and isinstance(attr_info[0], str)
            }

            inherited_attrs = {}
            # 收集所有父类的导入映射
            all_parent_imports = {}
            for parent in info['parents']:
                parent_attrs, parent_name_to_import = self._collect_inherited_attrs(parent, visited = set())
                # 合并父类的导入映射
                all_parent_imports.update(parent_name_to_import)
                for attr, attr_info in parent_attrs.items():
                    if attr in existing_attrs:
                        continue
                    if attr in inherited_attrs:
                        continue
                    inherited_attrs[attr] = attr_info
            # 更新全局的导入映射
            name_to_import.update(all_parent_imports)

            # 从 inherited_attrs 中提取有默认值的属性的导入需求
            for attr, attr_info in inherited_attrs.items():
                attr_type, has_default, default_value_str = self._parse_attr_info(attr_info)
                if has_default and default_value_str and default_value_str not in ('None', 'True', 'False'):
                    # 尝试解析默认值字符串提取导入
                    try:
                        default_ast = ast.parse(default_value_str, mode = 'eval')
                        imports_needed.update(
                                    self._extract_imports_from_default_value(default_ast.body, existing_imports, name_to_import))
                    except:
                        # 如果解析失败，尝试简单的字符串匹配
                        if '.' in default_value_str:
                            # 模块.类 形式
                            parts = default_value_str.split('.')
                            if len(parts) >= 2:
                                module_name = parts[0]
                                if module_name not in existing_imports and module_name not in ('dict', 'list', 'tuple', 'set',
                                                                                               'frozenset', 'str', 'int', 'float',
                                                                                               'bool'):
                                    imports_needed.add((module_name, parts[-1], 'from'))
                        elif default_value_str and not default_value_str[0].isdigit() and default_value_str[0] not in ("'", '"', '['):
                            # 简单的名称，可能是常量或类，查找其导入信息
                            if default_value_str not in existing_imports and default_value_str not in ('dict', 'list', 'tuple', 'set',
                                                                                                       'frozenset', 'str', 'int',
                                                                                                       'float', 'bool', 'object'):
                                if default_value_str in name_to_import:
                                    module, import_type = name_to_import[default_value_str]
                                    imports_needed.add((module, default_value_str, import_type))
                                else:
                                    imports_needed.add((None, default_value_str, 'import'))

        # 添加必要的导入

        if imports_needed:
            optimized_code = self._add_default_value_imports(optimized_code, imports_needed)

        return optimized_code

    def _collect_inherited_attrs(self, class_name: str, visited: Optional[Set[str]] = None) -> tuple:
        """收集所有父类属性（递归，支持跨文件），返回 (attrs_dict, name_to_import_dict)
        attrs_dict: {attr_name: (attr_type, has_default, default_value_str)}
        name_to_import_dict: {name: (module, import_type)} 父类文件的导入映射
        """
        if visited is None:
            visited = set()

        if class_name in visited:
            return {}, {}

        visited.add(class_name)

        # 先在当前文件中查找
        if class_name in self.class_info:
            info = self.class_info[class_name]
            attrs = {}
            name_to_import = self._get_name_to_import_info()

            # 收集当前类的属性
            for attr_info in info['attrs']:
                attr_name = attr_info[0]
                attr_type, has_default, default_value_str = self._parse_attr_info(attr_info[1 :])
                attrs[attr_name] = (attr_type, has_default, default_value_str)

            # 递归收集父类属性
            for parent in info['parents']:
                parent_attrs, parent_name_to_import = self._collect_inherited_attrs(parent, visited)
                name_to_import.update(parent_name_to_import)
                for attr, attr_info in parent_attrs.items():
                    if attr not in attrs:
                        attrs[attr] = attr_info

            return attrs, name_to_import

        # 如果当前文件中没有，尝试跨文件查找
        parent_imports = self._find_parent_import_in_tree(class_name)
        if parent_imports:
            # 尝试从导入的模块中解析
            # 如果父类不在源代码中（外部库），静默返回空字典
            try:
                parent_attrs, parent_name_to_import = self._load_parent_from_module(class_name, parent_imports)
                if parent_attrs:
                    return parent_attrs, parent_name_to_import
            except RuntimeError:
                # 父类文件不在源代码中，跳过（可能是外部库）
                return {}, {}

        return {}, {}

    def _find_parent_import_in_tree(self, class_name: str) -> Optional[str]:
        """在当前文件的 AST 中查找父类的导入"""
        for class_name_key, info in self.class_info.items():
            if 'parent_imports' in info:
                if class_name in info['parent_imports']:
                    return info['parent_imports'][class_name]
        return None

    def _load_parent_from_module(self, class_name: str, module_path: str) -> tuple:
        """从模块文件中加载父类属性"""
        if not self.file_path:
            raise RuntimeError(f"无法加载父类 {class_name}：当前文件路径未设置")

        # 步骤1: 检查是否能读取父类文件
        module_parts = module_path.split('.')
        if not module_parts:
            raise RuntimeError(f"无法加载父类 {class_name}：模块路径 {module_path} 无效")

        # 查找文件：从项目根目录查找
        # 使用绝对路径来查找项目根目录
        base_dir = self.file_path.resolve().parent

        # 找到项目根目录（包含 src 目录的目录）
        current = base_dir
        project_root = None
        while current.parent != current:
            if (current / 'src').exists() and (current / 'src').is_dir():
                project_root = current
                break
            current = current.parent

        if not project_root:
            # 如果找不到项目根目录，尝试从当前文件目录查找
            # 再向上查找一级（因为当前文件可能在 src 的子目录中）
            if (base_dir.parent / 'src').exists():
                project_root = base_dir.parent
            else:
                project_root = base_dir

        # 构建可能的文件路径（从项目根目录开始）
        # 注意：module_parts 可能包含 'src'，需要判断是否从项目根目录开始
        possible_paths = []

        # 检查 module_parts 是否以 'src' 开头
        if module_parts and module_parts[0] == 'src':
            # 如果模块路径从 'src' 开始，直接使用
            # 方式1: src/strategy/strg_base_engine.py
            if len(module_parts) > 1:
                possible_paths.append(project_root / '/'.join(module_parts[:-1]) / (module_parts[-1] + '.py'))
            # 方式2: src/strategy/strg_base_engine/__init__.py
            possible_paths.append(project_root / '/'.join(module_parts) / '__init__.py')
        else:
            # 如果模块路径不是从 'src' 开始，需要添加 'src'
            # 方式1: src/{module_path}.py
            possible_paths.append(project_root / 'src' / '/'.join(module_parts[:-1]) / (module_parts[-1] + '.py'))
            # 方式2: src/{module_path}/__init__.py
            possible_paths.append(project_root / 'src' / '/'.join(module_parts) / '__init__.py')

        # 第一步：检查是否能读取父类文件
        file_path = None
        for path in possible_paths:
            if path.exists() and path.is_file():
                file_path = path
                break

        if file_path is None:
            # 父类文件不在源代码中（可能是外部库），静默返回空元组
            return {}, {}

        # 第二步：读取文件
        try:
            source_code = file_path.read_text(encoding = 'utf-8')
        except Exception as e:
            raise RuntimeError(f"第一步失败：读取父类文件失败\n"
                               f"  文件: {file_path}\n"
                               f"  错误: {e}") from e

        # 第三步：解析文件
        try:
            optimizer = SlotsOptimizer(source_code, file_path)
        except Exception as e:
            raise RuntimeError(f"第一步失败：解析父类文件失败\n"
                               f"  文件: {file_path}\n"
                               f"  错误: {e}") from e

        # 第四步：检查是否找到目标类
        if class_name not in optimizer.class_info:
            raise RuntimeError(f"第一步失败：在文件中未找到父类\n"
                               f"  文件: {file_path}\n"
                               f"  查找的类: {class_name}\n"
                               f"  文件中找到的类: {list(optimizer.class_info.keys())}")

        info = optimizer.class_info[class_name]

        # 第五步：检查是否是 @dataclass
        if not info.get('has_dataclass'):
            raise RuntimeError(f"第一步失败：父类没有 @dataclass 装饰器\n"
                               f"  文件: {file_path}\n"
                               f"  类: {class_name}\n"
                               f"  has_dataclass: {info.get('has_dataclass')}, has_slots: {info.get('has_slots')}")

        # 第二步：检查是否能读取到属性
        attrs = {}
        for attr_info in info['attrs']:
            if len(attr_info) >= 4:
                attr_name, attr_type, has_default, default_value_str = attr_info[: 4]
            elif len(attr_info) == 3:
                attr_name, attr_type, has_default = attr_info
                default_value_str = None
            else:
                attr_name, attr_type = attr_info[: 2]
                has_default = False
                default_value_str = None
            attrs[attr_name] = (attr_type, has_default, default_value_str)

        # 递归收集父类属性（如果有父类）
        parent_name_to_import = optimizer._get_name_to_import_info()
        for parent in info['parents']:
            parent_attrs, parent_imports = optimizer._collect_inherited_attrs(parent, set())
            parent_name_to_import.update(parent_imports)
            for attr, attr_info in parent_attrs.items():
                if attr not in attrs:
                    attrs[attr] = attr_info

        if not attrs:
            raise RuntimeError(f"第二步失败：未读取到任何属性\n"
                               f"  文件: {file_path}\n"
                               f"  类: {class_name}\n"
                               f"  类的属性数量: {len(info.get('attrs', []))}")

        return attrs, parent_name_to_import

    def _analyze_child_attrs_order(self, class_node: ast.ClassDef) -> Dict:
        """分析子类已有属性的顺序，返回第一个和最后一个有默认值属性的AST节点和行号，以及最后一个无默认值属性的AST节点和行号"""
        first_with_default_node = None
        first_with_default_line = None
        last_with_default_node = None
        last_with_default_line = None
        last_without_default_node = None
        last_without_default_line = None

        for item in class_node.body:
            if isinstance(item, ast.AnnAssign):
                # 检查是否有默认值
                has_default = item.value is not None
                line_num = item.lineno - 1  # 转换为0索引

                if has_default:
                    if first_with_default_node is None:
                        first_with_default_node = item
                        first_with_default_line = line_num
                    last_with_default_node = item
                    last_with_default_line = line_num
                else:
                    # 无默认值的属性
                    last_without_default_node = item
                    last_without_default_line = line_num

        return {
                    'first_with_default_node': first_with_default_node,
                    'first_with_default_line': first_with_default_line,
                    'last_with_default_node': last_with_default_node,
                    'last_with_default_line': last_with_default_line,
                    'last_without_default_node': last_without_default_node,
                    'last_without_default_line': last_without_default_line,
                    'class_node': class_node
        }

    def _find_insertion_point(self, class_node: ast.ClassDef) -> Optional[int]:
        """找到类体中插入属性的位置（在文档字符串之后，第一个成员之前）"""
        # 类定义行号（从1开始）
        class_line = class_node.lineno - 1  # 转换为0索引

        # 查找类体开始位置（跳过文档字符串）
        body_start_line = None

        # 先检查类体中是否有属性定义
        for item in class_node.body:
            if isinstance(item, ast.AnnAssign):
                # 找到第一个属性定义，在其之前插入
                body_start_line = item.lineno - 1
                break

        if body_start_line is None:
            # 没有属性定义，查找第一个非文档字符串的内容
            # 属性应该插入在类的第一个内容之前（包括装饰器）
            first_item_line = None
            for item in class_node.body:
                if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                    # 文档字符串，跳过
                    continue
                else:
                    # 找到第一个非文档字符串的内容
                    first_item_line = item.lineno - 1
                    break

            if first_item_line is not None:
                # 从第一个内容往前查找，找到类体开始位置
                # 跳过空行和装饰器，属性应该插入在第一个装饰器或方法之前
                body_start_line = first_item_line
                for i in range(first_item_line - 1, class_line, -1):
                    line = self.lines[i].strip()
                    if line.startswith('@'):
                        # 装饰器，属性应该在装饰器之前
                        body_start_line = i
                    elif line == '':
                        # 空行，可以插入在这里
                        body_start_line = i + 1
                        break
                    else:
                        # 非空、非装饰器的行，停止
                        break

        if body_start_line is None:
            # 如果类体为空或只有文档字符串，在类定义后的第一个非空行插入
            for i in range(class_line + 1, len(self.lines)):
                line = self.lines[i].strip()
                if line.startswith('class '):
                    body_start_line = i
                    break
                elif line:
                    # 找到第一个非空行
                    body_start_line = i
                    break
            else:
                # 没有找到，在类定义后插入
                body_start_line = class_line + 1

        return body_start_line

    def _generate_attrs_code(self, attrs: Dict[str, tuple]) -> str:
        """生成属性代码（按正确顺序：无默认值的在前，有默认值的在后）"""
        if not attrs:
            return ''

        # 分离有默认值和无默认值的属性
        attrs_without_default = []
        attrs_with_default = []

        for attr_name, attr_info in attrs.items():
            attr_type, has_default, default_value_str = self._parse_attr_info(attr_info)
            if has_default:
                attrs_with_default.append((attr_name, attr_type, default_value_str))
            else:
                attrs_without_default.append((attr_name, attr_type))

        # 排序（保持稳定顺序）
        attrs_without_default.sort(key = lambda x: x[0])
        attrs_with_default.sort(key = lambda x: x[0])

        lines = [
                    f"# Auto-generated from parent classes: {', '.join(sorted(attrs.keys()))}",
                    "# This optimization copies parent class attributes to child class slots",
                    "# to improve attribute access performance when using @define(slots=True) or @dataclass(slots=True)"
        ]

        # 先添加无默认值的属性
        # 为防止导入循环，所有父类属性统一使用 any 类型
        for attr_name, attr_type in attrs_without_default:
            lines.append(f"{attr_name}: any")

        # 再添加有默认值的属性（包含默认值，以确保参数顺序正确）
        # 为防止导入循环，所有父类属性统一使用 any 类型
        for attr_name, attr_type, default_value_str in attrs_with_default:
            if default_value_str:
                lines.append(f"{attr_name}: any = {default_value_str}")
            else:
                lines.append(f"{attr_name}: any = None")

        return '\n'.join(lines)

    def _get_existing_imports(self) -> Set[str]:
        """获取当前文件中已有的导入名称"""
        imports = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.asname if alias.asname else alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                for alias in node.names:
                    imports.add(alias.asname if alias.asname else alias.name)
        return imports

    def _add_default_value_imports(self, code: str, imports_needed: Set[tuple]) -> str:
        """添加默认值需要的导入
        imports_needed: Set[tuple] where tuple is (module, name, import_type)
        """
        if not imports_needed:
            return code

        lines = code.split('\n')
        existing_imports = self._get_existing_imports()

        # 过滤掉已经存在的导入并生成导入语句
        imports_to_add = []
        for module, name, import_type in imports_needed:
            if name not in existing_imports:
                if import_type == 'from' and module:
                    imports_to_add.append((module, name, 'from'))
                else:
                    imports_to_add.append((name, name, 'import'))

        if not imports_to_add:
            return code

        # 查找导入语句的位置
        import_pos = None
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_pos = i
                break

        if import_pos is None:
            # 没有导入语句，在文件开头添加
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip() and not (line.strip().startswith('"""') or line.strip().startswith("'''")
                                         or line.strip().startswith('#')):
                    insert_pos = i
                    break
            # 添加导入语句，先添加 from 导入，再添加 import 导入
            from_imports = [(m, n) for m, n, t in imports_to_add if t == 'from']
            import_imports = [(m, n) for m, n, t in imports_to_add if t == 'import']

            for module, name in sorted(from_imports):
                lines.insert(insert_pos, f"from {module} import {name}")
                insert_pos += 1
            for module, name in sorted(import_imports):
                lines.insert(insert_pos, f"import {name}")
                insert_pos += 1
        else:
            # 在现有导入后添加
            for module, name, import_type in sorted(imports_to_add):
                if import_type == 'from':
                    lines.insert(import_pos + 1, f"from {module} import {name}")
                else:
                    lines.insert(import_pos + 1, f"import {name}")
                import_pos += 1

        return '\n'.join(lines)

    def _get_indent(self, line: str) -> str:
        """获取行的缩进"""
        match = re.match(r'^(\s*)', line)
        return match.group(1) if match else ''

    def _get_indent_from_ast_node(self, node: ast.AST, fallback_class_node: Optional[ast.ClassDef] = None) -> str:
        """从AST节点获取缩进：读取源代码中对应行的缩进"""
        if node is None:
            # 如果节点为None，尝试从fallback类节点计算缩进
            if fallback_class_node is not None:
                return self._calculate_class_body_indent(fallback_class_node)
            return '    '  # 最后的fallback

        line_num = node.lineno - 1  # 转换为0索引
        if 0 <= line_num < len(self.lines):
            line = self.lines[line_num]
            indent = self._get_indent(line)
            if indent:
                return indent

        # 如果找不到，尝试从类节点计算（如果提供了）
        if fallback_class_node is not None:
            return self._calculate_class_body_indent(fallback_class_node)

        # 最后的fallback：从当前节点的父类节点计算
        return '    '

    def _calculate_class_body_indent(self, class_node: ast.ClassDef) -> str:
        """计算类体的缩进：类体应该比类定义多一层缩进"""
        # 获取类定义行的缩进
        class_line_num = class_node.lineno - 1
        if 0 <= class_line_num < len(self.lines):
            class_line = self.lines[class_line_num]
            class_indent = self._get_indent(class_line)
            if class_indent:
                # 类体缩进 = 类定义缩进 + 一级缩进（通常是4个空格或1个tab）
                # 从类体中第一个属性或方法获取缩进增量
                for item in class_node.body:
                    item_line_num = item.lineno - 1
                    if 0 <= item_line_num < len(self.lines):
                        item_line = self.lines[item_line_num]
                        item_indent = self._get_indent(item_line)
                        if item_indent:
                            # 计算缩进增量
                            indent_increment = item_indent[len(class_indent):] if item_indent.startswith(class_indent) else ''
                            if indent_increment:
                                return class_indent + indent_increment

                # 如果找不到类体内容，从标准缩进推断
                # 先检查类定义行是否有tab
                if '\t' in class_indent:
                    return class_indent + '\t'
                else:
                    # 尝试从类体附近的行检测实际的缩进增量
                    for i in range(class_line_num + 1, min(class_line_num + 20, len(self.lines))):
                        if i < len(self.lines) and self.lines[i].strip():
                            line_indent = self._get_indent(self.lines[i])
                            if line_indent and line_indent.startswith(class_indent):
                                increment = line_indent[len(class_indent):]
                                if increment in ('  ', '    ', '\t'):
                                    return class_indent + increment
                    # 如果检测不到，默认使用4个空格
                    return class_indent + '    '

        # fallback：返回标准缩进
        return '    '


def slots_src(source_code: str, file_path: Optional[Path] = None) -> str:
    """在内存中优化代码，返回优化后的代码字符串（不修改文件）
    
    Args:
        source_code: 源代码字符串
        file_path: 可选的文件路径（用于错误报告）
    
    Returns:
        优化后的代码字符串，如果没有修改则返回原始代码
    """
    optimizer = SlotsOptimizer(source_code, file_path)
    return optimizer.optimize()


def main():
    """命令行入口：打印优化后的代码"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nuitka_slots_optimizer.py <file>")
        print("\nExample:")
        print("  python nuitka_slots_optimizer.py src/my_module.py")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: {input_path} does not exist", file = sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: {input_path} is not a file", file = sys.stderr)
        sys.exit(1)

    source_code = input_path.read_text(encoding = 'utf-8')
    optimized_code = slots_src(source_code, input_path)

    # 直接打印优化后的代码
    print(optimized_code, end = '')


if __name__ == "__main__":
    main()
