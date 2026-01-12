from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Set, DefaultDict, Optional, Callable

# from px_util.profile_util import profile
from py_hoist.multi_items import MultiItems

from .hoist_transformer import CodeAssignStmtInfo, AssignStmtInfo, AttrUsageInfo, transform_code_tree, AttrUsedInType
from .hoist_ast_util import (attr_chain_text_ast__, build_parent_map__, get_up_loop_node__, is_type_annotation__, walk_no_nested__,
                             find_stmt_ancestor__, get_if_info__, is_in_first_logical_path__, get_func_args__,
                             find_prev_upward_no_map__, find_base_def_no_map__, get_with_conflicts__, find_stmt_list_path__,
                             safe_unparse__, get_subattr_for_chain__, is_in_first_operand_of_boolop__, get_attr_name_and_pos__,
                             add_line_numbers__, sanitize_ident__, split_expr_token_to_items__, loop_has_prior_definition__,
                             find_name_ident_used_or_new__, get_loop_assign_attribute_names__, get_up_comprehension_expr__,
                             split_base_ident__)

# Debug flag for development diagnostics. Set to True to enable verbose
# search/creation logging for hoist decision debugging.
# Debug flag removed — debug prints eliminated in production code.

# from loguru_log import get_logger

# logger = get_logger(__name__)


def find_assign_node_func(target_name: str, from_lineno: int, top_lineno: int,
                          code_stmt_info_map: Dict[ast.Assign, CodeAssignStmtInfo]) -> Optional[CodeAssignStmtInfo]:
    for ast_asign, codeAssignStmtInfo in reversed(code_stmt_info_map.items()):
        if codeAssignStmtInfo.stmt_lineno > from_lineno:
            continue
        if codeAssignStmtInfo.stmt_lineno < top_lineno:
            break

        if target_name in codeAssignStmtInfo.assign_names:
            return codeAssignStmtInfo


#@profile
def _attr_usage_info_to_assignStmtInfo(
            depth: int,
            tree: ast.Module,
            func: ast.FunctionDef,
            func_parent_map: Dict[ast.AST, ast.AST],
            code_stmt_info_map: Dict[str, CodeAssignStmtInfo],
            attr_usage_infos: list[AttrUsageInfo],
            assign_stmt_info_map: Dict[ast.stmt, AssignStmtInfo],
            assignStmtInfo: AssignStmtInfo = None,
            attr_usage_info: AttrUsageInfo = None,
            owner_stmt: ast.Assign = None,
) -> Dict[ast.AST, ast.AST]:
    for attr_usage_info in attr_usage_infos:

        # logger.info(f"ATTR-BASE:  prop={ast.unparse(attr_node)}， base={ast.unparse(base_def) if base_def is not None else 'None'}， src_line={src_lines[ln - 1]}"
        #             )
        # 我们只关注属性链的倒数第二层（例如 main_layout.goods.addWidget -> main_layout.goods）

        # 根据 hoist_sub_attr_node 查找变量定义点（如 Assign RHS 精确等于 a.b）
        exists_assign_stmt = None
        target_text = ast.unparse(attr_usage_info.hoist_sub_attr_node)
        owner_stmt = find_stmt_ancestor__(attr_usage_info.attr_node, func_parent_map)
        assignStmtInfo = None
        if owner_stmt is None:
            continue

        def _pred(cand: ast.AST):
            if isinstance(cand, ast.Assign):
                # 仅匹配 RHS 为 Attribute（非 Call）的赋值
                return isinstance(cand.value, ast.Attribute) and ast.unparse(cand.value) == target_text
            return False

        # 先向上查找 base 的赋值语句（例如寻找 base = ...），找到就登记并跳到下一个属性；其它逻辑不变

        def _pred_base(cand: ast.AST):
            if isinstance(cand, ast.Assign):
                for t in cand.targets:
                    t_unp = safe_unparse__(t)
                    if t_unp == attr_usage_info.chain_parts.keys()[0]:
                        return True
            return False

        base_stmt = find_prev_upward_no_map__(
                    start_stmt = owner_stmt,
                    func = func,
                    func_parent_map = func_parent_map,
                    predicate = _pred_base,
        )
        base_line = base_stmt.lineno if base_stmt is not None else -1

        #先查找 base_def 是否有对应的赋值语句，base_def找出来
        exists_assign_stmt: ast.Assign = find_prev_upward_no_map__(start_stmt = owner_stmt,
                                                                   func = func,
                                                                   func_parent_map = func_parent_map,
                                                                   predicate = _pred)
        if exists_assign_stmt:
            find_base_re_assigned = find_assign_node_func(target_name = attr_usage_info.chain_parts.keys()[0],
                                                          from_lineno = owner_stmt.lineno,
                                                          top_lineno = exists_assign_stmt.lineno,
                                                          code_stmt_info_map = code_stmt_info_map)

            if find_base_re_assigned:
                attr_usage_info.base_loop_re_assigned = True
            else:
                find_var_re_assigned = find_assign_node_func(target_name = attr_usage_info.chain_2_text,
                                                             from_lineno = owner_stmt.lineno,
                                                             top_lineno = exists_assign_stmt.lineno,
                                                             code_stmt_info_map = code_stmt_info_map)
                if find_var_re_assigned:
                    attr_usage_info.base_loop_re_assigned = True
                elif exists_assign_stmt.lineno > base_line:
                    gen_by_orig_code = exists_assign_stmt in code_stmt_info_map  # True when stmt exists in original code
                    # Extract the hoist target name (LHS) when the Assign already exists in source.
                    first_t = exists_assign_stmt.targets[0]
                    if isinstance(first_t, ast.Name):
                        var_name = first_t.id
                    else:
                        var_name = safe_unparse__(first_t)

                    assignStmtInfo = assign_stmt_info_map[exists_assign_stmt]
                    assignStmtInfo.assign_node = exists_assign_stmt
                    assignStmtInfo.gen_by_orig_code = gen_by_orig_code
                    assignStmtInfo.attr_infos.append(attr_usage_info)
                    # propagate with-conflict usage
                    assignStmtInfo.use_by_with_conflict = assignStmtInfo.use_by_with_conflict or attr_usage_info.under_with_conflict
                    assignStmtInfo.bar_name = var_name
                    assignStmtInfo.chain_2_value = attr_usage_info.chain_2_text
                    assignStmtInfo.chain_text = attr_usage_info.chain_text
                    attr_usage_info.assignStmtInfo = assignStmtInfo

                else:
                    exists_assign_stmt = None  # 找到的赋值语句在 base_def 之前，视为无效，继续创建新赋值

        if exists_assign_stmt is None:
            create_outside_loop = False
            # 没有找到现有的 Assign 定义：需要创建一个合适位置的临时赋值语句并记录
            # 策略：
            # - 如果当前使用点在循环内，而且 base_def 存在且位于循环外，
            #   则把创建点放在 base_def 所在的语句处（循环外）；
            # - 否则在 parent_stmt 之前创建（与使用点同一作用域/前置位置）。
            insert_point = None

            # if attr_usage_info.chain_text == "df[column].to_numpy":
            #     print(f"xxxxxxxxxxxxxxxxxxxxx")

            loop_node = get_up_loop_node__(node = owner_stmt, func_parent_map = func_parent_map)

            comp_expr: ast.AST = None
            if loop_node is None:
                comp_expr = get_up_comprehension_expr__(attr_usage_info.attr_node, func_parent_map)

            if (loop_node or comp_expr) and not attr_usage_info.base_loop_re_assigned:  # 在循环内使用
                loop_node_has_base = False

                base_names: set[str] = {attr_usage_info.chain_parts.keys()[0]}

                k0 = attr_usage_info.chain_parts.keys()[0].split('.')[0]
                base_names.update(split_base_ident__(k0))

                if attr_usage_info.chain_parts.values()[1].startswith('['):
                    base_names.add(attr_usage_info.chain_parts.keys()[1])

                # If the base name appears in the loop target (for a header like
                # `for attr_name, attr_value in ...`) treat it as present in-loop.
                loop_target_names: set[str] = set()
                if isinstance(loop_node, (ast.For, getattr(ast, 'AsyncFor', type(None)))):
                    ast_names = get_loop_assign_attribute_names__(loop_node.target)
                    loop_target_names.update(ast_names)
                elif comp_expr is not None:
                    # comp_expr is a comprehension expression; collect targets
                    for gen in getattr(comp_expr, 'generators', []):
                        ast_names = get_loop_assign_attribute_names__(gen.target)
                        loop_target_names.update(ast_names)

                # Consider prior definitions in body and also loop-target names
                loop_node_has_base = bool(base_names & loop_target_names)
                loop_node_has_base = loop_node_has_base or loop_has_prior_definition__(func, loop_node or comp_expr, base_names)

                if not loop_node_has_base:
                    for _sub_node in ast.walk(attr_usage_info.hoist_sub_attr_node):
                        if isinstance(_sub_node, ast.Name):
                            name_to_check = _sub_node.id
                            # search recorded assign infos for a matching name
                            for _key, info in reversed(code_stmt_info_map.items()):
                                if name_to_check in info.assign_names and info.assign_in_loop:
                                    loop_node_has_base = True
                                    break
                            if loop_node_has_base:
                                break

                if not loop_node_has_base:
                    if attr_usage_info.base_def is not None:
                        base_def_stmt = find_stmt_ancestor__(attr_usage_info.base_def, func_parent_map)
                        if base_def_stmt is not None:
                            # base_def is outside this loop.
                            if loop_node and not get_up_loop_node__(node = base_def_stmt, func_parent_map = func_parent_map):
                                insert_point = loop_node
                                create_outside_loop = True
                            elif comp_expr and not get_up_comprehension_expr__(base_def_stmt, func_parent_map):
                                # For comprehension expressions, insert at the containing
                                # statement (e.g. the Assign that holds the comp_expr),
                                # not at the expression node itself.
                                insert_point = find_stmt_ancestor__(comp_expr, func_parent_map) or owner_stmt
                                create_outside_loop = True
                            else:
                                insert_point = base_def_stmt

                    else:
                        # no base_def available — insert before the loop/comprehension (create outside)
                        create_outside_loop = True
                        if loop_node:
                            insert_point = loop_node
                        elif comp_expr:
                            insert_point = find_stmt_ancestor__(comp_expr, func_parent_map)
                        else:
                            insert_point = owner_stmt
                else:
                    if comp_expr is not None:
                        return func_parent_map  # 无法提升

            if insert_point is None:
                insert_point = owner_stmt

            # 创建新的 Assign AST 节点： var_name = <hoist_sub_attr_expr>
            new_value = ast.copy_location(copy.deepcopy(attr_usage_info.hoist_sub_attr_node), attr_usage_info.hoist_sub_attr_node)
            new_name = ast.copy_location(ast.Name(id = attr_usage_info.calc_var_name, ctx = ast.Store()),
                                         attr_usage_info.hoist_sub_attr_node)
            new_assign = ast.Assign(targets = [new_name], value = new_value)
            # 把位置信息复制到 Assign 上，使用 insert_point 的位置以便插入语句能获得正确的行列信息
            ast.copy_location(new_assign, insert_point)
            # 确保 lineno/col_offset 字段可用（有时 copy_location 可能不完全设置），直接赋值以保证稳定性
            new_assign.lineno = insert_point.lineno
            new_assign.col_offset = insert_point.col_offset

            # 立即把新创建的 Assign 插入 AST（直接在 insert_point 位置，不作额外保护）
            path = find_stmt_list_path__(func, insert_point)
            if not path:
                # fallback to owner_stmt's location within the function
                path = find_stmt_list_path__(func, owner_stmt) if owner_stmt is not None else []
            if not path:
                # unable to locate an insertion point; skip creating this assign
                continue
            list_ref, idx, parent = path[0]
            list_ref.insert(idx, new_assign)

            # 插入后更新 func_parent_map
            func_parent_map = build_parent_map__(tree)

            assignStmtInfo = assign_stmt_info_map[new_assign]
            assignStmtInfo.assign_node = new_assign
            assignStmtInfo.attr_infos.append(attr_usage_info)
            assignStmtInfo.use_by_with_conflict = assignStmtInfo.use_by_with_conflict or attr_usage_info.under_with_conflict
            assignStmtInfo.create_outside_loop = create_outside_loop
            assignStmtInfo.bar_name = attr_usage_info.calc_var_name
            assignStmtInfo.chain_2_value = attr_usage_info.chain_2_text
            assignStmtInfo.chain_text = attr_usage_info.chain_text
            attr_usage_info.assignStmtInfo = assignStmtInfo
    return func_parent_map


#@profile
def _process_Attribute(
            depth: int,
            func: ast.FunctionDef,
            attr_node:ast.Attribute,
            chain_text :str,
            func_args: dict[str, ast.arg] ,
            func_parent_map: Dict[ast.AST, ast.AST],
            with_conflicts: Set[str],
            attr_usage_infos: list[AttrUsageInfo],
            in_index_attr_nodes: set[ast.Attribute],
            find_ident_func :Callable[[str,str],str],
            attr_usage_info: AttrUsageInfo = None,#因代码太长，以下变量只是为了在ide可以显示不同的颜色
            owner_stmt: ast.Assign = None, )   :

    # 跳过类型注解中的属性（如 list[pd.DataFrame] 中的 pd.DataFrame）
    if is_type_annotation__(attr_node, func_parent_map):
        return
    # 开发阶段：如果该属性恰好是一个 Assign 的 RHS（定义语句），
    # 则把它视为定义而非使用，提前过滤掉，避免后续把定义当成使用处理。
    owner_stmt = find_stmt_ancestor__(attr_node, func_parent_map)
    # 如果在 match/case 结构内部，则跳过处理（避免对 case 分支中的表达式做提升）
    if isinstance(owner_stmt, (getattr(ast, 'Match', type(None)), getattr(ast, 'match_case', type(None)))):
        return

    if isinstance(owner_stmt, ast.Assign):
        # 仅当 Assign 的 RHS 精确是该 Attribute 时过滤（避免误伤更复杂情况）
        if isinstance(owner_stmt.value, ast.Attribute) and ast.unparse(owner_stmt.value) == ast.unparse(attr_node):
            return

    used_in_loop = False
    if owner_stmt is not None:
        loop_node = get_up_loop_node__(node = attr_node, func_parent_map = func_parent_map)
        comp_node = get_up_comprehension_expr__(attr_node, func_parent_map)
        if loop_node is not None or comp_node is not None:
            used_in_loop = True
    # compute chain text and detect with-conflict
    under_with_conflict = chain_text in with_conflicts

    # if chain_text == 'symbol_position_map[pos_contract_symbol].append':
    #     print(f"chain_text: {chain_text}")

    # Decide precise usage context: BoolOp-first, If-test (first/other), Elif-test (first/other), or normal
    used_type = AttrUsedInType.IN_NORMAL
    if is_in_first_operand_of_boolop__(attr_node, func_parent_map):
        used_type = AttrUsedInType.IN_FIRST_BOOLOP
    else:
        # locate nearest If and whether we're in its test
        root_if, (immediate_if, branch_type) = get_if_info__(attr_node, func_parent_map)
        if immediate_if is not None and branch_type == 'test':
            is_first = is_in_first_logical_path__(attr_node, root_if, func_parent_map)
            parent_of_immediate = func_parent_map.get(immediate_if)
            # direct access: orelse is always present on ast.If
            if isinstance(parent_of_immediate, ast.If) and immediate_if in parent_of_immediate.orelse:
                used_type = AttrUsedInType.IN_ELIF_TEST_FIRST if is_first else AttrUsedInType.IN_ELIF_TEST_OTHER
            else:
                used_type = AttrUsedInType.IN_IF_TEST_FIRST if is_first else AttrUsedInType.IN_IF_TEST_OTHER

    attr_usage_info = AttrUsageInfo(used_in_loop = used_in_loop,
                                    attr_node = attr_node,
                                    used_type = used_type,
                                    under_with_conflict = under_with_conflict)
    attr_usage_info.base_def = find_base_def_no_map__(attr = attr_node,
                                                      func = func,
                                                      func_parent_map = func_parent_map,
                                                      func_args = func_args)
    attr_usage_infos.append(attr_usage_info)
    if attr_node in in_index_attr_nodes:
        attr_usage_info.in_index = True

    attr_usage_info.chain_text = chain_text
    attr_usage_info.chain_parts = split_expr_token_to_items__(chain_text)
    # 仅当链中存在至少一个"."（即有后续属性）时才继续，否则跳过
    len_parts = len(attr_usage_info.chain_parts)
    if len_parts <= 1:
        return

    attr_usage_info.base_text = attr_usage_info.chain_parts.keys()[0]
    attr_usage_info.has_more3_chains = (len_parts > 2)

    # 只提升属性链的前两段（如 a.b.c -> 提升 a.b，留下 c 供递归）
    attr_usage_info.chain_2_text = ''.join(attr_usage_info.chain_parts.values()[: 2])
    attr_usage_info.hoist_sub_attr_node = get_subattr_for_chain__(attr_node, attr_usage_info.chain_2_text)

    # 生成唯一的临时变量名（基于前两段并把 "." 换成 "_"）

    sub_ident_0 = attr_usage_info.chain_parts.keys()[0]
    if not sub_ident_0.isidentifier():
        new_sub_ident = sanitize_ident__(sub_ident_0)
        sub_ident_0 = find_ident_func(sub_ident_0, new_sub_ident)

    sub_ident_1 = sanitize_ident__(attr_usage_info.chain_parts.keys()[1])
    attr_usage_info.calc_var_name = sub_ident_0 + '_' + sub_ident_1
    # for readability.text_part_0


#@profile
def hoist_src(source: str,
              src_path:Path|str =None,
              depth: int = 0,
              code_stmt_info_map: dict[str, CodeAssignStmtInfo] =None,
              assignStmtInfo: AssignStmtInfo = None, #因代码太长，以下变量只是为了在ide可以显示不同的颜色
              attr_usage_info: AttrUsageInfo = None,
              codeAssignStmtInfo: CodeAssignStmtInfo = None,
              owner_stmt: ast.Assign = None,
              ) -> str:
    """递归的变量提升逻辑 —— 每次处理一轮，直到不再有新的提升。

    每次调用执行一轮处理：
    1. 解析源代码为新的 AST；
    2. 遍历并处理函数体，仅把符合条件的两层属性链（如 `self.x`）考虑提升；
    3. 为 LibCST 构建 `pos_map`（位置替换映射）和 `insert_map`（插入语句映射）；
    4. 使用 LibCST 应用修改以得到保留格式的新源代码；
                        var_name = '_'.join(parts[:2])
       则对新源代码递归调用 `transform_source_var` 继续下一轮。

    这样可以将每一轮的变化与后续轮次干净隔离，避免使用过期或不一致的引用。
    """
    try:
        tree: ast.Module = ast.parse(source)
    except SyntaxError as e:
        # Log a clear syntax error with location and print the full source for debugging
        # Use VS Code clickable format: File "path", line X
        print("----- SOURCE BEGIN -----\n" + add_line_numbers__(source) + "\n----- SOURCE END -----")
        if src_path:
            sp = str(src_path)
            print(f"Source parse failed (SyntaxError): {e.msg} at column {e.offset}")
            print(f'  File "{sp}", line {e.lineno}')
        else:
            print(f"Source parse failed (SyntaxError): {e.msg} at line {e.lineno}, column {e.offset}")

        raise
    except Exception as e:
        # Unexpected parsing error — log exception and source, then re-raise
        print("----- SOURCE BEGIN -----\n" + add_line_numbers__(source) + "\n----- SOURCE END -----")
        print(f"Unexpected error while parsing source: {e}")
        if src_path:
            sp = str(src_path)
            print(f'  File "{sp}", line 1')
        raise

    # 标记本轮是否有任何提升
    assign_stmt_info_map: Dict[ast.stmt, AssignStmtInfo] = DefaultDict(AssignStmtInfo)

    ###############
    src_lines = source.splitlines(keepends = True)
    index_attr_nodes: set[ast.Attribute] = set()  #node ->has_index_node

    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        func_args: dict[str, ast.arg] = get_func_args__(func)
        func_parent_map: Dict[ast.AST, ast.AST] = build_parent_map__(tree)  # 每个函数重建 func_parent_map
        with_conflicts: Set[str] = get_with_conflicts__(func)

        func_children: List[ast.AST] = list(walk_no_nested__(func))

        code_stmt_info_map: dict[ast.AST, CodeAssignStmtInfo] = {}

        attr_usage_infos: list[AttrUsageInfo] = []

        used_attr_node_map: dict[ast.Attribute, str] = {}
        in_index_attr_nodes: set[ast.Attribute] = set()  #node ->has_index_node

        used_name_ident_map: dict[str, str] = {}
        used_ident_counters: dict[str, int] = {}

        #查看变量名是否被使用过，没有就生成新的
        def find_ident_func(name: str, idnet_name: str) -> str:
            return find_name_ident_used_or_new__(name, idnet_name, used_name_ident_map, used_ident_counters)

        for attr_node in func_children:
            if isinstance(attr_node, ast.Assign):
                # Record assign-in-loop info keyed by the statement lineno (one or
                # more CodeAssignStmtInfo values may be stored under the same lineno)
                assign_node: ast.Assign = attr_node

                codeAssignStmtInfo = CodeAssignStmtInfo()
                code_stmt_info_map[assign_node] = codeAssignStmtInfo
                codeAssignStmtInfo.assign_node = assign_node
                codeAssignStmtInfo.stmt_lineno = assign_node.lineno
                codeAssignStmtInfo.assign_in_loop = get_up_loop_node__(node = assign_node,
                                                                       func_parent_map = func_parent_map) is not None
                # Create one CodeAssignStmtInfo per target name and store under
                # the statement lineno so callers that iterate by lineno can
                # inspect the associated entries.
                for target in assign_node.targets:
                    ast_names: list[str] = get_loop_assign_attribute_names__(target)
                    codeAssignStmtInfo.assign_names.update(ast_names)

            elif isinstance(attr_node, ast.Attribute) and isinstance(attr_node.ctx, ast.Load):
                #找出是否在[]内
                express_str = ast.unparse(attr_node)
                ### pos.contract.symbol找到 symbol_position_map[pos.contract.symbol].append，加入 index_attr_nodes中
                ###判断  node的外层是不是[]
                parent_node = func_parent_map.get(attr_node)
                chain_text = attr_chain_text_ast__(attr_node)
                chain_parts: MultiItems = split_expr_token_to_items__(chain_text)
                if len(chain_parts) > 1:
                    used_attr_node_map[attr_node] = chain_text
                    # 找最近的 Subscript 祖先（如果存在），并检查该 Subscript.slice 中是否包含深链属性（>=3 段）
                    subscript_ancestor = parent_node
                    while subscript_ancestor is not None and not isinstance(subscript_ancestor, ast.Subscript):
                        subscript_ancestor = func_parent_map.get(subscript_ancestor)

                    if isinstance(subscript_ancestor, ast.Subscript):
                        sub_slice = getattr(subscript_ancestor, 'slice', None)
                        if sub_slice is not None:
                            # 如果 slice 中存在任意深链属性（如 pos.contract.symbol），则把该 Subscript 标记为有深索引
                            for sub_attr in (n for n in ast.walk(sub_slice) if isinstance(n, ast.Attribute)):
                                sub_chain_text = attr_chain_text_ast__(sub_attr)
                                parts: MultiItems = split_expr_token_to_items__(sub_chain_text)
                                if len(parts) > 1:
                                    in_index_attr_nodes.add(attr_node)
                                    # 标记 Subscript 为深索引，同时标记对该 Subscript 的外层 Attribute（如 `.append`）
                                    subscript_ancestors_str = ast.unparse(subscript_ancestor)
                                    index_attr_nodes.add(subscript_ancestor)
                                    parent_of_sub = func_parent_map.get(subscript_ancestor)
                                    if isinstance(parent_of_sub, ast.Attribute):
                                        parent_of_sub_str = ast.unparse(parent_of_sub)
                                        index_attr_nodes.add(parent_of_sub)
                                    break

        ###############获取函数体内所有属性节点及其基础定义节点
        for attr_node, chain_text in used_attr_node_map.items():
            if attr_node in index_attr_nodes:
                continue

            _process_Attribute(depth = depth,
                               func = func,
                               attr_node = attr_node,
                               func_args = func_args,
                               chain_text = chain_text,
                               func_parent_map = func_parent_map,
                               with_conflicts = with_conflicts,
                               attr_usage_infos = attr_usage_infos,
                               in_index_attr_nodes = in_index_attr_nodes,
                               find_ident_func = find_ident_func)

        ############### 循环 attr_and_parent_def_map，看它的base_def 是否 code_assign_stmt_info_map，有的话，看它是否在循环内赋值 修改 base_re_assigned
        for attr_usage_info in attr_usage_infos:
            if attr_usage_info.used_in_loop:
                attr_usage_info.base_text = attr_usage_info.base_text
                # find any recorded assign info matching the base_text name
                name_to_check = attr_usage_info.base_text
                found_info = None
                for node, info in code_stmt_info_map.items():
                    if name_to_check in info.assign_names:
                        found_info = info
                        break
                if found_info is not None and found_info.assign_in_loop:
                    attr_usage_info.base_loop_re_assigned = True

        ############### 根据 属性查找它的定义，没有就生成
        func_parent_map = _attr_usage_info_to_assignStmtInfo(depth = depth,
                                                             tree = tree,
                                                             func = func,
                                                             func_parent_map = func_parent_map,
                                                             code_stmt_info_map = code_stmt_info_map,
                                                             attr_usage_infos = attr_usage_infos,
                                                             assign_stmt_info_map = assign_stmt_info_map)

    ############### 循环 assign_stmt_info_map，决定哪些可以提升，然后更新 pos_map 和 insert_map
    # 用于 LibCST 的映射
    pos_map: Dict[(int, int, int), AttrUsageInfo] = {}  # (lineno, col_offset, end_col,assign_idx,attr_idx) -> var name
    insert_map: Dict[Tuple[int, int, str], AssignStmtInfo] = {}  # (lineno, col_offset,assign_name) -> assign stmt text

    ############### 获得替换信息和插入信息
    # 标记是否需要继续递归（发现了多层链）
    attr_has_three_more_chains = False
    for assign_idx, (owner_stmt, assignStmtInfo) in enumerate(assign_stmt_info_map.items()):

        # 只有满足 can_hoist 的赋值才会被考虑
        if assignStmtInfo.can_replaced():
            # logger.info(f"Skipping hoist for assign statement: {assignStmtInfo.text} (cannot hoist)")

            # 为该 Assign 对应的每个属性创建 pos_map 条目，替换 sub-attr 为临时变量名
            for attr_idx, attr_usage_info in enumerate(assignStmtInfo.attr_infos):
                if attr_usage_info.under_with_conflict:
                    continue

                attr_usage_info.hoist_sub_attr_node = attr_usage_info.hoist_sub_attr_node
                # 使用 hoist_sub_attr_node 的位置作为替换的 start/end 基准，避免混合不同节点的位置
                full_chain_text = ast.unparse(attr_usage_info.hoist_sub_attr_node)
                klineno = attr_usage_info.hoist_sub_attr_node.lineno
                kcol_offset = attr_usage_info.hoist_sub_attr_node.col_offset
                line_text_local = src_lines[klineno - 1]

                # 优先使用 AST 提供的 end_col_offset（更稳健，能处理 f-string/格式化场景）
                # 回退策略：先尝试使用 get_attr_name_and_pos__ 获取更可靠的起始位置，
                # 然后在该源码行查找完整链文本；如果仍找不到，则在整个源码中寻找最接近的出现位置。
                (klineno2, kcol_offset2), _ = get_attr_name_and_pos__(attr_usage_info.hoist_sub_attr_node)

                pos_map[(klineno2, kcol_offset2, kcol_offset2 + len(full_chain_text))] = attr_usage_info
                attr_has_three_more_chains = attr_has_three_more_chains or attr_usage_info.has_more3_chains

                # Use centralized helper to determine attribute name and position

        if assignStmtInfo.need_insert():
            # 定位插入位置：在 owner_stmt 之前插入
            lineno = owner_stmt.lineno
            col_offset = owner_stmt.col_offset
            # 存储字符串形式，便于后续文本插入
            # Use the hoist assign AST stored on the AssignStmtInfo (not the owner stmt)
            insert_map[(lineno, col_offset, assignStmtInfo.bar_name)] = assignStmtInfo

    ############### 对pos_map排序，行号越大、列号越大 越靠前（降序）。
    # 支持 key 元素顺序可能不同（例如 (lineno,col,name) 或 (name,lineno,col)），
    # 从 key 中提取整数作为 line/col 并按降序排序。

    has_any_replce_or_insert, new_source = transform_code_tree(
                source = source,
                pos_map = pos_map,
                insert_map = insert_map,
    )

    if not has_any_replce_or_insert:
        return source

    depth += 1
    if depth >= 5:
        print(f"Maximum recursion depth {depth} reached; stopping further hoist to avoid infinite loop.")
        return new_source

    if index_attr_nodes or attr_has_three_more_chains and has_any_replce_or_insert:
        # 继续递归处理新源代码
        return hoist_src(new_source, src_path = src_path, depth = depth)
    else:
        return new_source
