from .Node import Node, Context
from typing import List, Tuple, Optional
from graphviz import Graph


class ProgramNode(Node):
    def __init__(self, heading, block):
        super().__init__("program", [heading, block])
        self.heading = heading
        self.block = block

    def to_string(self, context: Context) -> str:
        return f"{self.heading.to_string(context)}{self.block.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        vm_code = ["START"]
        vm_code.extend(self.block.to_vm(context).split("\n"))
        vm_code.append("STOP")
        return "\n".join(line for line in vm_code if line.strip())

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ProgramNode)
            and self.heading == other.heading
            and self.block == other.block
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        heading_id = self.heading.append_to_graph(graph)
        block_id = self.block.append_to_graph(graph)
        graph.edge(str(node_id), str(heading_id))
        graph.edge(str(node_id), str(block_id))
        return node_id


class ProgramHeadingNode(Node):
    def __init__(self, identifier):
        super().__init__("programHeading", [identifier])
        self.identifier = identifier

    def to_string(self, context) -> str:
        return f"program {self.identifier.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        # Program heading doesn't generate VM code
        return ""

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ProgramHeadingNode)
            and self.identifier == other.identifier
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        identifier_id = self.identifier.append_to_graph(graph)
        graph.edge(str(node_id), str(identifier_id))
        return node_id


class IdentifierNode(Node):
    def __init__(self, value: str):
        super().__init__("identifier", [], value)
        self.value = value

    def to_string(self, context) -> str:
        return self.value

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, IdentifierNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id


class BlockNode(Node):
    def __init__(self, declarations, compound_stmt):
        super().__init__("block", [declarations, compound_stmt])
        self.declarations = declarations
        self.compound_stmt = compound_stmt

    def to_string(self, context) -> str:
        decls = self.declarations.to_string(context)
        return f"{decls}\n{self.compound_stmt.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        decls = self.declarations.to_vm(context)
        stmts = self.compound_stmt.to_vm(context)
        return f"{decls}\n{stmts}".strip()

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, BlockNode) and self.declarations == other.declarations

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        decl_id = self.declarations.append_to_graph(graph)
        stmt_id = self.compound_stmt.append_to_graph(graph)
        graph.edge(str(node_id), str(decl_id))
        graph.edge(str(node_id), str(stmt_id))
        return node_id


class DeclarationsNode(Node):
    def __init__(self, declarations=None):
        super().__init__("declarations", declarations if declarations else [])
        self.declarations = declarations if declarations else []

    def to_string(self, context) -> str:
        if not self.declarations:
            return ""  # Empty declarations
        return "\n".join(decl.to_string(context) for decl in self.declarations)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, DeclarationsNode)
            and self.declarations == other.declarations
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for decl in self.declarations:
            decl_id = decl.append_to_graph(graph)
            graph.edge(str(node_id), str(decl_id))
        return node_id


class CompoundStatementNode(Node):
    def __init__(self, statements):
        super().__init__("compoundStatement", [statements])
        self.statements = statements

    def to_string(self, context) -> str:
        return f"begin\n    {self.statements.to_string(context)}\nend"

    def to_vm(self, context: Context) -> str:
        return self.statements.to_vm(context).strip()

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, CompoundStatementNode)
            and self.statements == other.statements
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        stmt_id = self.statements.append_to_graph(graph)
        graph.edge(str(node_id), str(stmt_id))
        return node_id


class StatementsNode(Node):
    def __init__(self, statements: List[Node]):
        super().__init__("statements", statements)
        self.statements = statements

    def to_string(self, context) -> str:
        return ";\n    ".join(stmt.to_string(context) for stmt in self.statements)

    def to_vm(self, context: Context) -> str:
        return "\n".join(
            stmt.to_vm(context) for stmt in self.statements if stmt
        ).strip()

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, StatementsNode) and self.statements == other.statements

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for stmt in self.statements:
            stmt_id = stmt.append_to_graph(graph)
            graph.edge(str(node_id), str(stmt_id))
        return node_id


class ProcedureStatementNode(Node):
    def __init__(self, identifier, expr_list):
        super().__init__("procedureStatement", [identifier, expr_list])
        self.identifier = identifier
        self.expr_list = expr_list

    def to_string(self, context) -> str:
        return (
            f"{self.identifier.to_string(context)}({self.expr_list.to_string(context)})"
        )

    def to_vm(self, context: Context) -> str:
        if self.identifier.value.lower() == "write":
            vm_code = []
            for expr in self.expr_list.expressions:
                if isinstance(expr, StringNode):
                    vm_code.append(f'PUSHS "{expr.value}"')
                    vm_code.append("WRITES")
                else:
                    vm_code.append(expr.to_vm(context))
                    vm_code.append("WRITEI")
            return "\n".join(vm_code)
        elif self.identifier.value.lower() == "writeln":
            vm_code = []
            if self.expr_list and self.expr_list.expressions:
                for expr in self.expr_list.expressions:
                    if isinstance(expr, StringNode):
                        vm_code.append(f'PUSHS "{expr.value}"')
                        vm_code.append("WRITES")
                    else:
                        vm_code.append(expr.to_vm(context))
                        vm_code.append("WRITEI")
            vm_code.append("WRITELN")
            return "\n".join(vm_code)
        elif self.identifier.value.lower() == "readln":
            vm_code = []
            vm_code.append("READ")  # Read string from keyboard
            vm_code.append("ATOI")  # Convert string to integer
            # Store in variable
            var = self.expr_list.expressions[0]
            var_addr = context.get_next_var_address()
            vm_code.append(f"STOREG {var_addr}")
            return "\n".join(vm_code)
        return ""

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ProcedureStatementNode)
            and self.identifier == other.identifier
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        expr_id = self.expr_list.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(expr_id))
        return node_id


class ExpressionListNode(Node):
    def __init__(self, expressions: List[Node]):
        super().__init__("expressionList", expressions)
        self.expressions = expressions

    def to_string(self, context) -> str:
        return ", ".join(expr.to_string(context) for expr in self.expressions)

    def to_vm(self, context: Context) -> str:
        return "\n".join(expr.to_vm(context) for expr in self.expressions)

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ExpressionListNode)
            and self.expressions == other.expressions
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for expr in self.expressions:
            expr_id = expr.append_to_graph(graph)
            graph.edge(str(node_id), str(expr_id))
        return node_id


class StringNode(Node):
    def __init__(self, value: str):
        super().__init__("string", [], value)
        self.value = value

    def to_string(self, context) -> str:
        return f"'{self.value}'"

    def to_vm(self, context: Context) -> str:
        return f'PUSHS "{self.value}"'

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, StringNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id


class EmptyStatementNode(Node):
    def __init__(self):
        super().__init__("emptyStatement", [])

    def to_string(self, context) -> str:
        return ""

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, EmptyStatementNode)

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        return node_id

class VariableDeclarationBlock(Node):
    def __init__(self, declarations: List[Node]):
        super().__init__("variableDeclarationBlock", declarations)
        self.declarations = declarations

    def to_string(self, context: Context) -> str:
        return "\n".join(decl.to_string(context) for decl in self.declarations)

    def to_vm(self, context: Context) -> str:
        # Allocate space for variables
        vm_code = []
        for decl in self.declarations:
            identifiers = decl.identifier.identifiers
            for id_node in identifiers:
                vm_code.append(f"PUSHG 0")  # Initialize with 0
                vm_code.append(f"STOREG {context.get_next_var_address()}")
        return '\n'.join(vm_code)

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, VariableDeclarationBlock)
            and self.declarations == other.declarations
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for decl in self.declarations:
            decl_id = decl.append_to_graph(graph)
            graph.edge(str(node_id), str(decl_id))
        return node_id

class VariableDeclarationList(Node):
    def __init__(self, declarations=None):
        self.declarations = declarations if declarations else []
        # Pass declarations as children to Node constructor
        super().__init__("variableDeclarationList", children=self.declarations)

    def to_string(self, context: Context) -> str:
        return ", ".join(decl.to_string(context) for decl in self.declarations)

    def to_vm(self, context: Context) -> str:
        return "\n".join(decl.to_vm(context) for decl in self.declarations)

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, VariableDeclarationList)
            and self.declarations == other.declarations
        )

    def __iter__(self):
        """Make this class iterable by returning an iterator over declarations"""
        return iter(self.declarations)

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)  # Use self.name instead of self.type
        for decl in self.declarations:
            decl_id = decl.append_to_graph(graph)
            graph.edge(str(node_id), str(decl_id))
        return node_id


class VariableDeclaration(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("variableDeclaration", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}: {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return f"ALLOC {self.identifier.value}"

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, VariableDeclaration)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id


class IdentifierListNode(Node):
    def __init__(self, identifiers: List[Node]):
        super().__init__("identifierList", identifiers)
        self.identifiers = identifiers

    def to_string(self, context: Context) -> str:
        return ", ".join(id.to_string(context) for id in self.identifiers)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, IdentifierListNode)
            and self.identifiers == other.identifiers
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for id in self.identifiers:
            id_id = id.append_to_graph(graph)
            graph.edge(str(node_id), str(id_id))
        return node_id

class TypeIdentifierNode(Node):
    def __init__(self, value: str):
        super().__init__("typeIdentifier", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, TypeIdentifierNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id


class VariableNode(Node):
    def __init__(self, identifier: Node):
        super().__init__("variable", [identifier])
        self.identifier = identifier

    def to_string(self, context: Context) -> str:
        return self.identifier.to_string(context)

    def to_vm(self, context: Context) -> str:
        return f"LOAD {self.identifier.value}"

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, VariableNode) and self.identifier == other.identifier

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        return node_id


class IfStatementNode(Node):
    def __init__(self, condition: Node, then_block: Node, else_block: Node = None):
        super().__init__("ifStatement", [condition, then_block, else_block])
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

    def to_string(self, context: Context) -> str:
        if self.else_block:
            return f"if {self.condition.to_string(context)} then {self.then_block.to_string(context)} else {self.else_block.to_string(context)}"
        return f"if {self.condition.to_string(context)} then {self.then_block.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        label_count = context.get_next_label()
        else_label = f"ELSE_{label_count}"
        end_label = f"ENDIF_{label_count}"
        
        vm_code = []
        # Evaluate condition
        vm_code.append(self.condition.to_vm(context))
        vm_code.append(f"JZ {else_label}")
        
        # Then block
        vm_code.append(self.then_block.to_vm(context))
        vm_code.append(f"JUMP {end_label}")
        
        # Else block
        vm_code.append(f"{else_label}:")
        if self.else_block:
            vm_code.append(self.else_block.to_vm(context))
        
        vm_code.append(f"{end_label}:")
        return '\n'.join(vm_code)

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, IfStatementNode)
            and self.condition == other.condition
            and self.then_block == other.then_block
            and self.else_block == other.else_block
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        cond_id = self.condition.append_to_graph(graph)
        then_id = self.then_block.append_to_graph(graph)
        graph.edge(str(node_id), str(cond_id))
        graph.edge(str(node_id), str(then_id))
        if self.else_block:
            else_id = self.else_block.append_to_graph(graph)
            graph.edge(str(node_id), str(else_id))
        return node_id


class ExpressionNode(Node):
    def __init__(self, left: Node, right: Node, operator: Node):
        super().__init__("expression", [left, operator, right])
        self.left = left
        self.right = right
        self.operator = operator

    def to_string(self, context: Context) -> str:
        return f"{self.left.to_string(context)} {self.operator.to_string(context)} {self.right.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        # Generate VM code for left and right expressions
        left_code = self.left.to_vm(context)
        right_code = self.right.to_vm(context)
        
        # Map operators to VM instructions
        op_map = {
            '+': 'ADD',
            '-': 'SUB', 
            '*': 'MUL',
            '/': 'DIV',
            'mod': 'MOD',
            'and': 'AND',
            'or': 'OR'
        }
        
        if isinstance(self.operator, RelationalOperatorNode):
            return f"{left_code}\n{right_code}\n{self.operator.to_vm(context)}"
        
        op = op_map.get(self.operator.value, 'NOP')
        return f"{left_code}\n{right_code}\n{op}"



    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ExpressionNode)
            and self.left == other.left
            and self.right == other.right
            and self.operator == other.operator
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        left_id = self.left.append_to_graph(graph)
        op_id = self.operator.append_to_graph(graph)
        right_id = self.right.append_to_graph(graph)
        graph.edge(str(node_id), str(left_id))
        graph.edge(str(node_id), str(op_id))
        graph.edge(str(node_id), str(right_id))
        return node_id


class AssigmentStatementNode(Node):
    def __init__(self, variable: Node, expression: Node):
        super().__init__("assignmentStatement", [variable, expression])
        self.variable = variable
        self.expression = expression

    def to_string(self, context: Context) -> str:
        return f"{self.variable.to_string(context)} := {self.expression.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        # Evaluate expression
        expr_code = self.expression.to_vm(context)
        # Store result in variable
        var_addr = context.get_var_address(self.variable.identifier.value)
        return f"{expr_code}\nSTOREG {var_addr}"
        
    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, AssigmentStatementNode)
            and self.variable == other.variable
            and self.expression == other.expression
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        var_id = self.variable.append_to_graph(graph)
        expr_id = self.expression.append_to_graph(graph)
        graph.edge(str(node_id), str(var_id))
        graph.edge(str(node_id), str(expr_id))
        return node_id

class RelationalOperatorNode(Node):
    def __init__(self, value: str):
        super().__init__("relationalOperator", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        op_map = {
            '>': 'SUP',
            '<': 'INF',
            '>=': 'SUPEQ',
            '<=': 'INFEQ',
            '=': 'EQUAL',
            '<>': 'NOT'
        }
        return op_map.get(self.value, 'NOP')

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, RelationalOperatorNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id

class ForStatementNode(Node):
    def __init__(self, identifier: Node, start_expr: Node, end_expr: Node, block: Node, direction: str):
        super().__init__("forStatement", [identifier, start_expr, end_expr, block])
        self.identifier = identifier
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.block = block
        self.direction = direction.lower()  # "to" or "downto"

    def to_string(self, context: Context) -> str:
        return (
            f"for {self.identifier.to_string(context)} := {self.start_expr.to_string(context)} "
            f"{self.direction} {self.end_expr.to_string(context)} do {self.block.to_string(context)}"
        )

    def to_vm(self, context: Context) -> str:
        label_count = context.get_next_label()
        loop_start_label = f"FOR_START_{label_count}"
        loop_end_label = f"FOR_END_{label_count}"
        
        vm_code = []

        # Initialize loop variable
        vm_code.append(f"{self.identifier.to_vm(context)} := {self.start_expr.to_vm(context)}")

        # Loop start label
        vm_code.append(f"{loop_start_label}:")

        # Condition check
        if self.direction == "to":
            # If identifier > end_expr, exit loop
            vm_code.append(self.identifier.to_vm(context))
            vm_code.append(self.end_expr.to_vm(context))
            vm_code.append(f"GT")
            vm_code.append(f"JZ {loop_end_label}")
        else:  # downto
            # If identifier < end_expr, exit loop
            vm_code.append(self.identifier.to_vm(context))
            vm_code.append(self.end_expr.to_vm(context))
            vm_code.append(f"LT")
            vm_code.append(f"JZ {loop_end_label}")

        # Execute loop body
        vm_code.append(self.block.to_vm(context))

        # Increment or decrement
        if self.direction == "to":
            vm_code.append(f"{self.identifier.to_vm(context)} := {self.identifier.to_vm(context)} + 1")
        else:
            vm_code.append(f"{self.identifier.to_vm(context)} := {self.identifier.to_vm(context)} - 1")

        # Jump back to start
        vm_code.append(f"JUMP {loop_start_label}")

        # End label
        vm_code.append(f"{loop_end_label}:")

        return '\n'.join(vm_code)

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ForStatementNode)
            and self.identifier == other.identifier
            and self.start_expr == other.start_expr
            and self.end_expr == other.end_expr
            and self.block == other.block
            and self.direction == other.direction
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name} ({self.direction})")
        id_id = self.identifier.append_to_graph(graph)
        start_id = self.start_expr.append_to_graph(graph)
        end_id = self.end_expr.append_to_graph(graph)
        block_id = self.block.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(start_id))
        graph.edge(str(node_id), str(end_id))
        graph.edge(str(node_id), str(block_id))
        return node_id

class TermNode(Node):
    def __init__(self, left: Node, right: Node, operator: Node):
        super().__init__("term", [left, operator, right])
        self.left = left
        self.right = right
        self.operator = operator

    def to_string(self, context: Context) -> str:
        return f"{self.left.to_string(context)} {self.operator.to_string(context)} {self.right.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        # Generate VM code for left and right expressions
        left_code = self.left.to_vm(context)
        right_code = self.right.to_vm(context)
        
        # Map operators to VM instructions
        op_map = {
            '+': 'ADD',
            '-': 'SUB', 
            '*': 'MUL',
            '/': 'DIV',
            'mod': 'MOD',
            'and': 'AND',
            'or': 'OR'
        }
        
        if isinstance(self.operator, RelationalOperatorNode):
            return f"{left_code}\n{right_code}\n{self.operator.to_vm(context)}"
        
        op = op_map.get(self.operator.value, 'NOP')
        return f"{left_code}\n{right_code}\n{op}"
    
    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TermNode)
            and self.left == other.left
            and self.right == other.right
            and self.operator == other.operator
        )
    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        left_id = self.left.append_to_graph(graph)
        op_id = self.operator.append_to_graph(graph)
        right_id = self.right.append_to_graph(graph)
        graph.edge(str(node_id), str(left_id))
        graph.edge(str(node_id), str(op_id))
        graph.edge(str(node_id), str(right_id))
        return node_id

class ConstantDefinitionBlock(Node):
    def __init__(self, definitions: List[Node]):
        super().__init__("constantDefinitionBlock", definitions)
        self.definitions = definitions

    def to_string(self, context: Context) -> str:
        return "\n".join(defn.to_string(context) for defn in self.definitions)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ConstantDefinitionBlock)
            and self.definitions == other.definitions
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for defn in self.definitions:
            defn_id = defn.append_to_graph(graph)
            graph.edge(str(node_id), str(defn_id))
        return node_id

class ConstantDefinitionList(Node):
    def __init__(self, definitions=None):
        self.definitions = definitions if definitions else []
        # Pass definitions
        super().__init__("constantDefinitionList", children=self.definitions)

    def to_string(self, context: Context) -> str:
        return ", ".join(defn.to_string(context) for defn in self.definitions)
    def to_vm(self, context: Context) -> str:
        return ""
    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []
    def __eq__(self, other) -> bool:    
        return (
            isinstance(other, ConstantDefinitionList)
            and self.definitions == other.definitions
        )
    def __iter__(self):
        """Make this class iterable by returning an iterator over definitions"""
        return iter(self.definitions)

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for defn in self.definitions:
            defn_id = defn.append_to_graph(graph)
            graph.edge(str(node_id), str(defn_id))
        return node_id

class ConstantDefinition(Node):
    def __init__(self, identifier: Node, value: Node):
        super().__init__("constantDefinition", [identifier, value])
        self.identifier = identifier
        self.value = value

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)} = {self.value.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ConstantDefinition)
            and self.identifier == other.identifier
            and self.value == other.value
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        val_id = self.value.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(val_id))
        return node_id

class ConstantNode(Node):
    def __init__(self, value: str):
        super().__init__("constant", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return f'PUSHS "{self.value}"'

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, ConstantNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id

class UnsignedIntegerNode(Node):
    def __init__(self, value: str):
        super().__init__("unsignedInteger", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return f'PUSHS {self.value}'

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, UnsignedIntegerNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id

class UnsignedRealNode(Node):
    def __init__(self, value: str):
        super().__init__("unsignedReal", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return f'PUSHS {self.value}'

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, UnsignedRealNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id

class SignNode(Node):
    def __init__(self, value: str):
        super().__init__("sign", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return ""  # Sign application usually handled in ConstantNode or arithmetic logic

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, SignNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id


class ConstantChrNode(Node):
    def __init__(self, value: str):
        super().__init__("constantChr", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return f'PUSHS "{self.value}"'

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, ConstantChrNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id


class TypeDeclarationBlock(Node):
    def __init__(self, declarations: List[Node]):
        super().__init__("typeDeclarationBlock", declarations)
        self.declarations = declarations

    def to_string(self, context: Context) -> str:
        return "\n".join(decl.to_string(context) for decl in self.declarations)

    def to_vm(self, context: Context) -> str:
        return ""
    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []
    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TypeDeclarationBlock)
            and self.declarations == other.declarations
        )
    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for decl in self.declarations:
            decl_id = decl.append_to_graph(graph)
            graph.edge(str(node_id), str(decl_id))
        return node_id

class TypeDefinitionList(Node):
    def __init__(self, definitions=None):
        self.definitions = definitions if definitions else []
        # Pass definitions
        super().__init__("typeDefinitionList", children=self.definitions)
    def to_string(self, context: Context) -> str:
        return ", ".join(defn.to_string(context) for defn in self.definitions)
    def to_vm(self, context: Context) -> str:
        return ""
    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []
    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TypeDefinitionList)
            and self.definitions == other.definitions
        )
    def __iter__(self):
        """Make this class iterable by returning an iterator over definitions"""
        return iter(self.definitions)
    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for defn in self.definitions:
            defn_id = defn.append_to_graph(graph)
            graph.edge(str(node_id), str(defn_id))
        return node_id

class TypeDefinition(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("typeDefinition", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)} = {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TypeDefinition)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id


class ScalarTypeNode(Node):
    def __init__(self, value: str):
        super().__init__("scalarType", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, ScalarTypeNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id

class SubRangeTypeNode(Node):
    def __init__(self, lower_bound: Node, upper_bound: Node):
        super().__init__("subRangeType", [lower_bound, upper_bound])
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def to_string(self, context: Context) -> str:
        return f"[{self.lower_bound.to_string(context)}..{self.upper_bound.to_string(context)}]"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SubRangeTypeNode)
            and self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        lower_id = self.lower_bound.append_to_graph(graph)
        upper_id = self.upper_bound.append_to_graph(graph)
        graph.edge(str(node_id), str(lower_id))
        graph.edge(str(node_id), str(upper_id))
        return node_id

class StringTypeNode(Node):
    def __init__(self, length: Node):
        super().__init__("stringType", [length])
        self.length = length

    def to_string(self, context: Context) -> str:
        return f"string[{self.length.to_string(context)}]"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, StringTypeNode) and self.length == other.length

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        length_id = self.length.append_to_graph(graph)
        graph.edge(str(node_id), str(length_id))
        return node_id

class ArrayTypeNode(Node):
    def __init__(self, index_type: Node, element_type: Node):
        super().__init__("arrayType", [index_type, element_type])
        self.index_type = index_type
        self.element_type = element_type

    def to_string(self, context: Context) -> str:
        return f"array[{self.index_type.to_string(context)}] of {self.element_type.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ArrayTypeNode)
            and self.index_type == other.index_type
            and self.element_type == other.element_type
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        index_id = self.index_type.append_to_graph(graph)
        element_id = self.element_type.append_to_graph(graph)
        graph.edge(str(node_id), str(index_id))
        graph.edge(str(node_id), str(element_id))
        return node_id

class TypeListNode(Node):
    def __init__(self, types: List[Node]):
        super().__init__("typeList", types)
        self.types = types

    def to_string(self, context: Context) -> str:
        return ", ".join(type_node.to_string(context) for type_node in self.types)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TypeListNode)
            and self.types == other.types
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for type_node in self.types:
            type_id = type_node.append_to_graph(graph)
            graph.edge(str(node_id), str(type_id))
        return node_id

class recordTypeNode(Node):
    def __init__(self, fields: List[Node]):
        super().__init__("recordType", fields)
        self.fields = fields

    def to_string(self, context: Context) -> str:
        return "{" + ", ".join(field.to_string(context) for field in self.fields) + "}"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, recordTypeNode)
            and self.fields == other.fields
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for field in self.fields:
            field_id = field.append_to_graph(graph)
            graph.edge(str(node_id), str(field_id))
        return node_id

class FieldListNode(Node):
    def __init__(self, fields: List[Node]):
        super().__init__("fieldList", fields)
        self.fields = fields

    def to_string(self, context: Context) -> str:
        return ", ".join(field.to_string(context) for field in self.fields)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FieldListNode)
            and self.fields == other.fields
        )
    def __iter__(self):
        """Make this class iterable by returning an iterator over fields"""
        return iter(self.fields)
    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for field in self.fields:
            field_id = field.append_to_graph(graph)
            graph.edge(str(node_id), str(field_id))
        return node_id

class FixedPartNode(Node):
    def __init__(self, fields: List[Node]):
        super().__init__("fixedPart", fields)
        self.fields = fields

    def to_string(self, context: Context) -> str:
        return ", ".join(field.to_string(context) for field in self.fields)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FixedPartNode)
            and self.fields == other.fields
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for field in self.fields:
            field_id = field.append_to_graph(graph)
            graph.edge(str(node_id), str(field_id))
        return node_id


class RecordSectionList(Node):
    def __init__(self, sections: List[Node]):
        super().__init__("recordSectionList", sections)
        self.sections = sections

    def to_string(self, context: Context) -> str:
        return ", ".join(section.to_string(context) for section in self.sections)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, RecordSectionList)
            and self.sections == other.sections
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for section in self.sections:
            section_id = section.append_to_graph(graph)
            graph.edge(str(node_id), str(section_id))
        return node_id

class RecordSectionNode(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("recordSection", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}: {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, RecordSectionNode)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id

class VariantPartNode(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("variantPart", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}: {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, VariantPartNode)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id

class TagNode(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("tag", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}: {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, TagNode)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id

class VariantListNode(Node):
    def __init__(self, tags: List[Node]):
        super().__init__("variantList", tags)
        self.tags = tags

    def to_string(self, context: Context) -> str:
        return ", ".join(tag.to_string(context) for tag in self.tags)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, VariantListNode)
            and self.tags == other.tags
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for tag in self.tags:
            tag_id = tag.append_to_graph(graph)
            graph.edge(str(node_id), str(tag_id))
        return node_id

class VariantNode(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("variant", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}: {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, VariantNode)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id

class ConstListNode(Node):
    def __init__(self, constants: List[Node]):
        super().__init__("constList", constants)
        self.constants = constants

    def to_string(self, context: Context) -> str:
        return ", ".join(constant.to_string(context) for constant in self.constants)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []
    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ConstListNode)
            and self.constants == other.constants
        )
    def __iter__(self):
        """Make this class iterable by returning an iterator over constants"""
        return iter(self.constants)
    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for constant in self.constants:
            constant_id = constant.append_to_graph(graph)
            graph.edge(str(node_id), str(constant_id))
        return node_id


class SetTypeNode(Node):
    def __init__(self, element_type: Node):
        super().__init__("setType", [element_type])
        self.element_type = element_type

    def to_string(self, context: Context) -> str:
        return f"set of {self.element_type.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SetTypeNode)
            and self.element_type == other.element_type
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        type_id = self.element_type.append_to_graph(graph)
        graph.edge(str(node_id), str(type_id))
        return node_id

class ProcedureDeclarationNode(Node):
    def __init__(self, identifier: Node, params: Node, block: Node):
        super().__init__("procedureDeclaration", [identifier, params, block])
        self.identifier = identifier
        self.params = params
        self.block = block

    def to_string(self, context: Context) -> str:
        return f"procedure {self.identifier.to_string(context)}({self.params.to_string(context)}) {self.block.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ProcedureDeclarationNode)
            and self.identifier == other.identifier
            and self.params == other.params
            and self.block == other.block
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        params_id = self.params.append_to_graph(graph)
        block_id = self.block.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(params_id))
        graph.edge(str(node_id), str(block_id))
        return node_id

class FormalParameterListNode(Node):
    def __init__(self, parameters: List[Node]):
        super().__init__("formalParameterList", parameters)
        self.parameters = parameters

    def to_string(self, context: Context) -> str:
        return ", ".join(param.to_string(context) for param in self.parameters)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FormalParameterListNode)
            and self.parameters == other.parameters
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for param in self.parameters:
            param_id = param.append_to_graph(graph)
            graph.edge(str(node_id), str(param_id))
        return node_id

class FormalParameterSectionListNode(Node):
    def __init__(self, sections: List[Node]):
        super().__init__("formalParameterSectionList", sections)
        self.sections = sections

    def to_string(self, context: Context) -> str:
        return ", ".join(section.to_string(context) for section in self.sections)

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FormalParameterSectionListNode)
            and self.sections == other.sections
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        for section in self.sections:
            section_id = section.append_to_graph(graph)
            graph.edge(str(node_id), str(section_id))
        return node_id

class FormalParameterSectionNode(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("formalParameterSection", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}: {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FormalParameterSectionNode)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id

class ParameterGroupNode(Node):
    def __init__(self, identifier: Node, type_node: Node):
        super().__init__("parameterGroup", [identifier, type_node])
        self.identifier = identifier
        self.type_node = type_node

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}: {self.type_node.to_string(context)};"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ParameterGroupNode)
            and self.identifier == other.identifier
            and self.type_node == other.type_node
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        type_id = self.type_node.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(type_id))
        return node_id

class FunctionDeclarationNode(Node):
    def __init__(self, identifier: Node, params: Node, return_type: Node, block: Node):
        super().__init__("functionDeclaration", [identifier, params, return_type, block])
        self.identifier = identifier
        self.params = params
        self.return_type = return_type
        self.block = block

    def to_string(self, context: Context) -> str:
        return f"function {self.identifier.to_string(context)}({self.params.to_string(context)}) : {self.return_type.to_string(context)} {self.block.to_string(context)}"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FunctionDeclarationNode)
            and self.identifier == other.identifier
            and self.params == other.params
            and self.return_type == other.return_type
            and self.block == other.block
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        params_id = self.params.append_to_graph(graph)
        return_type_id = self.return_type.append_to_graph(graph)
        block_id = self.block.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(params_id))
        graph.edge(str(node_id), str(return_type_id))
        graph.edge(str(node_id), str(block_id))
        return node_id

class IndexedVariableNode(Node):
    def __init__(self, identifier: Node, index: Node):
        super().__init__("indexedVariable", [identifier, index])
        self.identifier = identifier
        self.index = index

    def to_string(self, context: Context) -> str:
        return f"{self.identifier.to_string(context)}[{self.index.to_string(context)}]"

    def to_vm(self, context: Context) -> str:
        return ""

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, IndexedVariableNode)
            and self.identifier == other.identifier
            and self.index == other.index
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        id_id = self.identifier.append_to_graph(graph)
        index_id = self.index.append_to_graph(graph)
        graph.edge(str(node_id), str(id_id))
        graph.edge(str(node_id), str(index_id))
        return node_id

class SimpleExpressionNode(Node):
    def __init__(self, left: Node, operator: Optional[Node] = None, right: Optional[Node] = None):
        children = [left] if not operator else [left, operator] if not right else [left, operator, right]
        super().__init__("simpleExpression", children)
        self.left = left
        self.operator = operator
        self.right = right

    def to_string(self, context: Context) -> str:
        if self.operator and self.right:
            return f"{self.left.to_string(context)} {self.operator.to_string(context)} {self.right.to_string(context)}"
        elif self.operator:  # unary case
            return f"{self.operator.to_string(context)}{self.left.to_string(context)}"
        else:
            return self.left.to_string(context)

    def to_vm(self, context: Context) -> str:
        if self.operator and self.right:
            left_code = self.left.to_vm(context)
            right_code = self.right.to_vm(context)
            op_code = self.operator.to_vm(context)
            return f"{left_code}\n{right_code}\n{op_code}"
        elif self.operator:
            term_code = self.left.to_vm(context)
            if self.operator.value == '-':
                return f"{term_code}\nNEG"
            else:
                return term_code  # + sign is no-op
        else:
            return self.left.to_vm(context)

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, SimpleExpressionNode)
            and self.left == other.left
            and self.operator == other.operator
            and self.right == other.right
        )

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), self.name)
        graph.edge(str(node_id), str(self.left.append_to_graph(graph)))
        if self.operator:
            graph.edge(str(node_id), str(self.operator.append_to_graph(graph)))
        if self.right:
            graph.edge(str(node_id), str(self.right.append_to_graph(graph)))
        return node_id

class AdditionOperatorNode(Node):
    def __init__(self, value: str):
        super().__init__("additionOperator", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return f"ADD {self.value}"

    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []

    def __eq__(self, other) -> bool:
        return isinstance(other, AdditionOperatorNode) and self.value == other.value

    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id

class MultiplicativeOperatorNode(Node):
    def __init__(self, value: str):
        super().__init__("multiplicativeOperator", [], value)
        self.value = value

    def to_string(self, context: Context) -> str:
        return self.value

    def to_vm(self, context: Context) -> str:
        return f"MULT {self.value}"
    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        return True, []
    def __eq__(self, other) -> bool:
        return isinstance(other, MultiplicativeOperatorNode) and self.value == other.value
    def append_to_graph(self, graph: Graph) -> int:
        node_id = len(graph.body)
        graph.node(str(node_id), f"{self.name}: {self.value}")
        return node_id



