import sys

import ply.yacc as yacc
from Lexer import tokens

# Define precedence to resolve ambiguity in the grammar
precedence = (
    ("right", "ASSIGN"),
    ("left", "OR"),
    ("left", "AND"),
    ("left", "NOT"),
    ("left", "EQUAL", "NOT_EQUAL", "LT", "LE", "GT", "GE", "IN"),
    ("left", "PLUS", "MINUS"),
    ("left", "STAR", "SLASH", "DIV", "MOD"),
)


# Node class for AST construction
class Node:
    def __init__(self, type, children=None, leaf=None):
        self.type = type
        self.children = children if children else []
        self.leaf = leaf

    def _to_string(self, level=0):
        ret = "  " * level + self.type
        if self.leaf is not None:
            ret += ": " + str(self.leaf)
        ret += "\n"
        for child in self.children:
            if isinstance(child, Node):
                ret += child._to_string(level + 1)
            else:
                # Handle non-Node children (e.g., strings)
                ret += "  " * (level + 1) + str(child) + "\n"
        return ret

    def __str__(self):
        return self._to_string()


# Grammar rules
def p_program(p):
    """program : programHeading block DOT"""
    p[0] = Node("program", [p[1], p[2]])


def p_programHeading(p):
    """programHeading : PROGRAM identifier SEMI"""
    p[0] = Node("programHeading", [p[2]])


def p_identifier(p):
    """identifier : IDENT"""
    p[0] = Node("identifier", [], p[1])


def p_block(p):
    """block : declarations compoundStatement"""
    p[0] = Node("block", [p[1], p[2]])


def p_declarations(p):
    """declarations : declaration declarations
    | empty"""
    if len(p) > 2:
        p[0] = Node("declarations", [p[1]] + p[2].children)
    else:
        p[0] = Node("declarations", [])


def p_declaration(p):
    """declaration : constantDefinitionBlock
    | typeDeclarationBlock
    | variableDeclarationBlock
    | procedureAndFunctionDeclarationBlock"""
    p[0] = p[1]


def p_constantDefinitionBlock(p):
    """constantDefinitionBlock : CONST constantDefinitionList"""
    p[0] = Node("constantDefinitionBlock", [p[2]])


def p_constantDefinitionList(p):
    """constantDefinitionList : constantDefinitionList constantDefinition SEMI
    | constantDefinition SEMI"""
    if len(p) > 3:
        p[1].children.append(p[2])
        p[0] = p[1]
    else:
        p[0] = Node("constantDefinitionList", [p[1]])


def p_constantDefinition(p):
    """constantDefinition : identifier EQUAL constant"""
    p[0] = Node("constantDefinition", [p[1], p[3]])


def p_constant(p):
    """constant : unsignedNumber
    | sign unsignedNumber
    | identifier
    | sign identifier
    | string
    | constantChr"""
    if len(p) > 2:
        p[0] = Node("constant", [p[1], p[2]])
    else:
        p[0] = Node("constant", [p[1]])


def p_unsignedNumber(p):
    """unsignedNumber : unsignedInteger
    | unsignedReal"""
    p[0] = p[1]


def p_unsignedInteger(p):
    """unsignedInteger : NUM_INT"""
    p[0] = Node("unsignedInteger", [], p[1])


def p_unsignedReal(p):
    """unsignedReal : NUM_REAL"""
    p[0] = Node("unsignedReal", [], p[1])


def p_sign(p):
    """sign : PLUS
    | MINUS"""
    p[0] = Node("sign", [], p[1])


def p_string(p):
    """string : STRING_LITERAL"""
    p[0] = Node("string", [], p[1])


def p_constantChr(p):
    """constantChr : CHR LPAREN unsignedInteger RPAREN"""
    p[0] = Node("constantChr", [p[3]])


def p_typeDeclarationBlock(p):
    """typeDeclarationBlock : TYPE typeDefinitionList"""
    p[0] = Node("typeDeclarationBlock", [p[2]])


def p_typeDefinitionList(p):
    """typeDefinitionList : typeDefinitionList typeDefinition SEMI
    | typeDefinition SEMI"""
    if len(p) > 3:
        p[1].children.append(p[2])
        p[0] = p[1]
    else:
        p[0] = Node("typeDefinitionList", [p[1]])


def p_typeDefinition(p):
    """typeDefinition : identifier EQUAL type_"""
    p[0] = Node("typeDefinition", [p[1], p[3]])


def p_type_(p):
    """type_ : scalarType
    | subrangeType
    | typeIdentifier
    | stringType
    | arrayType"""
    p[0] = p[1]


def p_scalarType(p):
    """scalarType : LPAREN identifierList RPAREN"""
    p[0] = Node("scalarType", [p[2]])


def p_identifierList(p):
    """identifierList : identifierList COMMA identifier
    | identifier"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("identifierList", [p[1]])


def p_subrangeType(p):
    """subrangeType : unsignedInteger DOTDOT unsignedInteger"""
    p[0] = Node("subrangeType", [p[1], p[3]])


def p_identifierType(p):
    """identifierType : IDENT"""
    p[0] = Node("identifierType", [], p[1])


def p_stringType(p):
    """stringType : STRING LBRACK unsignedInteger RBRACK"""
    p[0] = Node("stringType", [p[3]])


def p_structuredType(p):
    """structuredType : arrayType
    | recordType
    | setType"""
    p[0] = p[1]


def p_arrayType(p):
    """arrayType : ARRAY LBRACK subrangeType RBRACK OF typeIdentifier"""
    p[0] = Node("arrayType", [p[3], p[6]])


def p_typeList(p):
    """typeList : typeList COMMA indexType
    | indexType"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("typeList", [p[1]])


def p_indexType(p):
    """indexType : simpleType"""
    p[0] = p[1]


def p_simpleType(p):
    """simpleType : identifierType"""
    p[0] = p[1]


def p_recordType(p):
    """recordType : RECORD fieldList END"""
    p[0] = Node("recordType", [p[2]])


def p_fieldList(p):
    """fieldList : fixedPart
    | fixedPart SEMI variantPart
    | variantPart"""
    if len(p) > 2:
        p[0] = Node("fieldList", [p[1], p[3]])
    else:
        p[0] = Node("fieldList", [p[1]])


def p_fixedPart(p):
    """fixedPart : recordSectionList"""
    p[0] = Node("fixedPart", [p[1]])


def p_recordSectionList(p):
    """recordSectionList : recordSectionList SEMI recordSection
    | recordSection"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("recordSectionList", [p[1]])


def p_recordSection(p):
    """recordSection : identifierList COLON type_"""
    p[0] = Node("recordSection", [p[1], p[3]])


def p_variantPart(p):
    """variantPart : CASE tag OF variantList"""
    p[0] = Node("variantPart", [p[2], p[4]])


def p_tag(p):
    """tag : identifier COLON typeIdentifier
    | typeIdentifier"""
    if len(p) > 2:
        p[0] = Node("tag", [p[1], p[3]])
    else:
        p[0] = Node("tag", [p[1]])


def p_typeIdentifier(p):
    """typeIdentifier : identifier
    | INTEGER
    | REAL
    | BOOLEAN
    | CHAR
    | STRING"""
    if isinstance(p[1], Node):  # If it's an identifier node
        p[0] = p[1]
    else:  # If it's a keyword like INTEGER, REAL, etc.
        p[0] = Node("typeIdentifier", [], p[1])


def p_variantList(p):
    """variantList : variantList SEMI variant
    | variant"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("variantList", [p[1]])


def p_variant(p):
    """variant : constList COLON LPAREN fieldList RPAREN"""
    p[0] = Node("variant", [p[1], p[4]])


def p_constList(p):
    """constList : constList COMMA constant
    | constant"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("constList", [p[1]])


def p_setType(p):
    """setType : SET OF baseType"""
    p[0] = Node("setType", [p[3]])


def p_baseType(p):
    """baseType : simpleType"""
    p[0] = p[1]


def p_variableDeclarationBlock(p):
    """variableDeclarationBlock : VAR variableDeclarationList SEMI"""
    p[0] = Node("variableDeclarationBlock", [p[2]])


def p_variableDeclarationList(p):
    """variableDeclarationList : variableDeclarationList SEMI variableDeclaration
    | variableDeclaration"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("variableDeclarationList", [p[1]])


def p_variableDeclaration(p):
    """variableDeclaration : identifierList COLON type_"""
    p[0] = Node("variableDeclaration", [p[1], p[3]])


def p_procedureAndFunctionDeclarationBlock(p):
    """procedureAndFunctionDeclarationBlock : procedureDeclaration
    | functionDeclaration"""
    p[0] = p[1]


def p_procedureDeclaration(p):
    """procedureDeclaration : PROCEDURE identifier formalParameterList_opt SEMI block"""
    p[0] = Node("procedureDeclaration", [p[2], p[3], p[5]])


def p_formalParameterList_opt(p):
    """formalParameterList_opt : formalParameterList
    | empty"""
    p[0] = p[1]


def p_formalParameterList(p):
    """formalParameterList : LPAREN formalParameterSectionList RPAREN"""
    p[0] = Node("formalParameterList", [p[2]])


def p_formalParameterSectionList(p):
    """formalParameterSectionList : formalParameterSectionList SEMI formalParameterSection
    | formalParameterSection"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("formalParameterSectionList", [p[1]])


def p_formalParameterSection(p):
    """formalParameterSection : parameterGroup
    | VAR parameterGroup
    | FUNCTION parameterGroup
    | PROCEDURE parameterGroup"""
    if len(p) > 2:
        p[0] = Node("formalParameterSection", [p[2]], p[1])
    else:
        p[0] = Node("formalParameterSection", [p[1]])


def p_parameterGroup(p):
    """parameterGroup : identifierList COLON typeIdentifier"""
    p[0] = Node("parameterGroup", [p[1], p[3]])


def p_functionDeclaration(p):
    """functionDeclaration : FUNCTION identifier formalParameterList_opt COLON resultType SEMI block"""
    p[0] = Node("functionDeclaration", [p[2], p[3], p[5], p[7]])


def p_resultType(p):
    """resultType : typeIdentifier"""
    p[0] = p[1]


def p_compoundStatement(p):
    """compoundStatement : BEGIN statements END"""
    p[0] = Node("compoundStatement", [p[2]])


def p_statements(p):
    """statements : statements SEMI statement
    | statement"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("statements", [p[1]])


def p_statement(p):
    """statement : simpleStatement
    | structuredStatement"""
    p[0] = p[1]


def p_simpleStatement(p):
    """simpleStatement : assignmentStatement
    | procedureStatement
    | emptyStatement_"""
    p[0] = p[1]


def p_assignmentStatement(p):
    """assignmentStatement : variable ASSIGN expression"""
    p[0] = Node("assignmentStatement", [p[1], p[3]])


def p_variable(p):
    """variable : identifier
    | indexedVariable"""
    p[0] = Node("variable", [p[1]])


def p_indexedVariable(p):
    """indexedVariable : identifier LBRACK expression RBRACK"""
    p[0] = Node("indexedVariable", [p[1], p[3]])


def p_expression(p):
    """expression : simpleExpression
    | simpleExpression relationalOperator simpleExpression"""
    if len(p) > 2:
        p[0] = Node("expression", [p[1], p[2], p[3]])
    else:
        p[0] = p[1]


def p_relationalOperator(p):
    """relationalOperator : EQUAL
    | NOT_EQUAL
    | LT
    | LE
    | GT
    | GE
    | IN"""
    p[0] = Node("relationalOperator", [], p[1])


def p_simpleExpression(p):
    """simpleExpression : term
    | sign term
    | simpleExpression additiveOperator term"""
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 3:
        p[0] = Node("simpleExpression", [p[1], p[2]])
    else:
        p[0] = Node("simpleExpression", [p[1], p[2], p[3]])


def p_additiveOperator(p):
    """additiveOperator : PLUS
    | MINUS
    | OR"""
    p[0] = Node("additiveOperator", [], p[1])


def p_term(p):
    """term : factor
    | term multiplicativeOperator factor"""
    if len(p) > 2:
        p[0] = Node("term", [p[1], p[2], p[3]])
    else:
        p[0] = p[1]


def p_multiplicativeOperator(p):
    """multiplicativeOperator : STAR
    | SLASH
    | DIV
    | MOD
    | AND"""
    p[0] = Node("multiplicativeOperator", [], p[1])


def p_factor(p):
    """factor : variable
    | unsignedConstant
    | LPAREN expression RPAREN"""
    if len(p) == 2:
        # Handle variable or unsigned constant
        p[0] = p[1]
    else:
        # Handle parenthesized expression
        p[0] = Node("parenthesizedExpression", [p[2]])


def p_unsignedConstant(p):
    """unsignedConstant : unsignedNumber
    | string
    | NIL
    | TRUE
    | FALSE"""
    if p[1] == "TRUE" or p[1] == "FALSE":
        p[0] = Node("booleanConstant", [], p[1])
    else:
        p[0] = p[1]


def p_procedureStatement(p):
    """procedureStatement : identifier
    | identifier LPAREN expressionList RPAREN"""
    if len(p) > 2:
        p[0] = Node("procedureStatement", [p[1], p[3]])
    else:
        p[0] = Node("procedureStatement", [p[1]])


def p_expressionList(p):
    """expressionList : expressionList COMMA expression
    | expressionList COMMA formattedExpression
    | expression
    | formattedExpression"""
    if len(p) > 2:
        p[1].children.append(p[3])
        p[0] = p[1]
    else:
        p[0] = Node("expressionList", [p[1]])


def p_formattedExpression(p):
    """formattedExpression : variable COLON expression COLON expression
    | variable COLON expression"""
    if len(p) > 4:
        p[0] = Node("formattedExpression", [p[1], p[3], p[5]])
    else:
        p[0] = Node("formattedExpression", [p[1], p[3]])


def p_emptyStatement_(p):
    """emptyStatement_ :"""
    p[0] = Node("emptyStatement")


def p_structuredStatement(p):
    """structuredStatement : compoundStatement
    | conditionalStatement
    | loopStatement"""
    p[0] = p[1]


def p_loopStatement(p):
    """loopStatement : forStatement
    | whileStatement
    | repeatStatement"""
    p[0] = p[1]


def p_forStatement(p):
    """forStatement : FOR identifier ASSIGN expression TO expression DO statement
    | FOR identifier ASSIGN expression DOWNTO expression DO statement"""
    if p[5] == "to":
        p[0] = Node("forStatement", [p[2], p[4], p[6], p[8]], "to")
    else:
        p[0] = Node("forStatement", [p[2], p[4], p[6], p[8]], "downto")


# Add while statement rule
def p_whileStatement(p):
    """whileStatement : WHILE expression DO statement"""
    p[0] = Node("whileStatement", [p[2], p[4]])


# Add repeat statement rule
def p_repeatStatement(p):
    """repeatStatement : REPEAT statements UNTIL expression"""
    p[0] = Node("repeatStatement", [p[2], p[4]])


def p_conditionalStatement(p):
    """conditionalStatement : ifStatement"""
    p[0] = p[1]


def p_ifStatement(p):
    """ifStatement : IF expression THEN statement
    | IF expression THEN statement ELSE statement"""
    if len(p) > 5:
        p[0] = Node("ifStatement", [p[2], p[4], p[6]])
    else:
        p[0] = Node("ifStatement", [p[2], p[4]])


def p_empty(p):
    """empty :"""
    p[0] = None


def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}', line {p.lineno}")
    else:
        print("Syntax error at EOF")


# Build the parser
parser = yacc.yacc()


# Test function
def parse(data):
    result = parser.parse(data)
    return result


def main(filepath):
    try:
        with open(filepath, "r") as file:
            data = file.read()
            result = parse(data)
            print(result)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Parsing completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
    else:
        main(sys.argv[1])  # passa apenas o caminho do arquivo
