program: programHeading block DOT EOF

programHeading : PROGRAM identifier SEMI ;

identifier:  IDENT ;

block : declarations compoundStatement

declarations : declarations declaration
             | 

declaration : constantDefinitionBlock
            | typeDeclarationBlock
            | variableDeclarationBlock
            | procedureAndFunctionDeclarationBlock



constantDefinitionBlock: CONST constanteDefinitionList

constanteDefinitionList : constanteDefinitionList constantDefinition SEMI
                        | constantDefinition SEMI


constanteDefinition : identifier EQUAL constant ;

constantChr : CHR LPAREN unsignedInteger RPAREN ;

constant : unsignedNumber
         | sign unsignedNumber
         | identifier
         | sign identifier 
         | string
         | constantChr
         ;

unsignedNumber : unsignedInteger
               | unsignedReal
               ;

unsignedInteger : NUM_INT ;

unsignedReal : NUM_REAL ;

sign : PLUS
     | MINUS 
     ;

bool_ : TRUE
      | FALSE
      ;

string : STRING_LITERAL ;

typeDeclarationBlock : TYPE typeDefinitionList

typeDefinitionList : typeDefinitionList typeDefinition SEMI
                   | typeDefinition SEMI


typeDefinition : identifier EQUAL type_ 

type_ : scalarType
      | subrangeType
      | identifierType
      | stringType
      | structuredType
      ;

scalarType : LPAREN identifierList RPAREN ;

subrangeType : constant DOTDOT constant ;

identifierType : identifier
               |  (CHAR | BOOLEAN | INTEGER | REAL | STRING)
               ;

stringtype : STRING LBRACK (identifier | unsignedNumber) RBRACK ;

structuredType : PACKED unpackedStructuredType
               | unpackedStructuredType ;

unpackedStructuredType : arrayType
                       | recordType
                       | setType ;

arrayType : ARRAY LBRACK typeList RBRACK OF componentType
          | ARRAY LBRACK2 typeList RBRACK2 OF componentType ;



typeList : typeList COMMA indexType
         | indexType

indexType : simpleType ;

componentType : type_ ;

recordType : RECORD fieldList? END ;

recordType : RECORD fieldList_opt END 

fieldList_opt : fieldList
              | empty


fieldList : fixedPart (SEMI variantPart)?
          | variantPart ;

fieldList : fixedPart variantPart_opt
          | variantPart


variantPart_opt : SEMI variantPart
                | empty


fixedPart : recordSection (SEMI recordSection)* ;

fixedPart : recordSectionList

recordSectionList : recordSectionList SEMI recordSection
                  | recordSection


recordSection : identifierList COLON type_ ;

variantPart : CASE tag OF variant (SEMI variant)* ;

variantPart : CASE tag OF variantList

variantList : variantList SEMI variant
            | variant


tag : identifier COLON typeIdentifier
    | typeIdentifier ;

variant : constList COLON LPAREN fieldList RPAREN ;

setType : SET OF baseType ;

baseType : simpleType ;



variableDeclarationBlock : VAR variableDeclarationList SEMI

variableDeclarationList : variableDeclarationList SEMI variableDeclaration
                        | variableDeclaration



variableDeclaration : identifierList COLON type_ ;



procedureAndFunctionDeclarationBlock : procedureAndFunctionDeclarationDeclaration SEMI ;

procedureAndFunctionDeclarationDeclaration : procedureDeclaration
                                           | functionDeclaration 
                                           ;
                                        


procedureDeclaration : PROCEDURE identifier formalParameterList_opt SEMI block

formalParameterList_opt : formalParameterList
                        | empty


formalParameterList : LPAREN formalParameterSection (SEMI formalParameterSection)* RPAREN ;

formalParameterList : LPAREN formalParameterSectionList RPAREN

formalParameterSectionList : formalParameterSectionList SEMI formalParameterSection
                           | formalParameterSection


formalParameterSection : parameterGroup
                       | VAR parameterGroup
                       | FUNCTION parameterGroup
                       | PROCEDURE parameterGroup
                       ;

parameterGroup : identifierList COLON typeIdentifier ;


identifierList : identifierList COMMA identifier
               | identifier


constList : constantList COMMA constant
          | constant


functionDeclaration : FUNCTION identifier (formalParameterList)? COLON resultType SEMI block ;
functionDeclaration : FUNCTION identifier formalParameterList_opt COLON resultType SEMI block


resultType : typeIdentifier ;


statement : unlabelledStatement ;

unlabelledStatement : simpleStatement
                    | structuredStatement
                    ;

simpleStatement : assignmentStatement 
                | procedureStatement
                | emptyStatement_
                ;

assignmentStatement : variable ASSIGN expression ;

variable : (AT identifier | identifier) ( LBRACK expression (COMMA expression)* RBRACK
                                        | LBRACK2 expression (COMMA expression)* RBRACK2
                                        | DOT identifier
                                        )*
                                        ;

variable : variableBase variableSuffixList

variableBase : AT identifier
             | identifier

variableSuffixList : variableSuffixList variableSuffix
                   | empty

variableSuffix : LBRACK expressionList RBRACK
               | LBRACK2 expressionList RBRACK2
               | DOT identifier


expressionList : expressionList COMMA expression
               | expression


expression : simpleExpression relationaloperator_opt

relationaloperator_opt : relationaloperator expression
                       | empty

relationaloperator : EQUAL
                   | NOT_EQUAL
                   | LT
                   | LE
                   | GE
                   | GT
                   | IN
                   ;

simpleExpression : term additiveoperator_opt;

additiveoperator_opt : additiveoperator simpleExpression
                     | empty

additiveoperator : PLUS
                 | MINUS
                 | OR
                 ;

term : signedFactor (multiplicativeoperator term)? ;

term : signedFactor multiplicativeoperator_opt

multiplicativeoperator_opt : multiplicativeoperator term
                           | empty

multiplicativeoperator : STAR
                       | SLASH
                       | DIV
                       | MOD
                       | AND
                       ;

signedFactor : PLUS factor
             | MINUS factor
             | factor

factor : variable
       | LPAREN expression RPAREN
       | functionDesignator
       | unsignedConstant
       | set_
       | NOT factor
       | bool_
       ;

unsignedConstant : unsignedNumber
                 | constantChr
                 | string
                 | NIL
                 ;

functionDesignator : identifier LPAREN parameterList RPAREN ;

parameterList : actualParameterList_opt

actualParameterList_opt : actualParameterList
                        | empty

actualParameterList : actualParameterList COMMA actualParameter
                    | actualParameter


set_ : LBRACK elementList RBRACK
     | LBRACK2 elementList RBRACK2
     ;

elementList : element (COMMA element)* ;

elementList : elementList COMMA element
            | element


element : expression DOTDOT expression
        | expression



procedureStatement : identifier (LPAREN parameterList RPAREN)? ;
procedureStatement : identifier
                   | identifier LPAREN parameterList RPAREN





actualParameter : expression parameterwidthList

parameterwidthList : parameterwidthList parameterwidth
                   | empty


parameterwidth : COLON expression ;

emptyStatement_ : ;

empty_: /* empty */ ;



structuredStatement : compoundStatement 
                    | conditionalStatement
                    | repetetiveStatement
                    | withStatement
                    ;

compoundStatement : BEGIN statements END ;


statements : statements SEMI statement
           | statement

conditionalStatement : ifStatement
                     | caseStatement
                     ;


ifStatement : IF expression THEN statement ELSE statement
            | IF expression THEN statement

caseStatement : CASE expression OF caseListElementList caseElse_opt END

caseListElementList : caseListElementList SEMI caseListElement
                    | caseListElement

caseListElement : constList COLON statement ;

repetetiveStatement : whileStatement
                    | repeatStatement
                    | forStatement
                    ;

whileStatement : WHILE expression DO statement ;

repeatStatement : REPEAT statements UNTIL expression ;

forStatement : FOR identifier ASSIGN forList DO statement ;

forList : initialValue (TO | DOWNTO) finalValue ;

initialValue : expression ;

finalValue : expression ;

withStatement : WITH recordVariableList DO statement ;

recordVariableList : variable
                   | recordVariableList COMMA variable

AND : 'AND' ;

ARRAY : 'ARRAY' ;

BEGIN : 'BEGIN' ;

BOOLEAN : 'BOOLEAN' ;

CASE : 'CASE' ;

CHAR : 'CHAR' ;

CHR : 'CHR' ;

CONST : 'CONST' ;

DIV : 'DIV' ;

DO : 'DO' ;

DOWNTO : 'DOWNTO' ;

ELSE : 'ELSE' ;

END : 'END' ;

FILE : 'FILE' ;

FOR : 'FOR' ;

FUNCTION : 'FUNCTION' ;

GOTO : 'GOTO' ;

IF : 'IF' ;

IN : 'IN' ;

INTEGER : 'INTEGER' ;

LABEL : 'LABEL' ;

MOD : 'MOD' ;

NIL : 'NIL' ;

NOT : 'NOT' ;

OF : 'OF' ;

OR : 'OR' ;

PACKED : 'PACKED' ;

PROCEDURE : 'PROCEDURE' ;

PROGRAM : 'PROGRAM' ;

REAL : 'REAL' ;

RECORD : 'RECORD' ;

REPEAT : 'REPEAT' ;

SET : 'SET' ;

THEN : 'THEN' ;

TO : 'TO' ;

TYPE : 'TYPE' ;

UNTIL : 'UNTIL' ;

VAR : 'VAR' ;

WHILE : 'WHILE' ;

WITH : 'WITH' ;

PLUS : '+' ;

MINUS : '-' ;

STAR : '*' ;

SLASH : '/' ;

ASSIGN : ':=' ;

COMMA : ',' ;

SEMI : ';' ;

COLON : ':' ;

EQUAL : '=' ;

NOT_EQUAL : '<>' ;

LT : '<' ;

LE : '<=';

GE : '>=' ;

GT : '>' ;

LPAREN : '(' ;

RPAREN : ')' ;

LBRACK : '[' ;

LBRACK2 : '(.';

RBRACK : ']' ;

RBRACK2 : '.)' ;

POINTER : '^' ;

AT : '@';

DOT : '.' ;

DOTDOT : '..' ;

LCURLY : '{' ;

RCURLY : '}' ;

UNIT : 'UNIT' ;

INTERFACE : 'INTERFACE' ;

STRING : 'STRING' ;

IMPLEMENTATION : 'IMPLEMENTATION' ;

TRUE : 'TRUE' ;

FALSE : 'FALSE' ;

WS : [ \t\r\n] -> skip ;

COMMENT_1 : '(*' .*? '*)' -> skip ;

COMMENT_2 : '{' .*? '}' -> skip ;

IDENT : ('A' .. 'Z') ('A' .. 'Z' | '0' .. '9' | '_')* ;

STRING_LITERAL : '\'' ('\'\'' | ~ ('\''))* '\'' ;

NUM_INT : ('0' .. '9')+ ;

NUM_REAL : ('0' .. '9')+ (('.' ('0' .. '9')+ (EXPONENT)?)? | EXPONENT) ;

fragment EXPONENT : ('E') ('+' | '-')? ('0' .. '9')+ ;