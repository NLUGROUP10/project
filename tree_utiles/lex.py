import json
tokens = (
    'NAME','INT','FLOAT',
    'PLUS','MINUS','TIMES','DIVIDE','EQUALS','POWER',
    'LPAREN','RPAREN',
    )

# Tokens

t_PLUS    = r'\+'
t_MINUS   = r'-'
t_POWER = r'\*\*'
t_TIMES   = r'\*'
t_DIVIDE  = r'/'
t_EQUALS  = r'='
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_NAME    = r'[a-zA-Z_][a-zA-Z0-9_]*'
def t_FLOAT(t):
    r'([0-9]*\.[0-9]+)'
    t.value = float(t.value)
    return t

def t_INT(t):
    r'\d+'
    t.value = int(t.value)
    return t




# Ignored characters
t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    print(f"Illegal character {t.value[0]!r}")
    t.lexer.skip(1)

# Build the lexer
import ply.lex as lex
lexer = lex.lex()

# Precedence rules for the arithmetic operators
precedence = (
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    ('right','UMINUS'),
    )
class Node:
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right
    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.val
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.val
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.val
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.val
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2
# dictionary of names (for storing variables)
names = { 'x':'x' }
res_head = None
def p_equation_assign(p):
    'equation : expression EQUALS expression'
    temp = Node('=',p[1],p[3])
    p[0] = temp
    # print((p[0]))
    global res_head
    res_head = p[0]



def p_expression_binop(p):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POWER expression'''
    p[0] = Node(p[2], p[1], p[3])
    global res_head
    res_head = p[0]


def p_expression_uminus(p):
    'expression : MINUS expression %prec UMINUS'
    p[0] = Node('-',left=None,right=p[2])
    global res_head
    res_head = p[0]

def p_expression_group(p):
    'expression : LPAREN expression RPAREN'
    p[0] = p[2]
    global res_head
    res_head = p[0]


def p_expression_int(p):
    'expression : INT'
    p[0] = Node(json.dumps(("int",p[1])))
    global res_head
    res_head = p[0]

def p_expression_float(p):
    'expression : FLOAT'
    p[0] = Node(json.dumps(("float",p[1])))
    global res_head
    res_head = p[0]

def p_expression_name(p):
    'expression : NAME'
    try:
        p[0] = Node(names[p[1]])
    except LookupError:
        print(f"Undefined name {p[1]!r}")
        p[0] = 0
    global res_head
    res_head = p[0]

def p_error(p):
    print(f"Syntax error at {p}")
# Build the parser
import ply.yacc as yacc
import collections
yacc.yacc(debug=True)
# test_data = "x = -(-12.99/4)**-3+(-1/22+2*3)"
test_data = "1.1**4"
yacc.parse(test_data)
def traverse(root):
    r = []
    queue = collections.deque()
    if not root:
        return []
    queue.append(root)
    while queue:
        n = queue.popleft()
        r.append(n.val)
        if n.left and type(n.left)==Node:
            queue.append(n.left)
        if n.right and type(n.right)==Node:
            queue.append(n.right)
    return r
print(test_data)
print("-------parsed tree---------")
def printTree(node, level=0):
    if node != None:
        printTree(node.left, level + 1)
        print(' ' * 4 * level + '->', node.val)
        printTree(node.right, level + 1)
# printTree(res_head)

res_head.display()