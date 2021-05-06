import json
from typing import List

from bert_seq2seq.tokenizer import load_chinese_base_vocab, T5PegasusTokenizer
from node import Node
import ply.lex as  lex
import ply.yacc as yacc
def make_parser():
    vocab_path = r"D:\codeproject\NLP\models\chinese_t5_pegasus_small\vocab.txt"
    word2idx = load_chinese_base_vocab(vocab_path)
    tokenizer = T5PegasusTokenizer(word2idx)

    def tokenize(s:str):
        res = tokenizer.tokenize(s)
        return [token for token in res[1:-1]]

    def make_sub_number_tree(l: List[str]):
        last = None
        for i,substr in enumerate(l[::-1]):
            if i==0:
                tmp = Node(json.dumps(["subnum_0", substr]))
            else:
                tmp = Node(json.dumps(["subnum_1", substr]))
            tmp.right = last
            last = tmp
        return last

    tokens = (
        'NAME', 'INT', 'FLOAT',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'EQUALS', 'POWER',
        'LPAREN', 'RPAREN',
    )



    # Tokens

    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_POWER = r'\*\*'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_EQUALS = r'='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'

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
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE'),
        ('left', 'POWER'),
        ('right', 'UMINUS',"UPLUS"),
    )
    names = {'x': 'x'}
    res_head = None

    def p_equation_assign(p):
        'equation : expression EQUALS expression'
        temp = Node(json.dumps(["operator_2",p[2]]), p[1], p[3])
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
        p[0] = Node(json.dumps(["operator_2",p[2]]), p[1], p[3])
        global res_head
        res_head = p[0]

    def p_expression_uminus(p):
        'expression : MINUS expression %prec UMINUS'
        p[0] = Node(json.dumps(["operator_1",'-']), left=None, right=p[2])
        global res_head
        res_head = p[0]

    def p_expression_uplus(p):
        'expression : PLUS expression %prec UPLUS'
        p[0] = Node(json.dumps(["operator_1",'+']), left=None, right=p[2])
        global res_head
        res_head = p[0]

    def p_expression_group(p):
        'expression : LPAREN expression RPAREN'
        p[0] = p[2]
        global res_head
        res_head = p[0]

    def p_expression_int(p):
        'expression : INT'
        s = str(p[1])
        l = tokenize(s)
        global res_head
        if len(l)==1:
            p[0] = Node(json.dumps(("int_0", l[0])))
            res_head = p[0]
        else:
            # p[0] = Node(json.dumps(("int", p[1])))
            node1 = Node(json.dumps(("int_1", l[0])))
            node1.right = make_sub_number_tree(l[1:])
            p[0] = node1
            res_head = p[0]

    def p_expression_float(p):
        'expression : FLOAT'
        s = str(p[1])
        l = tokenize(s)
        global res_head
        if len(l)==1:
            p[0] = Node(json.dumps(("float_1", l[0])))

            res_head = p[0]
        else:
            # p[0] = Node(json.dumps(("int", p[1])))
            node1 = Node(json.dumps(("float_2", l[0])))
            node1.right = make_sub_number_tree(l[1:])
            p[0] = node1
            res_head = p[0]


    def p_expression_name(p):
        'expression : NAME'
        try:
            p[0] = Node(json.dumps(["var_0",p[1]]))
        except LookupError:
            print(f"Undefined name {p[1]!r}")
            p[0] = 0
        global res_head
        res_head = p[0]

    def p_error(p):
        print(f"Syntax error at {p}")

    # dictionary of names (for storing variables)
    names = {'x': 'x'}

    parser = yacc.yacc()
    def input(text):
        result = parser.parse(text, lexer=lexer)
        return result

    return input

if __name__ == '__main__':
    calc = make_parser()
    # print(calc("x = -(-12.99/4)**-3*2+1"))
    # testdata = "x = -(-12.99/4)**-3*2+1"
    testdata = "x=8/(1/-8000000)/100000.0"
    res:Node = calc(testdata)
    res = res.add_root_node(res)
    print(testdata)
    res.display()
    print(res.pre_order_traverse(res))





