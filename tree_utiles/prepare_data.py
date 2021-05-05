import re

from closureparser import make_parser
import json


def remove_bucket(equation):
    """去掉冗余的括号
    """
    l_buckets, buckets = [], []
    for i, c in enumerate(equation):
        if c == '(':
            l_buckets.append(i)
        elif c == ')':
            buckets.append((l_buckets.pop(), i))
    eval_equation = eval(equation)
    for l, r in buckets:
        new_equation = '%s %s %s' % (
            equation[:l], equation[l + 1:r], equation[r + 1:]
        )
        try:
            if is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                equation = new_equation
        except:
            pass
    return equation.replace(' ', '')

def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b

## 苏神baseline 读取数据
def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    # index = 0
    for l in open(filename,encoding="utf-8"):
        # index += 1
        # if index == 100:
        #     break
        l = json.loads(l)
        # print(l)
        question, equation, answer = l['original_text'], l['equation'], l['ans']
        # 处理带分数
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        # 分数去括号
        question = re.sub('\((\d+/\d+)\)', '\\1', question)
        # 处理百分数
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        # 冒号转除号、剩余百分号处理
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        try:
            # print(equation)
            # print(answer)
            # print("~~~~~~~`")
            if is_equal(eval(equation), eval(answer)):
                D.append([question, remove_bucket(equation), answer])
        except Exception as e:
            print(e)
            continue
    for data in D:
        data[1]= "x=" + data[1]

    return D

if __name__ == '__main__':
    path = r"C:\Users\tianshu\PycharmProjects\project\data\ape\valid.ape.json"
    datas = load_data(path)
    cal = make_parser()
    for data in datas:
        # print(cal(data[1]),"\t\t\t",data)
        print(data)
        res = cal(data[1])
        res.display()
        # print(res)
        if res==None:
            print(data)
    # print(datas)

