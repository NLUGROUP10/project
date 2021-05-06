import re
import pprint

from bert_seq2seq.tokenizer import load_chinese_base_vocab, T5PegasusTokenizer
from closureparser import make_parser
import json

from tree_utiles.node import Node
import numpy as np

def get_data_from_root(root:Node,width = 2, depth = 10):

    pre_order = root.pre_order_traverse(root)
    types = []
    values = []
    paths = []
    rels = []
    onehots =  np.identity(width, dtype=int).tolist()
    def longestCommonPrefix(strs):
        ret = ''
        for i in zip(*strs):
            if len(set(i)) == 1:
                ret += i[0]
            else:
                break
        return ret

    def path2positioncode(path:str):
        res = []
        final = [0]*(width*depth)
        path = path[1:]
        path = path[::-1]
        for c in path:
            res = res+onehots[int(c)]
        res = res[:depth*width]
        final[:len(res)] = res
        return final
    num_node = len(pre_order)
    for i in range(num_node):
        line_i = []
        for j in range(num_node):
            path1 = pre_order[i][1]
            path2 = pre_order[j][1]
            common = longestCommonPrefix([path1,path2])
            upper = len(path1)-len(common)
            down  = len(path2)-len(common)
            rel = f"{upper}|{down}"
            line_i.append(rel)
        rels.append(line_i)
    for i in pre_order:
        types.append(json.loads(i[0])[0])
        values.append(json.loads(i[0])[1])
        paths.append(path2positioncode(i[1]))
    res_dict = {"types":types,"values":values,"paths":paths,"rels":rels,"serialize":root.serialize(root)}
    return json.dumps(res_dict,ensure_ascii=False)



## 苏神baseline 读取数据
def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """

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
                D.append([question,answer, remove_bucket(equation) ])
        except Exception as e:
            print(e)
            continue
    for data in D:
        data[2]= "x=" + data[2]
    return D

def prepare(path_in,path_save):
    datas = load_data(path_in)
    cal = make_parser()
    D = []
    for data in datas:
        # print(cal(data[1]),"\t\t\t",data)
        # print(data)
        res = cal(data[2])
        # res.display()
        # print(res)
        tree = get_data_from_root(res)
        D.append(json.dumps([data[0],data[2],data[1],tree],ensure_ascii=False))
    with open(path_save,"w+",encoding="utf-8") as outpath:
        for line in D:
            outpath.write(line)
            outpath.write('\n')
if __name__ == '__main__':
    splits = ["train","test","valid"]
    for split in splits:
        path = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\{split}.ape.json"
        outpath = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\cleaned\\{split}.ape.json"
        prepare(path,outpath)




