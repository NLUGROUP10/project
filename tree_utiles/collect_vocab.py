import itertools
import json

import tqdm

from tree_utiles.node import Node


def collect_vocab_from_file(path):
    types_set = set()
    values_set = set()
    rels_set = set()
    max_depth = 0
    with open(path,"r",encoding="utf-8") as inputfile:
        for line in tqdm.tqdm(inputfile):
            line = json.loads(line)
            text = line[0]
            tgt_dict = json.loads(line[-1])
            # print(tgt_dict)
            types = tgt_dict["types"]
            values = tgt_dict["values"]
            rels = tgt_dict['rels']
            rels = list(itertools.chain(*rels))
            # for rel in rels:
            #     u,d = rel.split("|")
            #     u = int(u)
            #     d = int(d)
            #     if u>40 or d>40:
            #         print(u,d,rel)
            #         s = tgt_dict["serialize"]
            #         tree = Node.deserialize(s)
            #         tree.display()
            #         print(text)
            #         break
            # for rel in rels:
            #     u,d = rel.split("|")
            #     u = int(u)
            #     d = int(d)
            #     max_depth = max([u,d,max_depth])

            types_set.update(set(types))
            values_set.update(set(values))
            rels_set.update(set(rels))
        return types_set,values_set,rels_set











if __name__ == '__main__':

    splits = ["train","test"]
    types_set, values_set, rels_set = {},{},{}
    for split in splits:
        # path = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\{split}.ape.json"
        outpath = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\cleaned\\{split}.ape.json"
        types, values, rels =  collect_vocab_from_file(outpath)
        types_set.update(types)
        values_set.update(values)
        rels_set.update(rels)

