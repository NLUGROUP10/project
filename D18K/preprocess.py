import json
import pprint
with open("dev_cleaned.json","r",encoding="UTF-8") as inputfile:
    res = json.load(inputfile)
print(len(res))
with open("eval_cleaned.json","r",encoding="UTF-8") as inputfile:
    res = json.load(inputfile)
print(len(res))
for i in res:
    pprint.pprint(i["text"])