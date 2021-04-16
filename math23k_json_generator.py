# -*- coding: utf-8 -*-
"""math23k_json_generator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w2pCeqQKDn6itroovt-iXjJJQExMZn2o
"""

import json

def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
  print("Reading lines...")
  f = open(filename, encoding="utf-8")
  js = ""
  data = []
  for i, s in enumerate(f):
    # print([i, s])
    js += s
    i += 1
    if i % 7 == 0:  # every 7 line is a json
      data_d = json.loads(js)
      if "千米\小时" in data_d["original_text"]:
        js = ""
        continue
      data.append(data_d)
      js = ""

  return data

data = load_raw_data("data/raw_math23k.json")

x = [d['original_text'] for d in data]

f = open('data/math23k_train.json', 'w')
f.write('[')
for i in x:
  f.write('\"' + i + '\"')
  if i != x[-1]:
    f.write(', ')

f.write(']')
f.close()
