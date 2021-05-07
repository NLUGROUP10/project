import re

import torch
from bert_seq2seq.utils import load_gpt
import sys

# sys.path.append("/Users/xingzhaohu/Downloads/code/python/ml/ml_code/bert/bert_seq2seq")
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
# import pandas as pd
import numpy as np
import os
import json
import time
import  glob
import bert_seq2seq
from torch.utils.data import Dataset, DataLoader
from bert_seq2seq.t5_ch import T5Model
from bert_seq2seq.tokenizer import T5PegasusTokenizer, load_chinese_base_vocab
from bert_seq2seq.t5_ch import T5Model

train_data_path ="/home/zp2053/codes/nlu/ape/train.ape.json"

val_data_path = "/home/zp2053/codes/nlu/ape/valid.ape.json"

vocab_path = "/scratch/zp2053/bert_seq2seq/vocab/t5_small_vocab.txt"
model_path = "/scratch/zp2053/bert_seq2seq/weights/t5_small_pytorch_model.bin"
model_save_path = "/scratch/zp2053/bert_seq2seq/weights/t5_small_math_ques_model.bin"
batch_size = 16
lr = 1e-5
word2idx = load_chinese_base_vocab(vocab_path)
tokenizer = T5PegasusTokenizer(word2idx)


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
                D.append((question, remove_bucket(equation), answer))
        except Exception as e:
            print(e)
            continue
    return D

class SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """

    def __init__(self, data):
        ## 一般init函数是加载所有数据
        super(SeqDataset, self).__init__()
        # 读原始数据
        # self.sents_src, self.sents_tgt = read_corpus(poem_corpus_dir)
        self.data = data

        self.idx2word = {k: v for v, k in word2idx.items()}
        self.maxlen = 256
        self.tokenizer = tokenizer
    def __getitem__(self, i):

        single_data = self.data[i]
        # print(single_data)
        original_text = single_data[0]
        equation = single_data[1]

        token_ids_src, _ = tokenizer.encode(original_text, max_length=256)
        token_ids_tgt, _ = tokenizer.encode(equation, max_length=256)
        output = {
            "token_ids_src": token_ids_src,
            "token_ids_tgt": token_ids_tgt,
        }
        return output

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """

    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids_src = [data["token_ids_src"] for data in batch]
    max_length_src = max([len(t) for t in token_ids_src])
    token_ids_tgt = [data["token_ids_tgt"] for data in batch]
    max_length_tgt = max([len(t) for t in token_ids_tgt])

    token_ids_padded = padding(token_ids_src, max_length_src)
    target_ids_padded = padding(token_ids_tgt, max_length_tgt)
    labels_ids = target_ids_padded.clone()
    labels_ids[labels_ids == 0] = -100
    target_ids_padded = target_ids_padded[:, :-1].contiguous()
    labels_ids = labels_ids[:, 1:].contiguous()

    return token_ids_padded, target_ids_padded, labels_ids


class Trainer:
    def __init__(self):
        # 加载数据
        data = load_data(train_data_path)

        # 判断是否有可用GPU
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: " + str(self.device))
        # 定义模型
        self.model = T5Model(word2idx,size = "small")
        ## 加载预训练的模型参数～
        self.model.load_pretrain_params(model_path)
        # 将模型发送到计算设备(GPU或CPU)
        self.model.set_device(self.device)
        # 声明需要优化的参数
        self.optim_parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)
        # 声明自定义的数据加载器
        self.dataset = SeqDataset(data)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.best_acc = 0.0

    def train(self, epoch):
        # 一个epoch的训练
        self.model.train()
        self.iteration(epoch, dataloader=self.dataloader, train=True)

    def save(self, save_path):
        """
        保存模型
        """
        self.model.save_all_params(save_path)
        print("{} saved!".format(save_path))

    def iteration(self, epoch, dataloader, train=True):
        total_loss = 0
        report_loss = 0
        start_time = time.time()  ## 得到当前时间
        step = 0
        for token_ids, target_ids, labels_ids in tqdm(dataloader):
            step += 1
            # print(token_ids.shape)
            # print(target_ids.shape)
            # print(labels_ids.shape)
            if step % 500 == 0:
                self.save(model_save_path)
                self.model.eval()
                test_data = ["王艳家买了一台洗衣机和一台电冰箱，一共花了6000元，电冰箱的价钱是洗衣机的3/5，求洗衣机的价钱．",
                             "六1班原来男生占总数的2/5，又转来5名男生，现在男生占总数的5/11，女生有多少人？",
                             "两个相同的数相乘，积是3600，这个数是多少."]
                for text in test_data:
                    print(self.model.sample_generate_encoder_decoder(text, add_eos=True))
                self.model.train()
                print("report loss is " + str(report_loss))

            if step % 10000 == 0:
                ## 2000步集中测试一下
                print("validing..........")
                acc = self.validation()
                print("valid acc is " + str(acc))
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save(model_save_path)

            # 因为传入了target标签，因此会计算loss并且返回
            loss = self.model(token_ids,labels=labels_ids, decoder_input_ids=target_ids)[0]
            # 反向传播
            if train:
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播, 获取新的梯度
                loss.backward()
                # 用获取的梯度更新模型参数
                self.optimizer.step()

            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            report_loss += loss.item()




        end_time = time.time()
        spend_time = end_time - start_time
        # 打印训练信息
        print("epoch is " + str(epoch) + ". loss is " + str(total_loss) + ". spend time is " + str(spend_time))
        # 保存模型
        self.save(model_save_path)

    def validation(self):
        val_data = load_data(val_data_path)
        self.model.eval()
        right = 0.0
        num = len(val_data)
        # for each_data in tqdm(val_data, total=num):
        for each_data in val_data:
            equation = self.model.sample_generate_encoder_decoder(each_data[0],add_eos=True)

            pred_ans = self.model(equation.replace(" ", ""))
            ans1 = each_data[2]
            try:
                if "/" in each_data[2] or "+" in each_data[2] or "-" in each_data[2] or "*" in each_data[2]:
                    # print(each_data[2])
                    # equation1 = re.sub('\((\d+/\d+)\)', '\\1', str(each_data[2]))
                    ans1 = eval(each_data[2])
                if abs(float(pred_ans) - float(ans1)) < 0.01:
                    right += 1
                    # print("right! pred is " + str(pred_ans) + " ans is " + str(each_data[2]))
                else:
                    pass
                    # print("err! pred is " + str(pred_ans) + " ans is " + str(each_data[2]))
            except Exception as e:
                print(e)

        self.model.train()
        return right / num

    def sanitycheck(self):
        for i in range(5):
            example = next(iter(self.dataloader))
            # print(example)
            srcs,tgts,lables = example
            # intput_tokens  = example['token_ids_src']
            # tgt_tokens = example['token_ids_tgt']
            for i in range(srcs.shape[0]):
                intput_tokens = srcs[i,:].tolist()
                tgt_tokens = tgts[i,:].tolist()
                # lable = lables[i,:].tolist()
                print(tokenizer.decode(intput_tokens))
                print(tokenizer.decode(tgt_tokens))
                # print(tokenizer.decode(lable))



if __name__ == '__main__':

    trainer = Trainer()
    model = trainer.model
    encoder = model.model.encoder
    save_path = "./encoder.bin"
    torch.save(encoder.state_dict(), save_path)
    train_epoches = 10
    trainer.sanitycheck()
    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
