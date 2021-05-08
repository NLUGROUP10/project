import argparse
import json
from collections import OrderedDict
from typing import Tuple

import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader
from bert_seq2seq.tokenizer import load_chinese_base_vocab
from seq2tree import T5SmallConfig, Seq2TreeModel
from tree_dataloader import Seq2TreeData
from utils import values, types


class LightningModel(pl.LightningModule):
    def __init__(self,args):
        super(LightningModel, self).__init__()
        self.args = args
        self.__build_model(self.args)

    def __build_model(self,args):
        config = T5SmallConfig()
        config.types_vocab_size = args.types_vocab_size
        config.values_vocab_size = args.values_vocab_size
        self.model = Seq2TreeModel(config)


    def forward(self,batch:Tuple):
        token_ids_padded, types_ids_padded, values_ids_padded, positions_ids_padded, rels_ids_padded, labels_types_ids, labels_values_ids = batch
        decoder_input_dict = {
            "types": types_ids_padded,
            "values": values_ids_padded,
            "tree_positions": positions_ids_padded,
            "rel_tokens":rels_ids_padded
        }
        res = self.model(
            input_ids=token_ids_padded,
            decoder_input_dict = decoder_input_dict
        )
        print(res)


class TreeCELoss(nn.Module):
    def __init__(self, ignore_idx=-100):
        super(TreeCELoss, self).__init__()
        self.ignore = ignore_idx
        self.lossfn  = nn.CrossEntropyLoss(ignore_index=self.ignore)

    def forward(self, types_pred, values_pred, types_label, values_label):
        loss1 = self.lossfn(types_pred.view(-1,types_pred.size(-1)), types_label.view(-1))
        loss2 = self.lossfn(values_pred.view(-1,values_pred.size(-1)), values_label.view(-1))
        loss = loss1+loss2
        return loss


if __name__ == '__main__':

    vocab_path = r"D:\codeproject\NLP\models\chinese_t5_pegasus_small\vocab.txt"
    word2idx = load_chinese_base_vocab(vocab_path)
    outpath = f"C:\\Users\\tianshu\\PycharmProjects\\project\\data\\ape\\cleaned\\test.ape.json"
    test_data = Seq2TreeData(outpath, word2idx, values, types)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--path_model_1", type=str, required=False)
    parser.add_argument("--path_model_2", type=str, required=False)
    parser.add_argument("--base_dir_1", type=str, required=False)
    parser.add_argument("--base_dir_2", type=str, required=False)
    parser.add_argument("--use_classic_ens", type=bool, default=False)
    args = parser.parse_args()
    args.types_vocab_size = len(test_data.types_vocab.list)
    args.values_vocab_size = len(test_data.values_vocab.list)

    model = Model(args)
    print(model)
    dataloader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=test_data.collect_funtion)

    for i, batch in enumerate(dataloader):
        res = model.forward(batch)





