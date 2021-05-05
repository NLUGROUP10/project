from collections import OrderedDict

import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader

from seq2tree.seq2tree import T5SmallConfig, Seq2TreeModel, MaskedLoss


class LightningModel(pl.LightningModule):
    def __init__(self,args):
        super(LightningModel, self).__init__()
        self.args = args
        self.lr = 0
        self.__prepare_data(self.args)
        self.__build_model(self.args)
        self.hparams = self.args

    def __build_model(self,args):
        config = T5SmallConfig()
        self.model = Seq2TreeModel(config)

        self.types_criterion = MaskedLoss(pad_idx=self.vocab.types_vocab.pad_idx,
            oov_idx=self.vocab.types_vocab.unk_idx, empty_idx=self.vocab.types_vocab.empty_idx) if not self.args.only_values else None
        self.values_criterion = MaskedLoss(pad_idx=self.vocab.values_vocab.pad_idx,
            oov_idx=self.vocab.values_vocab.unk_idx, empty_idx=self.vocab.values_vocab.empty_idx)




