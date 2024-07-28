from __future__ import absolute_import, division, print_function


import time
import torch.optim as optim
from torch.utils.data import DataLoader


import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from options import LiteMonoOptions
class test:
    def __init__(self, options):
        self.opt = options
        self.models = {}
        self.models["encoder"] = networks.LiteMono(model=self.opt.model,
                                                   drop_path_rate=self.opt.drop_path)
        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
    def test(self):
        input = torch.ones(12,3,192,640)
        features = self.models["encoder"](input)
        # print(features[0].size())
        # print(features[1].size())
        # print(features[2].size())
        #print(self.models["encoder"])
        #print(self.models["depth"])
        outputs = self.models["depth"](features)
        # print(outputs)



options = LiteMonoOptions()
opts = options.parse()
if __name__ == "__main__":
    trainer = test(opts)
    train = trainer.test()

