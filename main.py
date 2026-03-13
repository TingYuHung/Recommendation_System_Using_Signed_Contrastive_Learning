import os
import sys

import numpy as np
import pandas as pd
import argparse
import torch

from dataset.filmsdataset import MyFilmsDataset
from modules.utils import set_random_seed , train_and_test_model
from modules.data_util import gen_data 
from test.utils_test import test_model
from config import config as cf



if __name__ == "__main__":
    print('This is main function.')
    set_random_seed(42)

    # parser = argparse.ArgumentParser(description="Chose mod")
    # parser.add_argument("-g" ,"--gen" , action="store_true" , help="generate graph data")
    # parser.add_argument("-t" , "--train" , action="store_true" , help="train-mode")
    # parser.add_argument("-e" , "--evaluation" , action="store_true" , help="evaluation-mode")

    # args = parser.parse_args()
    

    if cf.GEN_DATA:
        print("generate graph")
        gen_data(config=cf)


    if cf.TRAINING_MODE:
        """
        train step:
        .....
        """
        train_and_test_model(config=cf)
