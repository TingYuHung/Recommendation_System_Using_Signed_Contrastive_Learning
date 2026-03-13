from argparse import Namespace
import torch

class config(Namespace):
    """mode setting"""
    GEN_DATA : bool = False

    TRAINING_MODE : bool = True

    SAVE_MODE : bool = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    """path"""
    data_path : str = 'dataset/data/films'

    save_path : str = 'tmp/'

    """data paarameter"""
    feature_dim : int = 64
    # graph node feature dimension

    hid_dim : int = 256
    # hidden dimension

    out_dim : int = 256
    # embeding feature dimension

    split_ratio : float = 0.8
    # train/test split ratio

    train_aug : bool = True

    test_aug : bool = False

    aug_prob : float = 0.2

    perm_contrastive : bool = True
    # contrastive learning with preference permutation
    # only works for training

    """model parameter"""
    head_nums = 8
    # number of attention head
    # must divisable for out_dim

    conv_num = 2

    class_num = 11

    """training parameter"""
    epochs : int = 100

    iteration : int = 5

    lr : float = 1e-3

    weight_decay : float = 1e-6 #5e-4

    contra_loss_w = 1e-3

    inter_contra_w = 0.5
    intra_contra_w = 0.5
