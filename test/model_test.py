"""
Testing program for network model

TODO: apply pytest for auto testing
"""

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

from typing import Dict

import torch
import torch.nn.functional as F

from modules import model, utils


def get_test_data(args) -> Dict[str, torch.Tensor]:
    data = dict()

    node_num = 6
    node_dim = 16
    
    data["node"] = torch.rand((node_num, node_dim)).to(args.device)
    data["node_label"] = torch.tensor([i % 2 for i in range(node_num)]).to(args.device)

    edge_num = node_num * (node_num - 1)
    edge_src_tmp = []
    edge_dst_tmp = []
    edge_label_tmp = []
    for i in range(node_num):
        for j in range(i + 1, node_num):
            edge_src_tmp.append(i)
            edge_dst_tmp.append(j)
            edge_label_tmp.append(0)

            edge_src_tmp.append(j)
            edge_dst_tmp.append(i)
            edge_label_tmp.append(1)

    data["edge"] = torch.tensor([edge_src_tmp, edge_dst_tmp]).to(args.device)
    data["edge_label"] = torch.tensor(edge_label_tmp).to(args.device)

    return data

def test_conv(args):
    # TODO: separate into several smaller test?
    
    data = get_test_data(args)

    in_dim = data["node"].shape[1]
    out_dim = in_dim

    # create model
    conv = model.GraphAttentionConv(in_dim, out_dim).to(args.device)

    # forward
    x = conv(data["node"], data["edge"])

    # backward
    # TODO: what should the loss be?
    loss = (x - data["node"]).abs().mean()
    loss.backward()

    print("test_conv no error")    

def test_conv_bi(args):
    # TODO: separate into several smaller test?
    
    data = get_test_data(args)

    in_dim = data["node"].shape[1]
    out_dim = in_dim

    # create model
    conv = model.GraphAttentionConv((in_dim, in_dim), out_dim).to(args.device)

    # forward
    x = conv((data["node"], data["node"]), data["edge"])

    # backward
    # TODO: what should the loss be?
    loss = (x - data["node"]).abs().mean()
    loss.backward()

    print("test_conv no error")   

def test_encoder(args):
    # TODO: separate into several smaller test?
    
    data = get_test_data(args)

    in_dim = data["node"].shape[1]
    hid_dim = in_dim
    out_dim = in_dim

    # create model
    enc = model.GraphEncoder(in_dim, hid_channels=hid_dim, out_channels=out_dim).to(args.device)

    # forward
    x = enc(data["node"], data["edge"])

    # backward
    # TODO: what should the loss be?
    loss = (x[-1] - data["node"]).abs().mean()
    loss.backward()

    print("test_encoder no error")

def test_model(args):
    # TODO: separate into several smaller test?
    
    data = get_test_data(args)

    in_dim = data["node"].shape[1]
    hid_dim = in_dim
    fea_dim = in_dim

    # create model
    net = model.SHGCRecom(in_dim, hid_channels=hid_dim, fea_channels=fea_dim, heads=1).to(args.device)

    # forward
    edges = [data["edge"], data["edge"]]
    x = net(data["node"], edges, edges, data["edge"])

    # backward
    # TODO: what should the loss be?
    loss = F.cross_entropy(x["pred"], data["edge_label"])
    loss.backward()

    print("test_encoder no error")


if __name__ == "__main__":
    utils.set_random_seed(42)

    args = SimpleNamespace()
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    test_conv(args)
    test_encoder(args)
    test_model(args)

    test_conv_bi(args)