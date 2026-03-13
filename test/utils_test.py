from ast import Pass
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

from typing import Dict

import torch
import torch.nn.functional as F

from modules import model, utils, data_util
from torch_geometric.data import HeteroData , Data
import torch_geometric.transforms as T

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

def test_sep_graph():
    data_path = os.path.join(os.path.dirname(__file__),'..','dataset','data','films')
    print(f'data path : {data_path}')
    train_graph , test_graph = data_util.get_graph(root = data_path)
    
    print('train_g : /n',train_graph)
    print('test_g : /n',test_graph)

    SP_pos , SP_neg , CP_pos , CP_neg  = data_util.aug_and_split_graph(graph = train_graph)
    # print('graph')
    # print(SP_pos , SP_neg , CP_pos , CP_neg )
    # print('rate')
    # print(SP_pos['rate'])
    # print(SP_neg['rate'])
    # print(CP_pos['rate'])
    # print(CP_neg['rate'])

def load_data(args) -> Dict[ str , Data]:
    data_path = os.path.join(os.path.dirname(__file__),'..','dataset','data','films')
    print(f'data path : {data_path}')
    train_graph , test_graph = data_util.get_graph(root = data_path)
    train_graph = train_graph.to(args.device)
    test_graph = test_graph.to(args.device)

    SP_pos , SP_neg , CP_pos , CP_neg  = data_util.aug_and_split_graph(graph = train_graph)
    SP_pos , SP_neg , CP_pos , CP_neg  = T.ToUndirected()(SP_pos) , T.ToUndirected()(SP_neg) , T.ToUndirected()(CP_pos) , T.ToUndirected()(CP_neg)
    SP_pos , SP_neg , CP_pos , CP_neg  = SP_pos.to_homogeneous() , SP_neg.to_homogeneous() , CP_pos.to_homogeneous() , CP_neg.to_homogeneous() 
    data = dict()
    data['SP_pos'] = SP_pos
    data['SP_neg'] = SP_neg
    data['CP_pos'] = CP_pos
    data['CP_neg'] = CP_neg
    """
    x : node feature
    edge_index : [2 , num_node]
    dege_attr : [num_node , 1]
    node_type : ... [ 0 , 0 , 0, 1, 1, 1, 1]
    edge_type : ...
    """
    return data

def test_conv(args):
    data = load_data(args)
    
    in_dim = data['SP_pos'].x.shape[1]
    out_dim = in_dim

    
    # create model
    conv = model.GraphAttentionConv(in_dim, out_dim).to(args.device)

    # forward
    x = conv(data['SP_pos'].x.to(args.device), data['SP_pos'].edge_index.to(args.device))

    loss = (x - data['SP_pos'].x.to(args.device)).abs().mean()
    loss.backward()

    print("test_conv no error")   


def test_conv_bi(args):
    data = load_data(args)

    in_dim = data['SP_pos'].x.shape[1]
    out_dim = in_dim

    # create model
    conv = model.GraphAttentionConv((in_dim, in_dim), out_dim).to(args.device)

    # forward
    # x = conv((data["node"], data["node"]), data["edge"])
    x = conv((data['SP_pos'].x, data['SP_pos'].x), data['SP_pos'].edge_index)

    # backward
    # TODO: what should the loss be?
    loss = (x - data['SP_pos'].x).abs().mean()
    loss.backward()

    print("test_conv no error") 

def test_encoder(args):
    # TODO: separate into several smaller test?
    
    data = load_data(args)
    data['SP_pos'] = data['SP_pos'].to(args.device)

    data0 = get_test_data(args)
    # in_dim = data["node"].shape[1]
    in_check : bool = (data['SP_pos'].x.shape[1] == data0["node"].shape[1])
    print('data shape',data['SP_pos'].x.shape)
    in_dim = data['SP_pos'].x.shape[1]
    hid_dim = in_dim
    out_dim = in_dim
    print(f' indim : {in_dim}/{hid_dim}/{out_dim}  {in_check}')
    # create model
    enc = model.GraphEncoder(in_dim, hid_channels=hid_dim, out_channels=out_dim , heads=1).to(args.device)

    # forward
    # x0 = enc(data0["node"], data0["edge"])
    
    print('data0\n',data0["node"], data0["edge"])
    print('data : \n',data['SP_pos'].x, data['SP_pos'].edge_index)
    x = enc(data['SP_pos'].x, data['SP_pos'].edge_index)

    # backward
    # TODO: what should the loss be?
    # loss = (x[-1] - data["node"]).abs().mean()
    loss = (x[-1] - data['SP_pos'].x).abs().mean()
    loss.backward()

    print("test_encoder no error")

def test_model(args):
    # TODO: separate into several smaller test?
    
    data = load_data(args)
    data['SP_pos'] = data['SP_pos'].to(args.device)    
    data['SP_neg'] = data['SP_neg'].to(args.device)
    data['CP_pos'] = data['CP_pos'].to(args.device)
    data['CP_neg'] = data['CP_neg'].to(args.device)

    data0 = get_test_data(args)
    # in_dim = data0["node"].shape[1]
    in_check : bool = (data['SP_pos'].x.shape[1] == data0["node"].shape[1])
    in_dim = data["SP_pos"].x.shape[1]
    hid_dim = in_dim
    fea_dim = in_dim
    print(f' indim : {in_dim}/{hid_dim}/{fea_dim}  {in_check}')
    # create model
    net = model.SHGCRecom(in_dim, hid_channels=hid_dim, fea_channels=fea_dim,heads=1).to(args.device)

    # # forward
    edges = [data0["edge"], data0["edge"]]
    # x = net(data["node"], edges, edges, data["edge"])
    
    print(data0["node"].shape,data0["node"])
    print(data["SP_pos"].x.shape,data["SP_pos"].x)
    print('//////////////////////')
    print(data0["edge"].shape,data0["edge"])
    print(data["SP_pos"].edge_index.shape,data["SP_pos"].edge_index)
    edge_pos = [data["SP_pos"].edge_index, data["CP_pos"].edge_index]
    edge_neg = [data["SP_neg"].edge_index, data["CP_neg"].edge_index]
    x = net(data["SP_pos"].x, edge_pos, edge_neg, data["SP_pos"].edge_index)
    # print('dege :\n')
    # print(edges)
        
    print('/////////////////////////////')
    # backward
    # TODO: what should the loss be?
    loss = F.cross_entropy(x["pred"], data["edge_label"])
    loss = F.cross_entropy(x["pred"], data['SP_pos'].edge_attr)
    loss.backward()

    print("test_encoder no error")

if __name__ == "__main__":
    utils.set_random_seed(42)

    args = SimpleNamespace()
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # test_sep_graph()
    # test_conv(args)
    # test_conv_bi(args)
    test_encoder(args)
    test_model(args)