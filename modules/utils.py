from ast import Dict
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random
from types import SimpleNamespace
from torch_geometric.data import HeteroData

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from modules.model import SHGCRecom
from modules.data_util import get_graph , load_data ,load_test_data
from modules.Loss import Contrastive_Loss

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_and_test_model(config):
    train_graph , test_graph = get_graph(config)

    net = SHGCRecom(in_channels=config.feature_dim,
                    hid_channels=config.hid_dim,
                    fea_channels=config.out_dim,
                    heads=config.head_nums,
                    conv_num=config.conv_num,
                    class_num=config.class_num).to(config.device)
    
    model, history = train(model=net, train_graph=train_graph, config=config)

    acc = eval(model = model ,train_graph = train_graph ,test_graph = test_graph ,config=config)

    return

def train(model: SHGCRecom, train_graph: HeteroData, config):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    history = dict()
    history['loss_cross'] = []
    history['loss_contras'] = []
    history['loss'] = []
    for epoch in tqdm.tqdm(range(config.epochs)):
        opt.zero_grad()

        if config.perm_contrastive:
            node , edge_pos ,edge_neg , pred_edge ,label, edge_pos_perm, edge_neg_perm, people_num, perm = load_data(train_graph=train_graph ,config=config)
            x = model(node , edge_pos ,edge_neg , pred_edge)
            x_perm = model(node, edge_pos_perm, edge_neg_perm, pred_edge)
            loss_cross = F.cross_entropy(x["pred"], label)
            loss_contras = Contrastive_Loss(config, x, x_perm, people_num, perm)

        else:
            node , edge_pos ,edge_neg , pred_edge ,label = load_data(train_graph=train_graph ,config=config)
            x = model(node , edge_pos ,edge_neg , pred_edge)
            loss_cross = F.cross_entropy(x["pred"], label)
            loss_contras = Contrastive_Loss(config, x)

        loss = loss_cross + loss_contras * config.contra_loss_w
        # loss = torch.tensor(loss_contras)         
        loss.backward()
        opt.step()

        history['loss_cross'].append(loss_cross.item())
        history['loss_contras'].append(loss_contras.item())
        history['loss'].append(loss.item())
    if config.SAVE_MODE:
        torch.save(model.state_dict() , os.path.join(config.save_path , 'model.h5'))
    return model , history


def eval(model : SHGCRecom,train_graph: HeteroData, test_graph : HeteroData, config):
    model.eval()
    acc = dict()

    if config.perm_contrastive:
        node , edge_pos ,edge_neg , pred_edge ,train_label, _, _, _, _ = load_data(train_graph=train_graph ,config=config)
    else:
        node , edge_pos ,edge_neg , pred_edge ,train_label = load_data(train_graph=train_graph ,config=config)
    train_x = model(node , edge_pos ,edge_neg , pred_edge)

    node , edge_pos ,edge_neg ,pred_edge ,test_label = load_test_data(train_graph=train_graph ,test_graph=test_graph ,config=config)
    test_x = model(node , edge_pos ,edge_neg , pred_edge)

    train_correct = int(train_x['pred'].max(dim=1).indices.eq(train_label).sum().item())
    train_acc = train_correct / len(train_label)

    train_rat_pred = train_x['pred'].max(dim=1).indices / 2
    train_rat = train_label / 2
    train_mse = F.mse_loss(train_rat_pred, train_rat)

    # acc['train']
    test_correct = int(test_x['pred'].max(dim=1).indices.eq(test_label).sum().item())
    test_acc = test_correct / len(test_label)

    test_rat_pred = test_x['pred'].max(dim=1).indices / 2
    test_rat = test_label / 2
    test_mse = F.mse_loss(test_rat_pred, test_rat)

    print(f'Top1 Accuracy of training data : {train_acc}')
    print(f'Mean Square Error of training data : {train_mse}')
    print(f'Top1 Accuracy of testing data : {test_acc}')
    print(f'Mean Square Error of testing data : {test_mse}')

    return acc