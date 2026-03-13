import os
import torch
from torch_geometric.data import HeteroData , Data
import torch_geometric.transforms as T
import dataset.Augmentation as A
import dataset.filmsdataset as ds

def gen_data(config):
    processed_data = [os.path.join(config.data_path+'/processed',file) for file in os.listdir(config.data_path+'/processed')]
    if processed_data is not None:
        print('clean processed data for testing.')
        for data in processed_data:
            try:
                os.remove(data)
                print(f'remove {data}')
            except OSError as e:
                print(f"Error:{ e.strerror}")
    films = ds.MyFilmsDataset(config.data_path)
    return

aug = {
        'Sign' : {
            'name' : 'Sign_Perturbation',
            'func' : A.Sign_Perturbation,
        },
        'Con' : {
            'name' : 'Connectivity_Perturbation',
            'func' : A.Connectivity_Perturbation,
        },
}


def aug_and_split_graph(graph : HeteroData, augmentations : bool = True, prob=0.5):
    if augmentations:
        sign_aug = A.Sign_Perturbation(graph=A.Graph_Copy(graph) , SignPerturbationRate = prob)
        # sign_aug = A.Connectivity_Perturbation(graph=A.Graph_Copy(graph) , EdgeRemovingRate = prob)
        # connect_aug = A.Sign_Perturbation(graph=A.Graph_Copy(graph) , SignPerturbationRate = prob)
        connect_aug = A.Connectivity_Perturbation(graph=A.Graph_Copy(graph) , EdgeRemovingRate = prob)
    else:
        sign_aug = graph=A.Graph_Copy(graph)
        connect_aug = graph=A.Graph_Copy(graph)

    sign_pos , sign_neg = A.Seperate_Pos_and_Neg_Edge(sign_aug)
    connect_pos , connect_neg = A.Seperate_Pos_and_Neg_Edge(connect_aug)

    return sign_pos , sign_neg , connect_pos , connect_neg


def get_graph(config):
    films = ds.MyFilmsDataset(root=config.data_path)
    train_ds = films.get('Train')
    test_ds = films.get('Test')

    return train_ds.to(config.device) , test_ds.to(config.device)

def load_data(train_graph , config):
    train_graph_copy = train_graph.clone()

    aug_prob = config.aug_prob
    aug = config.train_aug

    SP_pos , SP_neg , CP_pos , CP_neg  = aug_and_split_graph(graph = train_graph, augmentations=aug, prob=aug_prob)
    SP_pos , SP_neg , CP_pos , CP_neg  = T.ToUndirected()(SP_pos) , T.ToUndirected()(SP_neg) , T.ToUndirected()(CP_pos) , T.ToUndirected()(CP_neg)
    SP_pos , SP_neg , CP_pos , CP_neg  = SP_pos.to_homogeneous() , SP_neg.to_homogeneous() , CP_pos.to_homogeneous() , CP_neg.to_homogeneous() 
    
    train_graph = train_graph.to_homogeneous()
    edge_pos = [SP_pos.edge_index, CP_pos.edge_index]
    edge_neg = [SP_neg.edge_index, CP_neg.edge_index]
    pred_edge = train_graph.edge_index
    node = train_graph.x
    label = ((train_graph.edge_attr + 2.5) * 2).long()

    if not config.perm_contrastive:        
        return node, edge_pos, edge_neg, pred_edge, label

    perm_graph, perm = A.Preference_Permutation(train_graph_copy)
    SP_pos_perm , SP_neg_perm , CP_pos_perm , CP_neg_perm  = aug_and_split_graph(graph=perm_graph, augmentations=aug, prob=aug_prob)
    SP_pos_perm , SP_neg_perm , CP_pos_perm , CP_neg_perm  = T.ToUndirected()(SP_pos_perm) , T.ToUndirected()(SP_neg_perm) , T.ToUndirected()(CP_pos_perm) , T.ToUndirected()(CP_neg_perm)
    SP_pos_perm , SP_neg_perm , CP_pos_perm , CP_neg_perm  = SP_pos_perm.to_homogeneous() , SP_neg_perm.to_homogeneous() , CP_pos_perm.to_homogeneous() , CP_neg_perm.to_homogeneous() 

    edge_pos_perm = [SP_pos_perm.edge_index, CP_pos_perm.edge_index]
    edge_neg_perm = [SP_neg_perm.edge_index, CP_neg_perm.edge_index]

    people_num = train_graph_copy['people'].num_nodes

    return node, edge_pos, edge_neg, pred_edge, label, edge_pos_perm, edge_neg_perm, people_num, perm

def load_test_data(train_graph ,test_graph ,config):
    aug_prob =config.aug_prob
    aug = config.test_aug

    SP_pos , SP_neg , CP_pos , CP_neg  = aug_and_split_graph(graph = train_graph, augmentations = aug, prob=aug_prob)
    SP_pos , SP_neg , CP_pos , CP_neg  = T.ToUndirected()(SP_pos) , T.ToUndirected()(SP_neg) , T.ToUndirected()(CP_pos) , T.ToUndirected()(CP_neg)
    SP_pos , SP_neg , CP_pos , CP_neg  = SP_pos.to_homogeneous() , SP_neg.to_homogeneous() , CP_pos.to_homogeneous() , CP_neg.to_homogeneous() 
    
    train_graph = T.ToUndirected()(train_graph)
    train_graph = train_graph.to_homogeneous()
    test_graph = test_graph.to_homogeneous()
    
    node = train_graph.x
    edge_pos = [SP_pos.edge_index, CP_pos.edge_index]
    edge_neg = [SP_neg.edge_index, CP_neg.edge_index]
    pred_edge = test_graph.edge_index
    label = ((test_graph.edge_attr + 2.5) * 2).long()
    
    return node ,edge_pos ,edge_neg ,pred_edge ,label