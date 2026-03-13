import torch
import random
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.utils import coalesce

def Connectivity_Perturbation(graph: HeteroData, EdgeRemovingRate):
    # # print("----------In Connectivity_Perturbation----------")
    # # print(graph['people','rate','object'].edge_index)
    # # print(graph['people','rate','object'].edge_attr)
    # positive_links_before = 0
    # negative_links_before = 0
    # positive_links_after = 0
    # negative_links_after = 0

    # original_edge_index = graph['people','rate','object'].edge_index
    # for i in graph['people','rate','object'].edge_attr :
    #     if i >= 0 :
    #         positive_links_before = positive_links_before + 1
    #     else:
    #         negative_links_before = negative_links_before + 1
    # x = 0
    # index0 = graph['people','rate','object'].edge_index[0]
    # index1 = graph['people','rate','object'].edge_index[1]
    # for i in range(len(graph['people','rate','object'].edge_attr)):
    #     if random.random() < EdgeRemovingRate:
    #         graph['people','rate','object'].edge_attr = torch.cat([graph['people','rate','object'].edge_attr[0:i - x], graph['people','rate','object'].edge_attr[i - x+1:]])
    #         index0 = torch.cat([index0[0:i - x], index0[i - x+1:]])
    #         index1 = torch.cat([index1[0:i - x], index1[i - x+1:]])
    #         x = x + 1
    # index0 = index0.numpy()
    # index1 = index1.numpy()
    # # graph['people','rate','object'].edge_index = torch.tensor(np.array([index0, index1]) ,dtype=torch.float)
    # graph['people','rate','object'].edge_index = torch.tensor(np.array([index0, index1]))                
    # for i in graph['people','rate','object'].edge_attr :
    #     if i >= 0 :
    #         positive_links_after = positive_links_after + 1
    #     else:
    #         negative_links_after = negative_links_after + 1
    # # print(positive_links_before, negative_links_before, positive_links_after, negative_links_after)
    # if positive_links_before - positive_links_after > 0:
    #     new_element = torch.rand(positive_links_before - positive_links_after)
    #     for i in range(len(new_element)):
    #         new_element[i] = new_element[i] * 2.5
    #     graph['people','rate','object'].edge_attr = torch.cat((graph['people','rate','object'].edge_attr, new_element))
    #     index0 = graph['people','rate','object'].edge_index[0]
    #     index1 = graph['people','rate','object'].edge_index[1]
    #     for i in range(positive_links_before - positive_links_after):
    #         notice = 0
    #         while(notice == 0):
    #             notice = 1
    #             new_edge0 = torch.rand(1)
    #             new_edge1 = torch.rand(1)
    #             new_edge0 = (new_edge0 * 5 + 1).int()
    #             new_edge1 = (new_edge1 * 5 + 1).int()
    #             for j in range(len(original_edge_index[0])):
    #                 if new_edge0 == original_edge_index[0][j] and new_edge1 == original_edge_index[1][j]:
    #                     notice = 0
    #                 elif new_edge0 == new_edge1:
    #                     notice = 0

    #         index0 = torch.cat((index0, new_edge0))
    #         index1 = torch.cat((index1, new_edge1))

    #     index0 = index0.numpy()
    #     index1 = index1.numpy()
    #     # graph['people','rate','object'].edge_index = torch.tensor(np.array([index0, index1]) ,dtype=torch.float)
    #     graph['people','rate','object'].edge_index = torch.tensor(np.array([index0, index1]))            

    # if negative_links_before - negative_links_after > 0:
    #     new_element = torch.rand(negative_links_before - negative_links_after)
    #     for i in range(len(new_element)):
    #         new_element[i] = new_element[i] * -2.5
    #     graph['people','rate','object'].edge_attr = torch.cat((graph['people','rate','object'].edge_attr, new_element))
    #     index0 = graph['people','rate','object'].edge_index[0]
    #     index1 = graph['people','rate','object'].edge_index[1]
    #     for i in range(negative_links_before - negative_links_after):
    #         notice = 0
    #         while(notice == 0):
    #             notice = 1
    #             new_edge0 = torch.rand(1)
    #             new_edge1 = torch.rand(1)
    #             new_edge0 = (new_edge0 * 5 + 1).int()
    #             new_edge1 = (new_edge1 * 5 + 1).int()
    #             for j in range(len(original_edge_index[0])):
    #                 if new_edge0 == original_edge_index[0][j] and new_edge1 == original_edge_index[1][j]:
    #                     notice = 0
    #                 elif new_edge0 == new_edge1:
    #                     notice = 0

    #         index0 = torch.cat((index0, new_edge0))
    #         index1 = torch.cat((index1, new_edge1))

    #     index0 = index0.numpy()
    #     index1 = index1.numpy()
    #     # graph['people','rate','object'].edge_index = torch.tensor(np.array([index0, index1]) ,dtype=torch.float)
    #     graph['people','rate','object'].edge_index = torch.tensor(np.array([index0, index1]))
    # # print("-------------Output of Connectivity_Perturbation-------------------")
    # # print(graph['people','rate','object'].edge_index)
    # # print(graph['people','rate','object'].edge_attr)
    # return graph
    
    # TODO: make this not hard coded
    people_num = graph['people'].num_nodes
    object_num = graph['object'].num_nodes
    edge_num = graph['people','rate','object'].num_edges

    new_graph = graph.clone()
    edge_index = new_graph['people','rate','object'].edge_index
    edge_attr = new_graph['people','rate','object'].edge_attr

    # drop edges 
    mask = torch.rand(edge_num, device=edge_index.device) < EdgeRemovingRate
    p = (edge_attr[mask] >= 0).sum()

    q = (edge_attr[mask] < 0).sum()

    edge_index = edge_index[:, ~mask]
    edge_attr = edge_attr[~mask]

    # add back edges
    # modified from PyGCL
    # this is not the correct implementations since it does not exclude existing edges
    # however, much faster
    # checking existing edges will be hell slow
    num_add = p + q

    new_edge_index_src = torch.randint(0, people_num, size=(1, num_add)).to(edge_index.device)
    new_edge_index_dst = torch.randint(0, object_num, size=(1, num_add)).to(edge_index.device)
    new_edge_index = torch.cat([new_edge_index_src, new_edge_index_dst])

    # TODO: make this parameters
    RATING_LEVEL = 5 
    new_edge_attr_pos = torch.randint(low=0, high=RATING_LEVEL, size=(p,)).to(edge_index.device) * 0.5
    new_edge_attr_neg = torch.randint(low=-RATING_LEVEL, high=-1, size=(q,)).to(edge_index.device) * 0.5
    new_edge_attr = torch.cat([new_edge_attr_pos, new_edge_attr_neg])

    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, new_edge_attr])

    # remove dulplicate edges
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=(people_num + object_num), reduce="max")

    new_graph['people','rate','object'].edge_index = edge_index
    new_graph['people','rate','object'].edge_attr = edge_attr

    return new_graph


def Sign_Perturbation(graph: HeteroData, SignPerturbationRate: float):
    # for i in range(len(graph['people','rate','object'].edge_attr)):
    #     if random.random() < SignPerturbationRate:
    #         graph['people','rate','object'].edge_attr[i] = graph['people','rate','object'].edge_attr[i] * -1

    new_graph = graph.clone()
    mask = torch.rand(new_graph['people','rate','object'].num_edges, device=new_graph['people','rate','object'].edge_attr.device) < SignPerturbationRate
    new_graph['people','rate','object'].edge_attr[mask] = new_graph['people','rate','object'].edge_attr[mask] * (-1)
    
    return new_graph

def map_perm(val: torch.Tensor, perm: torch.Tensor):
    mapping = {k: v.item() for k, v in enumerate(perm)}

    # From `https://stackoverflow.com/questions/13572448`.
    palette, key = zip(*mapping.items())
    key = torch.tensor(key, device=val.device)
    palette = torch.tensor(palette, device=val.device)

    index = torch.bucketize(val.ravel(), palette)
    remapped = key[index].reshape(val.shape)

    return remapped

def Preference_Permutation(graph: HeteroData):
    """
    Randomly permute the edgeindex of the whole user type node
    That is, all the source index of an user node i become another user node j
    """
    new_graph = graph.clone()

    people_num = graph['people'].num_nodes
    edge_index = new_graph['people','rate','object'].edge_index

    perm = torch.randperm(people_num, device=edge_index.device)
    new_graph['people','rate','object'].edge_index[0] = map_perm(edge_index[0], perm).to(edge_index.device)

    return new_graph, perm

def Graph_Copy(graph : HeteroData):
    # graph_copy = HeteroData()
    # graph_copy['people'].x = torch.tensor(graph['people'].x.numpy().copy())
    # graph_copy['object'].x = torch.tensor(graph['object'].x.numpy().copy())
    # # graph_copy['people'].x = torch.tensor(graph['people'].x.numpy().copy() , dtype = torch.float16)
    # # graph_copy['object'].x = torch.tensor(graph['object'].x.numpy().copy() , dtype = torch.float16)
    # # graph_copy['people','rate','object'].edge_index = torch.tensor(graph['people','rate','object'].edge_index.numpy().copy() , dtype = torch.float16) 
    # # graph_copy['people','rate','object'].edge_attr = torch.tensor(graph['people','rate','object'].edge_attr.numpy().copy() , dtype = torch.float16)
    # graph_copy['people','rate','object'].edge_index = torch.tensor(graph['people','rate','object'].edge_index.numpy().copy()) 
    # graph_copy['people','rate','object'].edge_attr = torch.tensor(graph['people','rate','object'].edge_attr.numpy().copy())
    # return graph_copy

    return graph.clone()

def Seperate_Pos_and_Neg_Edge(graph):
    graph_pos = Graph_Copy(graph)
    graph_neg = Graph_Copy(graph)
    # pos_index = [[], []]
    # pos_attr = []
    # neg_index = [[], []]
    # neg_attr = []
    # # print("In Augmentation : ", graph['people','rate','object'].edge_attr)
    # # print(graph['people','rate','object'].edge_index[0])
    # # print(graph['people','rate','object'].edge_index[1])
    # for i in range(len(graph['people','rate','object'].edge_attr)):
    #     if graph['people','rate','object'].edge_attr[i] > 0:
    #         pos_attr.append(graph['people','rate','object'].edge_attr[i])
    #         pos_index[0].append(graph['people','rate','object'].edge_index[0][i])
    #         pos_index[1].append(graph['people','rate','object'].edge_index[1][i])
    #     else:
    #         neg_attr.append(graph['people','rate','object'].edge_attr[i])
    #         neg_index[0].append(graph['people','rate','object'].edge_index[0][i])
    #         neg_index[1].append(graph['people','rate','object'].edge_index[1][i])

    # graph_pos['people','rate','object'].edge_attr = torch.tensor(pos_attr)
    # graph_neg['people','rate','object'].edge_attr = torch.tensor(neg_attr)
    # graph_pos['people','rate','object'].edge_index = torch.tensor(pos_index)
    # graph_neg['people','rate','object'].edge_index = torch.tensor(neg_index)

    pos_mask = graph['people','rate','object'].edge_attr >= 0

    graph_pos['people','rate','object'].edge_index = graph['people','rate','object'].edge_index[:, pos_mask]
    graph_pos['people','rate','object'].edge_attr = graph['people','rate','object'].edge_attr[pos_mask]

    graph_neg['people','rate','object'].edge_index = graph['people','rate','object'].edge_index[:, ~pos_mask]
    graph_neg['people','rate','object'].edge_attr = graph['people','rate','object'].edge_attr[~pos_mask]

    return graph_pos, graph_neg
