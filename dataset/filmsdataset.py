from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from torch_geometric.data import Dataset, InMemoryDataset, download_url,  HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
try:
    import Augmentation as A
except ImportError:
    import dataset.Augmentation as A
from modules.utils import set_random_seed

def Get_film_data(filename):
    # read csv
    data = pd.read_csv(filename,sep=' ',header=None)
    data.columns=['people','object','rating']

    # define parameter dim
    num_of_people = len(data['people'].unique())
    nun_of_obj = len(data['object'].unique())
    num_people_fea = 64
    num_obj_fea = 64

    node_fea_poeple = torch.rand((num_of_people,num_people_fea))
    node_fea_obj = torch.rand((nun_of_obj,num_obj_fea))
    edg_people_to_obj = torch.tensor(np.array([data['people'] , data['object']])) - 1
    edg_attr = torch.tensor(np.array(data['rating'] - 2.5))
    # construct graph
    graph = HeteroData()

    graph['people'].x = node_fea_poeple
    graph['object'].x = node_fea_obj

    graph['people','rate','object'].edge_index = edg_people_to_obj

    graph['people','rate','object'].edge_attr = edg_attr

    return graph


def graph_split(graph : HeteroData , split_ration : float = 0.8):
    train_graph , test_graph = HeteroData() , HeteroData()
    people , object = graph['people'].x.numpy() , graph['object'].x.numpy()
    edge_index , edge_attr = graph['people' , 'rate' , 'object'].edge_index , graph['people' , 'rate' , 'object'].edge_attr
    edge_index , edge_attr = edge_index.numpy().transpose() , edge_attr.numpy().transpose()

    trian_index , test_index , train_attr , test_attr = train_test_split(edge_index , edge_attr,train_size=split_ration)

    trian_index , test_index , train_attr , test_attr = torch.tensor(trian_index.transpose()) , torch.tensor(test_index.transpose()) ,torch.tensor(train_attr) , torch.tensor(test_attr) 
    train_graph['people'].x , test_graph['people'].x = torch.tensor(people.copy()) , torch.tensor(people.copy())
    train_graph['object'].x , test_graph['object'].x = torch.tensor(object.copy()) ,torch.tensor(object.copy()) 

    train_graph['people','rate','object'].edge_index = trian_index
    train_graph['people','rate','object'].edge_attr = train_attr

    test_graph['people','rate','object'].edge_index = test_index
    test_graph['people','rate','object'].edge_attr = test_attr

    return train_graph , test_graph


class MyFilmsDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['ratings.txt']

    @property
    def processed_file_names(self):
        return ['Original.pt', 'Train.pt', 'Test.pt']
    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            # print(raw_path.split('/')[-1].split('.')[0])
            # data_name = raw_path.split('/')[-1].split('.')[0]
            data = Get_film_data(raw_path)
            torch.save(data, os.path.join(self.processed_dir, 'Original.pt'))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            train_graph , test_graph = graph_split(graph = data , split_ration = 0.8)
            torch.save(train_graph , os.path.join(self.processed_dir , 'Train.pt'))
            torch.save(test_graph , os.path.join(self.processed_dir , 'Test.pt'))


    def len(self):
        return len(self.processed_file_names)

    def get(self, type : str = 'Original'):
        data = torch.load(os.path.join(self.processed_dir, f'{type}.pt'))
        return data

if __name__ == "__main__":
    set_random_seed(42)
    print(f'Test for heteograph dataset.')

    data_path = 'data/films'
    processed_data = [os.path.join(data_path+'/processed',file) for file in os.listdir(data_path+'/processed')]
    if processed_data is not None:
        print('clean processed data for testing.')
        for data in processed_data:
            try:
                os.remove(data)
                print(f'remove {data}')
            except OSError as e:
                print(f"Error:{ e.strerror}")


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
    
    films = MyFilmsDataset(data_path)
    ori = films.get('Original')
    train = films.get('Train')
    test = films.get('Test')
    print(ori['rate'])
    print(train['rate'])
    print(test['rate'])