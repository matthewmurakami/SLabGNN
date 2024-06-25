import pickle
import train_model
import test_acc
import gnn_model
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch


def load_dataset_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    if isinstance(dataset, list) and all(isinstance(d, Data) for d in dataset):
        return dataset
    else:
        raise ValueError("The loaded dataset is not a list of Data objects).")

if __name__ == "__main__":    
    train_pickle_file = 'GNN/data/train_dataset.pkl'
    dataset = load_dataset_from_pickle(train_pickle_file)

    train_index = int(len(dataset) * 0.95)

    train_dataset = dataset[:train_index]
    val_dataset = dataset[train_index:]

    batch_size = 20
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_features = train_dataset[0].x.shape[1]  # Number of features per node
    num_targets = train_dataset[0].y.shape[0]  # 2: 'ter.Mean' and 'ratio.Mean'

    model = gnn_model.GCN(num_features, num_targets)
    
    learning_rate = 0.001 #1e-3
    num_epochs = 100

    output_filepath = f'GNN/models/Abs_model_b{batch_size}_e{num_epochs}_lr{learning_rate}.pth'

    # print()
    # print('====================')
    # print(f'Number of graphs: {len(dataset)}')
    # print(f'Number of features: {num_features}')
    # print(f'Number of targets: {num_targets}')
    # data = dataset[0] 
    # print()
    # print(data)
    # print('=============================================================')
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Has self-loops: {data.has_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')
    # print('=============================================================')
    # print()
    # print(f'Number of training graphs: {len(train_dataset)}')
    # print(f'Number of test graphs: {len(val_dataset)}')
    # print('=============================================================')
    # print()
    # # for step, data in enumerate(train_loader):
    # #     print(f'Step {step + 1}:')
    # #     print('=======')
    # #     print(f'Number of graphs in the current batch: {data.num_graphs}')
    # #     print(data)
    # #     print()
    # exit()
    
    #train_model.train_model(train_loader, val_loader, model, output_filepath, learning_rate, num_epochs)


    test_pickle_file = 'GNN/data/test_dataset.pkl'
    model_path = 'GNN\models\Abs_model_b20_e100_lr0.001.pth'

    test_dataset = load_dataset_from_pickle(test_pickle_file)
    test_loader = DataLoader(val_dataset)

    model = gnn_model.GCN(num_features, num_targets)
    model.load_state_dict(torch.load(model_path))
    
    test_acc.test_model(test_loader, model)



