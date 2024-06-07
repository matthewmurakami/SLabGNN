import torch
import pickle
import train_model
import gnn_model
import argparse
from torch_geometric.data import Data, DataLoader

def load_dataset_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Ensure the dataset is a list of Data objects
    if isinstance(dataset, list) and all(isinstance(d, Data) for d in dataset):
        return dataset
    else:
        raise ValueError("The loaded dataset is not in the expected format (list of Data objects).")


if __name__ == "__main__":
    train_pickle_file = '../data/train_dataset.pkl'
    val_pickle_file = '../data/train_dataset.pkl'
    
    train_dataset = load_dataset_from_pickle(train_pickle_file)
    val_dataset = load_dataset_from_pickle(val_pickle_file)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    num_features = train_dataset[0].x.shape[1]  # Number of features per node
    num_targets = train_dataset[0].y.shape[0]  # Number of target values (2: 'ter.Mean' and 'ratio.Mean')

    model = gnn_model.GCN(num_features, num_targets)

    output_filepath = 'model_output/Abs_model.pth'
    learning_rate = 1e-4
    early_stopping_count = 10
    early_stopping_loss_eps = 1e-3
    
    train_model.train(train_loader, val_loader, model, output_filepath, learning_rate, early_stopping_count, early_stopping_loss_eps)



    """if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a single model')
    parser.add_argument('--train-image-filepath', type=str, required=True,
                        help='Path to the input file that stores the training graphs')
   
    parser.add_argument('--val-image-filepath', type=str, required=True,
                        help='Path to the input file that stores the validation graphs')
   
    parser.add_argument('--output-filepath', type=str, required=True,
                        help='Path to the folder/directory where the results should be stored')
                        
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--early-stopping-count', type=int, default=10, help='The number of epochs past the current best loss to continue training in search of a better final loss.')
    parser.add_argument('--early-stopping-loss-eps', type=float, default=1e-3, help="The loss epsilon (delta below which loss values are considered equal) when handling early stopping.")

    parser.add_argument('--num-classes', type=int, default=2, help='The number of classes the model is to predict.')

    args = parser.parse_args()
    train_image_filepath = args.train_image_filepath
    val_image_filepath = args.val_image_filepath

    output_filepath = args.output_filepath
    learning_rate = args.learning_rate
    early_stopping_count = args.early_stopping_count
    early_stopping_loss_eps = args.early_stopping_loss_eps

    f = open( train_image_filepath, 'rb')
    train_dataset = pickle.load(f)
    f.close()
    f = open( val_image_filepath, 'rb')
    val_dataset = pickle.load(f)
    f.close()

    # define the model
    num_features = train_dataset[0].x.shape[1]
    model = gnn_model.GCN(num_features, 1)
    train_model.train(train_dataset, val_dataset, model, output_filepath, learning_rate, early_stopping_count, early_stopping_loss_eps)
"""
