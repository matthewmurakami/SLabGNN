import torch
import pickle
import train_model
import gnn_model

if __name__ == "__main__":
    import argparse

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

    num_classes = args.num_classes

    # define the datasets
    with open(train_image_filepath, 'rb') as f:
        train_dataset = pickle.load(f)
    
    with open(val_image_filepath, 'rb') as f:
        val_dataset = pickle.load(f)

    # # define the datasets
    # f = open( train_image_filepath, 'rb')
    # train_dataset = pickle.load(f)
    # f.close()
    # f = open( val_image_filepath, 'rb')
    # val_dataset = pickle.load(f)
    # f.close()

    # define the model
    num_features = train_dataset[0].x.shape[1]
    model = gnn_model.GCN(num_features, 1)
    train_model.train(train_dataset, val_dataset, model, output_filepath, learning_rate, early_stopping_count, early_stopping_loss_eps)

    # model = gnn_model.GCN(train_dataset[0].x[0].shape[0], num_classes)
    #train_model.train(train_dataset, val_dataset, model, output_filepath, num_classes, learning_rate, early_stopping_count, early_stopping_loss_eps)    
