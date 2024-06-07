import torch
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GraphConv, GATv2Conv, global_mean_pool
from torch.nn import Linear, ReLU,CrossEntropyLoss

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = 32
        self.num_heads = 1

        self.conv_first = GATv2Conv(num_node_features, self.hidden_channels, heads=self.num_heads, concat=False, edge_dim=1)
        self.conv1 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=False, edge_dim=1)
        self.conv_last = GATv2Conv(self.hidden_channels * self.num_heads, output_dim, heads=1, concat=False, edge_dim=1)
        self.linear_layer = Linear(self.output_dim, self.output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input x contains NaN or Inf values")
        if torch.isnan(edge_index).any() or torch.isinf(edge_index).any():
            print("Input edge_index contains NaN or Inf values")
    
    

        x = self.conv_first(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv_last(x, edge_index)
        
        x = global_mean_pool(x, data.batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.linear_layer(x)
        x = x.squeeze(1)

        print("IN GNN MODEL: ", x)
        
        return x
    










# x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weights
# x = self.conv_first(x, edge_index, edge_weights)
# x = F.leaky_relu(x)
# x = F.dropout(x, training=self.training)
# x = self.conv1(x, edge_index, edge_weights)
# x = F.leaky_relu(x)
# x = F.dropout(x, training=self.training)
# x = self.conv_last(x, edge_index, edge_weights)
# x = global_mean_pool(x, data.batch)
# x = F.dropout(x, p=0.5, training=self.training)