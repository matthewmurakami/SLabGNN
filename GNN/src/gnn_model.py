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
        self.num_heads = 2

        self.conv_first = GATv2Conv(num_node_features, self.hidden_channels, heads=self.num_heads, concat=True)
        self.conv1 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True)
        self.conv_last = GATv2Conv(self.hidden_channels * self.num_heads, output_dim, heads=1, concat=False)


    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weights

        x = self.conv_first(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv1(x, edge_index, edge_weights)
        x = F.leaky_relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv_last(x, edge_index, edge_weights)
        
        x = global_mean_pool(x, data.batch)

        x = F.dropout(x, p=0.5, training=self.training)
        
        return x